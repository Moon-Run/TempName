"""Standalone chunked GEMM / NCCL / tensor-codec mechanism prototype.

This deliberately does not modify Flux's epilogue or remove its IPC barrier.
Fresh invocation allocations and stream dependencies replace reusable remote
flags. Every rank must construct and call instances in the same order.
NCCL stage order is common to all ranks; only destination production within a
stage uses a source-relative ring. This is not Phase6's per-source CTA swizzle.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import threading

import torch
import torch.distributed as dist

try:
    from .codec import BF16_BYTES, QUANT_BYTES, TILE, decode_tile, encode_tile, hadamard
except ImportError:
    from codec import BF16_BYTES, QUANT_BYTES, TILE, decode_tile, encode_tile, hadamard


@dataclass(frozen=True)
class Chunk:
    index: int
    row: int
    column: int
    rows: int
    columns: int
    local_tiles: tuple


def chunk_layout(m, n, world, chunk_tiles):
    """Map common chunk IDs to rectangular complete physical tile regions."""
    if world not in (2, 4, 8) or m <= 0 or m % (world * TILE) or n <= 0 or n % TILE:
        raise ValueError('Require TP2/4/8, M divisible by TP*128, and N divisible by 128')
    if type(chunk_tiles) is not int or chunk_tiles not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError('chunk_tiles must be a supported power of two from 1 through 64')
    tile_rows, tile_columns = m // (world * TILE), n // TILE
    total = tile_rows * tile_columns
    if total % chunk_tiles:
        raise ValueError('Chunks must divide the physical destination tile count')
    if chunk_tiles <= tile_columns:
        if tile_columns % chunk_tiles:
            raise ValueError('A partial-row chunk must divide the tile columns')
        row_tiles, column_tiles = 1, chunk_tiles
    elif chunk_tiles == total:
        row_tiles, column_tiles = tile_rows, tile_columns
    else:
        raise ValueError('Multi-row chunks are supported only for a complete destination')
    chunks = []
    for row in range(0, tile_rows, row_tiles):
        for col in range(0, tile_columns, column_tiles):
            physical = tuple(r * tile_columns + c for r in range(row, row + row_tiles)
                             for c in range(col, col + column_tiles))
            chunks.append(Chunk(len(chunks), row * TILE, col * TILE,
                                row_tiles * TILE, column_tiles * TILE, physical))
    return tuple(chunks)


def validate_plan(plan):
    """Strict pure-CPU protocol check before any distributed operation."""
    shape = plan['shape']
    m, n, k, world = (shape[key] for key in ('M', 'N', 'K_local', 'world_size'))
    if any(type(value) is not int for value in (m, n, k, world)) or k <= 0 or k % 32:
        raise ValueError('Shape dimensions must be integers and K_local divisible by 32')
    if shape.get('tile_m', TILE) != TILE or shape.get('tile_n', TILE) != TILE:
        raise ValueError('The initial runtime supports physical 128x128 tiles only')
    chunks = chunk_layout(m, n, world, plan['chunk_tiles'])
    order = plan['chunk_order']
    if len(order) != world or any(sorted(row) != list(range(len(chunks))) or
                                  any(type(value) is not int for value in row) for row in order):
        raise ValueError('Each destination must have a complete common-source chunk permutation')
    tiles = (m // TILE) * (n // TILE)
    mask = plan['quantize_mask']
    if len(mask) != world or any(len(row) != tiles for row in mask):
        raise ValueError('Mask must be [source, physical_tile]')
    per_destination = tiles // world
    for src, row in enumerate(mask):
        if any(type(value) not in (bool, int) or value not in (0, 1) for value in row):
            raise ValueError('Mask must contain binary values')
        if any(row[src * per_destination:(src + 1) * per_destination]):
            raise ValueError('Local contributions must remain BF16')
    return chunks


def packet_bytes(chunk, source, destination, mask, per_destination):
    if source == destination:
        return 0
    return sum(QUANT_BYTES if mask[source][destination * per_destination + tile] else BF16_BYTES
               for tile in chunk.local_tiles)


def stage_splits(plan, rank, stage):
    chunks = validate_plan(plan)
    shape = plan['shape']
    world = shape['world_size']
    count = shape['M'] // (world * TILE) * (shape['N'] // TILE)
    if not 0 <= rank < world or not 0 <= stage < len(chunks):
        raise ValueError('Rank or stage outside the plan')
    order, mask = plan['chunk_order'], plan['quantize_mask']
    outgoing = [packet_bytes(chunks[order[dst][stage]], rank, dst, mask, count)
                for dst in range(world)]
    incoming = [packet_bytes(chunks[order[rank][stage]], src, rank, mask, count)
                for src in range(world)]
    return incoming, outgoing


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ChunkedGemmRS:
    """Experimental BF16 forward with per-chunk quantization and fixed ring sum.

    Work.wait() is called inside the consumer CUDA stream. For ProcessGroupNCCL
    this creates a current-stream wait for NCCL completion; it does not claim
    synchronous CPU completion. No CUDA graph capture or CPU fallback exists.
    """
    owns_output = True
    placement = 'phase7_tensor_chunked'
    gpu_validated = False

    def __init__(self, group, plan, mode='barrier', profile=False):
        if mode not in ('barrier', 'pipeline'):
            raise ValueError('mode must be barrier or pipeline')
        self.plan = json.loads(json.dumps(plan))
        self.chunks = validate_plan(self.plan)
        shape = self.plan['shape']
        self.m, self.n, self.k, self.world = [shape[key] for key in ('M', 'N', 'K_local', 'world_size')]
        if not dist.is_initialized() or str(dist.get_backend(group)).lower() != 'nccl':
            raise ValueError('ChunkedGemmRS requires an initialized NCCL process group')
        self.group, self.rank = group, dist.get_rank(group)
        if dist.get_world_size(group) != self.world:
            raise ValueError('Plan TP degree does not match the process group')
        if not torch.cuda.is_available():
            raise ValueError('ChunkedGemmRS requires CUDA')
        self.device = torch.cuda.current_device()
        self.rows = self.m // self.world
        self.tiles_per_destination = (self.rows // TILE) * (self.n // TILE)
        # Live routing is immutable, even if the caller later mutates its plan
        # dictionary. Packet lengths and collective matching cannot drift.
        self._order = tuple(tuple(row) for row in self.plan['chunk_order'])
        self._mask = tuple(tuple(row) for row in self.plan['quantize_mask'])
        self.plan_sha256 = _sha(self.plan)
        self.selection_sha256 = _sha(self.mask)
        # Validate complete schedules, modes, masks and shapes before launching
        # variable-split collectives. This is setup, never a timed-path exchange.
        identity = _sha(dict(plan=self.plan, mode=mode))
        identities = [None] * self.world
        dist.all_gather_object(identities, identity, group=group)
        if len(set(identities)) != 1:
            raise ValueError('All ranks must use the same complete plan and consumer mode')
        self._mode, self.profile = mode, profile
        self.producer = torch.cuda.Stream(device=self.device)
        self.launcher = torch.cuda.Stream(device=self.device)
        self.consumer = torch.cuda.Stream(device=self.device)
        self._lock = threading.Lock()
        self._epoch = 0
        self.last_diagnostics = {}
        self._splits = [stage_splits(self.plan, self.rank, stage) for stage in range(len(self.chunks))]
        self.wire_bytes_per_rank = sum(sum(outgoing) for _, outgoing in self._splits)
        self.bf16_wire_bytes_per_rank = (self.world - 1) * self.rows * self.n * 2

    @property
    def mode(self):
        return self._mode

    @property
    def order(self):
        return self._order

    @property
    def mask(self):
        return self._mask

    def _scope(self, name):
        return torch.profiler.record_function('phase7.' + name) if self.profile else nullcontext()

    def _tiles(self, piece, chunk):
        for index, physical in enumerate(chunk.local_tiles):
            row, col = divmod(index, chunk.columns // TILE)
            yield physical, piece[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE]

    def _pack(self, piece, chunk, destination):
        return torch.cat([encode_tile(tile, bool(self.mask[self.rank][
            destination * self.tiles_per_destination + physical]))
            for physical, tile in self._tiles(piece, chunk)])

    def _unpack(self, packet, chunk, source):
        result = torch.empty((chunk.rows, chunk.columns), device=packet.device, dtype=torch.bfloat16)
        offset = 0
        for physical, tile in self._tiles(result, chunk):
            selected = bool(self.mask[source][self.rank * self.tiles_per_destination + physical])
            length = QUANT_BYTES if selected else BF16_BYTES
            tile.copy_(decode_tile(packet[offset:offset + length], selected))
            offset += length
        return result

    def _consume(self, pending, output):
        with torch.cuda.stream(self.consumer):
            # This wait establishes payload AND scale readiness on this stream.
            pending['work'].wait()
            chunk, received, local = pending['chunk'], pending['received'], pending['local']
            for tensor in (pending['send'], received, local):
                tensor.record_stream(self.consumer)
            with self._scope('consume.stage_' + str(pending['stage'])):
                values, offset = {}, 0
                for src, size in enumerate(pending['incoming']):
                    if src != self.rank:
                        values[src] = self._unpack(received[offset:offset + size], chunk, src)
                    offset += size
                result = torch.zeros_like(local)
                for step in range(1, self.world + 1):
                    src = (self.rank + step) % self.world
                    result = (result + (local if src == self.rank else values[src])).to(torch.bfloat16)
                output[chunk.row:chunk.row + chunk.rows,
                       chunk.column:chunk.column + chunk.columns].copy_(result)

    def forward(self, input, weight, reduce_scatter_option=None):
        if reduce_scatter_option is not None:
            raise ValueError('The standalone Phase7 runtime does not accept Flux options')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Phase7 prototype does not support CUDA graph capture')
        if torch.cuda.current_device() != self.device:
            raise ValueError('Use the instance on its construction device')
        for value, expected in ((input, (self.m, self.k)), (weight, (self.n, self.k))):
            if (not value.is_cuda or value.device.index != self.device or value.dtype != torch.bfloat16 or
                    tuple(value.shape) != expected or not value.is_contiguous()):
                raise ValueError(f'Expected contiguous CUDA BF16 tensor with shape {expected}')
        if torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad):
            raise ValueError('Use the explicit Phase7 autograd adapter or torch.no_grad()')
        with self._lock:
            return self._forward(input, weight)

    def _forward(self, input, weight):
        caller = torch.cuda.current_stream()
        ready = torch.cuda.Event()
        ready.record(caller)
        self.producer.wait_event(ready)
        input.record_stream(self.producer)
        weight.record_stream(self.producer)
        output = torch.empty((self.rows, self.n), device=input.device, dtype=torch.bfloat16)
        output.record_stream(self.consumer)
        pending_all = []
        for stage in range(len(self.chunks)):
            packets = {}
            with torch.cuda.stream(self.producer), self._scope('produce.stage_' + str(stage)):
                local = None
                # Preserve the planner's local-first destination rotation.
                # This production order is independent of the ring SUM order.
                for step in range(self.world):
                    dst = (self.rank + step) % self.world
                    chunk = self.chunks[self.order[dst][stage]]
                    first = dst * self.rows + chunk.row
                    piece = input[first:first + chunk.rows].matmul(
                        weight[chunk.column:chunk.column + chunk.columns].t())
                    if dst == self.rank:
                        local = piece
                    else:
                        packets[dst] = self._pack(piece, chunk, dst)
                send = torch.cat([packets[dst] for dst in range(self.world) if dst != self.rank])
                produced = torch.cuda.Event()
                produced.record(self.producer)
            incoming, outgoing = self._splits[stage]
            with torch.cuda.stream(self.launcher), self._scope('exchange.stage_' + str(stage)):
                self.launcher.wait_event(produced)
                send.record_stream(self.launcher)
                received = torch.empty(sum(incoming), device=input.device, dtype=torch.uint8)
                work = dist.all_to_all_single(received, send, output_split_sizes=incoming,
                    input_split_sizes=outgoing, group=self.group, async_op=True)
            pending = dict(work=work, received=received, send=send, local=local,
                           chunk=self.chunks[self.order[self.rank][stage]], incoming=incoming, stage=stage)
            pending_all.append(pending)
            if self.mode == 'pipeline':
                self._consume(pending, output)
        if self.mode == 'barrier':
            with torch.cuda.stream(self.consumer):
                # Equal packet/layout/math to pipeline. Only admission changes:
                # no decoding until every stage's communication has completed.
                for pending in pending_all:
                    pending['work'].wait()
            for pending in pending_all:
                self._consume(pending, output)
        done = torch.cuda.Event()
        done.record(self.consumer)
        caller.wait_event(done)
        self._epoch += 1
        # All tensors are fresh per invocation. Every borrowed allocation has
        # record_stream protection through its last use, including NCCL reads.
        # The retained Work/payload references survive all scheduling above;
        # caller's completion dependency covers every output write.
        self.last_diagnostics = dict(invocation=self._epoch, mode=self.mode,
            chunks=len(self.chunks), world_size=self.world, plan_sha256=self.plan_sha256,
            selection_sha256=self.selection_sha256, wire_bytes_per_rank=self.wire_bytes_per_rank,
            lifetime='fresh_allocation_with_record_stream', backend='nccl_tensor_prototype',
            timing_measured=False, gpu_validated=False)
        return output


@torch.no_grad()
def reference_forward(input, weight, group, mask):
    """Independent full GEMM/all-gather reference, including fixed BF16 sum.

    The reference transform is repeated directly rather than reusing packet
    packing, offsets, or chunk ordering, so it catches transport/layout errors.
    GEMM tiling may itself change rounded BF16 products: compare with tolerance.
    This is a diagnostic operation and must never be included in timing.
    """
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    partial = input.matmul(weight.t()).contiguous()
    all_partials = [torch.empty_like(partial) for _ in range(world)]
    dist.all_gather(all_partials, partial, group=group)
    m, n = partial.shape
    if m % (world * TILE) or n % TILE:
        raise ValueError('Reference requires complete 128x128 tiles')
    rows, columns = m // world, n // TILE
    result = torch.zeros((rows, n), device=input.device, dtype=torch.bfloat16)
    for step in range(1, world + 1):
        src = (rank + step) % world
        contribution = all_partials[src][rank * rows:(rank + 1) * rows].clone()
        for row in range(rows // TILE):
            for col in range(columns):
                physical = (rank * rows // TILE + row) * columns + col
                if not mask[src][physical]:
                    continue
                tile = contribution[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE]
                values = tile.float()
                adaptive = (448 * (values.square().mean(-1) + 1e-6).clamp_min(1e-12).rsqrt()).clamp(1e-3, 1e3)
                transformed = hadamard(values * adaptive.unsqueeze(-1))
                quant = (transformed.abs().amax(-1).clamp_min(1e-12) / 448).clamp(1e-12, 1e6)
                encoded = (transformed / quant.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
                tile.copy_((hadamard(encoded.float() * quant.unsqueeze(-1)) / adaptive.unsqueeze(-1)).to(torch.bfloat16))
        result = (result + contribution).to(torch.bfloat16)
    return result
