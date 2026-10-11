import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=120))
world = dist.get_world_size()
assert world == 8
device = torch.device('cuda', rank)

x = torch.full((1024,), rank + 1, device=device, dtype=torch.bfloat16)
dist.all_reduce(x)
torch.testing.assert_close(x, torch.full_like(x, 36), rtol=0, atol=0)

x = torch.tensor([rank], device=device, dtype=torch.int64)
gathered = [torch.empty_like(x) for _ in range(world)]
dist.all_gather(gathered, x)
assert [v.item() for v in gathered] == list(range(world))

x = torch.full((world * 1024,), rank + 1, device=device, dtype=torch.bfloat16)
y = torch.empty((1024,), device=device, dtype=torch.bfloat16)
dist.reduce_scatter_tensor(y, x)
torch.testing.assert_close(y, torch.full_like(y, 36), rtol=0, atol=0)

x = torch.arange(world, device=device, dtype=torch.int64) + rank * world
y = torch.empty_like(x)
dist.all_to_all_single(y, x)
expected = torch.arange(world, device=device, dtype=torch.int64) * world + rank
torch.testing.assert_close(y, expected, rtol=0, atol=0)
torch.cuda.synchronize()
dist.barrier()

props = torch.cuda.get_device_properties(rank)
result = dict(rank=rank, world=world, gpu_name=props.name, gpu_uuid=str(props.uuid),
              sm_count=props.multi_processor_count, torch=torch.__version__,
              cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version()),
              checks=['all_reduce_bf16', 'all_gather', 'reduce_scatter_bf16', 'all_to_all'],
              passed=True)
(Path(__file__).parent / f'nccl-rank{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
print(f'Rank {rank}: NCCL four collective checks PASS', flush=True)
dist.destroy_process_group()
