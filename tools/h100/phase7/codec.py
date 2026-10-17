"""Tensor H128 / E4M3 wire codec for the Phase7 experimental runtime.

This uses the Phase4 reference math, not the fused Flux CUDA epilogue.
One physical 128x128 tile is either 32768 BF16 bytes or 17408 encoded
bytes: all FP8 coefficients, followed by FP32 quant and adaptive scales.
"""
import math

import torch


TILE = 128
BF16_BYTES = TILE * TILE * 2
QUANT_BYTES = TILE * (TILE + 4 + 4)


def hadamard(x):
    if x.shape[-1] != TILE:
        raise ValueError('The wire transform requires complete H128 groups')
    value = x.float().clone()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        paired = value.reshape(*value.shape[:-1], -1, 2, stride)
        left, right = paired[..., 0, :].clone(), paired[..., 1, :].clone()
        value = torch.stack((left + right, left - right), dim=-2).reshape_as(value)
    return value / math.sqrt(TILE)


def encode_tile(tile, quantized):
    """Return contiguous uint8 packet; no device-to-host diagnostic sync."""
    if tuple(tile.shape) != (TILE, TILE) or tile.dtype != torch.bfloat16:
        raise ValueError('Expected one BF16 128x128 physical tile')
    if not quantized:
        return tile.contiguous().view(torch.uint8).reshape(-1)
    values = tile.float()
    variance = values.square().sum(-1) / TILE + 1e-6
    adaptive = (448 * variance.clamp_min(1e-12).rsqrt()).clamp(1e-3, 1e3)
    transformed = hadamard(values * adaptive.unsqueeze(-1))
    quant = (transformed.abs().amax(-1).clamp_min(1e-12) / 448).clamp(1e-12, 1e6)
    payload = (transformed / quant.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return torch.cat([value.contiguous().view(torch.uint8).reshape(-1)
                      for value in (payload, quant, adaptive)])


def decode_tile(packet, quantized):
    expected = QUANT_BYTES if quantized else BF16_BYTES
    if packet.dtype != torch.uint8 or packet.ndim != 1 or packet.numel() != expected:
        raise ValueError('Invalid tile packet dtype, shape, or length')
    if not quantized:
        return packet.contiguous().view(torch.bfloat16).reshape(TILE, TILE)
    coefficients = TILE * TILE
    payload = packet[:coefficients].contiguous().view(torch.float8_e4m3fn).reshape(TILE, TILE)
    quant = packet[coefficients:coefficients + TILE * 4].contiguous().view(torch.float32)
    adaptive = packet[coefficients + TILE * 4:].contiguous().view(torch.float32)
    return (hadamard(payload.float() * quant.unsqueeze(-1)) /
            adaptive.unsqueeze(-1)).to(torch.bfloat16)


def roundtrip_tile(tile, quantized):
    return decode_tile(encode_tile(tile, quantized), quantized)
