"""CR3/F4 control plus a candidate-only DDP gradient accumulation helper.

All existing projection code remains byte identical. The candidate adds one
include to its runtime and one header; it does not alter the decoder or GEMM.
"""
from pathlib import Path
from common import HERE, require
import ma4_f_sources as f4

RUNTIME, HEADER, WRAPPER, DECODER = f4.RUNTIME, f4.HEADER, f4.WRAPPER, f4.DECODER
VARIANTS = ('F4', 'F4-gacc')
BASELINE_TUNING = dict(decoder_threads=128, decoder_blocks=2048, decoder_rows=1)
ROLES = ('attn', 'mlp')
ROLE_K = dict(attn=512, mlp=2048)
DEFAULT_TUNING = {role: dict(BASELINE_TUNING) for role in ROLES}
GRADIENT_BATCHING_HEADER = 'src/gemm_rs/ma10_grad_batch.cuh'
GRADIENT_BATCHING_INCLUDE = '\n#include "gemm_rs/ma10_grad_batch.cuh"\n'


def sources(reference):
    source = (Path(reference) / RUNTIME).read_text()
    require('ma10_grad_batch' not in source, 'MA10 helper is already installed')
    return {RUNTIME: source + GRADIENT_BATCHING_INCLUDE,
            GRADIENT_BATCHING_HEADER: (HERE / 'ma10_grad_batch.cuh').read_text()}


def tuning_config(tuning=None):
    tuning = DEFAULT_TUNING if tuning is None else tuning
    require(isinstance(tuning, dict) and set(tuning) == set(ROLES),
            'MA10 tuning requires attn and mlp')
    for role in ROLES:
        item = tuning[role]
        require(isinstance(item, dict) and set(item) == set(BASELINE_TUNING)
                and all(type(value) is int for value in item.values()) and item == BASELINE_TUNING,
                'MA10 fixes the original F4 decoder geometry: ' + role)
    return {role: dict(BASELINE_TUNING) for role in ROLES}


def kernel_name(role, tuning=None, arm='F4'):
    tuning_config(tuning)
    require(role in ROLES and arm in VARIANTS, 'Unknown MA10 role or arm')
    return 'taco_ma4f_decode_consumer_kernel'
