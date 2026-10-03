"""Explicit nine-way MLP-only phase5 routes."""
POLICIES = ['native', 'original', 'taco_separate', 'remote_bf16', 'interleaved_bf16',
            'remote_all', 'interleaved_all', 'remote_selective', 'interleaved_selective']
ARRIVAL_POLICIES = ['remote_arrival_selective','interleaved_arrival_selective']
NATIVE_QUANT_POLICY = 'native_taco'


def codec_placement(policy):
    assert policy in POLICIES + ARRIVAL_POLICIES + [NATIVE_QUANT_POLICY]
    if policy == NATIVE_QUANT_POLICY: return 'tensor'
    if policy == 'taco_separate': return 'separate'
    return 'fused' if policy.endswith(('_all', '_selective')) else 'bf16'


def base_policy(policy):
    return 'remote_first' if policy.startswith('remote_') else 'interleaved' if policy.startswith('interleaved_') else 'original'


def mapping_policy(policy):
    if policy in ARRIVAL_POLICIES:
        return policy.removesuffix('_selective')
    return base_policy(policy)


def module_routes(observed, policy, layers=12):
    assert policy in POLICIES + ARRIVAL_POLICIES + [NATIVE_QUANT_POLICY]
    counts = {'attention': 0, 'mlp': 0}
    routes = {}
    for name, shape in observed.items():
        if name.endswith('.self_attention.linear_proj'):
            kind, k = 'attention', 2048
        elif name.endswith('.mlp.linear_fc2'):
            kind, k = 'mlp', 8192
        else:
            raise AssertionError(name)
        assert (shape['M'],shape['N'],shape['K_global'],shape['K_local']) == (2048,2048,k,k//4)
        counts[kind] += 1
        routes[name] = 'native' if policy=='native' or policy==NATIVE_QUANT_POLICY and kind=='attention' else policy if kind=='mlp' else 'original'
    assert counts == {'attention':layers, 'mlp':layers}
    return routes
