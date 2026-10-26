"""Launch-correlated path checks for MA8 activation retention.

No profiler time is interpreted as an unprofiled speedup. The historical
column reader is explicitly separate from the marker-required MA8 gate.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def end(event):
    return event['ts'] + event.get('dur', 0)


def contains(parent, child):
    return (parent['tid'] == child['tid'] and parent['ts'] <= child['ts'] + .01
            and end(child) <= end(parent) + .01)


class TraceIndex:
    def __init__(self, events):
        self.events = events
        self.runtime = defaultdict(list)
        self.device = defaultdict(list)
        self.external = defaultdict(list)
        self.cpu = defaultdict(list)
        for event in events:
            cat, args = event.get('cat'), event.get('args', {})
            if cat in ('cuda_runtime', 'cuda_driver'):
                self.runtime[event['tid']].append(event)
            elif cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
                if args.get('correlation') is not None:
                    self.device[args['correlation']].append(event)
                if args.get('External id') is not None:
                    self.external[args['External id']].append(event)
            elif cat == 'cpu_op':
                self.cpu[event['tid']].append(event)
        for rows in self.runtime.values():
            rows.sort(key=lambda e: e['ts'])
        self.starts = {tid: [e['ts'] for e in rows] for tid, rows in self.runtime.items()}

    def work(self, call):
        rows, starts = self.runtime[call['tid']], self.starts.get(call['tid'], [])
        lo = bisect_left(starts, call['ts'] - .01)
        hi = bisect_right(starts, end(call) + .01)
        launches = {e['args']['correlation']: e for e in rows[lo:hi]
                    if contains(call, e) and e.get('args', {}).get('correlation') is not None}
        associated = {id(e): e for e in self.external.get(call.get('args', {}).get('External id'), [])}
        for correlation in launches:
            associated.update({id(e): e for e in self.device[correlation]})
        for event in associated.values():
            require(event.get('args', {}).get('correlation') in launches,
                    'Device event lacks a launch inside its owning scope')
        return list(associated.values()), launches

    def clears(self, call, work, launches, qkv_bias=False):
        """Permit only observed original library workspace initialization."""
        require(not any(e['cat'] == 'gpu_memcpy' for e in work), 'Unexpected GPU copy')
        counts = Counter()
        owners_seen = set()
        for event in work:
            if event['cat'] != 'gpu_memset':
                continue
            launch = launches[event['args']['correlation']]
            size = event['args'].get('bytes')
            if size == 4:
                names, kind, token = ('aten::mm', 'aten::matmul'), 'cublas_4byte', 'gemm'
            elif qkv_bias and size == 48:
                names, kind, token = ('aten::sum',), 'qkv_bias_48byte', 'reduce_kernel'
            else:
                raise ValueError('Unexpected GPU memset size or owner')
            owners = [e for e in self.cpu[call['tid']]
                      if e['name'] in names and contains(call, e) and contains(e, launch)]
            require(owners, 'Allowed-size memset is outside the original library operation')
            owner = min(owners, key=lambda e: e['dur'])
            require(id(owner) not in owners_seen, 'Repeated workspace clear inside one library operation')
            owners_seen.add(id(owner))
            consumers = [e for e in work if e['cat'] == 'kernel' and token in e['name'].lower()
                         and contains(owner, launches[e['args']['correlation']])]
            require(len(consumers) == 1 and
                    consumers[0]['args']['stream'] == event['args']['stream'] and
                    end(event) <= consumers[0]['ts'] + .01,
                    'Workspace clear is not followed by its original same-stream kernel')
            counts[kind] += 1
        return dict(counts)


def _arm(arm):
    require(arm in ('F4', 'F4-cache'), 'Unknown MA8 arm')
    return arm != 'F4'


def _role(module):
    if module.endswith('.self_attention.linear_qkv'):
        return 'qkv'
    if module.endswith('.mlp.linear_fc1'):
        return 'fc1'
    raise ValueError('Unexpected retained-activation module')


def _marked_column_pairs(index, arm):
    candidate = _arm(arm)
    classname = 'MA8RetainedActivationLinear' if candidate else 'MA8ReferenceLinear'
    forwards = [e for e in index.events if e.get('cat') == 'cpu_op' and e['name'] == classname]
    backwards = [e for e in index.events if e.get('cat') == 'cpu_op' and e['name'] == classname + 'Backward']
    require(len(forwards) == len(backwards) == 256, 'Missing MA8 column autograd coverage')
    expected_names = {classname, classname + 'Backward'}
    require(not any(e.get('cat') == 'cpu_op' and e['name'].startswith('MA8')
                    and e['name'] not in expected_names for e in index.events), 'Wrong column arm executed')
    scopes = [e for e in index.events if e.get('cat') == 'user_annotation'
              and e['name'].startswith('ma8.column.') and e['tid'] in index.runtime]
    mapped = {}
    used = set()
    modules = Counter()
    for phase, calls in (('forward', forwards), ('backward', backwards)):
        for call in calls:
            prefix = 'ma8.column.' + phase + '.'
            matching = [e for e in scopes if e['name'].startswith(prefix) and contains(call, e)]
            require(len(matching) == 1, 'Missing or duplicate column role marker')
            scope = matching[0]
            require(id(scope) not in used, 'Column marker was attributed twice')
            used.add(id(scope))
            module = scope['name'][len(prefix):]
            seq = call['args']['Sequence number']
            key = (seq, phase)
            require(key not in mapped, 'Duplicate column autograd sequence')
            mapped[key] = (_role(module), module, call)
            modules[(module, phase)] += 1
    require(len(used) == len(scopes) == 512, 'Unexpected column marker coverage')
    require(len(modules) == 128 and set(modules.values()) == {4}, 'Expected 64 modules and four microbatches')
    pairs = []
    for forward in forwards:
        seq = forward['args']['Sequence number']
        role, module, _ = mapped[(seq, 'forward')]
        require((seq, 'backward') in mapped, 'Missing matching column backward')
        back_role, back_module, backward = mapped[(seq, 'backward')]
        require((role, module) == (back_role, back_module), 'Forward/backward module identity differs')
        pairs.append((role, forward, backward))
    return pairs


def _column_records(index, pairs, candidate):
    records = defaultdict(list)
    used = set()
    for role, forward, backward in pairs:
        fw, fl = index.work(forward)
        bw, bl = index.work(backward)
        require(not used.intersection(map(id, fw + bw)), 'Column device work attributed twice')
        used.update(map(id, fw + bw))
        fclears = index.clears(forward, fw, fl)
        bclears = index.clears(backward, bw, bl, qkv_bias=role == 'qkv')
        fk = [e for e in fw if e['cat'] == 'kernel']
        bk = [e for e in bw if e['cat'] == 'kernel']
        fag = [e for e in fk if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        fm = [e for e in fk if 'gemm' in e['name'].lower()]
        adds = [e for e in fk if 'elementwise' in e['name'].lower() and 'add' in e['name'].lower()]
        ag = [e for e in bk if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        rs = [e for e in bk if 'nccl' in e['name'].lower() and 'ReduceScatter' in e['name']]
        mm = sorted([e for e in bk if 'gemm' in e['name'].lower()], key=lambda e: e['ts'])
        bias = [e for e in bk if 'reduce_kernel' in e['name']]
        require(len(fag) == len(fm) == 1 and len(adds) == int(role == 'qkv') and
                len(fk) == 2 + int(role == 'qkv'), 'Column forward path changed')
        require(end(fag[0]) <= fm[0]['ts'] + .01, 'Forward GEMM consumed an unfinished gather')
        require(fag[0]['args']['stream'] != fm[0]['args']['stream'], 'Forward communication stream changed')
        if adds:
            require(adds[0]['args']['stream'] == fm[0]['args']['stream'] and
                    end(fm[0]) <= adds[0]['ts'] + .01, 'Forward bias add order or stream changed')
        require(len(ag) == int(not candidate) and len(rs) == 1 and len(mm) == 2 and
                len(bias) == int(role == 'qkv') and len(bk) == 3 + int(not candidate) + int(role == 'qkv'),
                'Column backward AllGather/ReduceScatter/GEMM/bias coverage changed')
        dgrad, wgrad = mm
        reduce_scatter = rs[0]
        require(dgrad['args']['stream'] == wgrad['args']['stream'] and
                dgrad['args']['stream'] == fm[0]['args']['stream'] and
                reduce_scatter['args']['stream'] != dgrad['args']['stream'], 'Column streams changed')
        require(end(dgrad) <= wgrad['ts'] + .01 and end(dgrad) <= reduce_scatter['ts'] + .01,
                'Column GEMM order or dgrad-to-ReduceScatter dependency changed')
        require(end(bl[dgrad['args']['correlation']]) <= bl[reduce_scatter['args']['correlation']]['ts'] + .01 and
                end(bl[reduce_scatter['args']['correlation']]) <= bl[wgrad['args']['correlation']]['ts'] + .01,
                'Original dgrad, ReduceScatter, wgrad submission order changed')
        if ag:
            require(end(ag[0]) <= wgrad['ts'] + .01 and end(ag[0]) <= reduce_scatter['ts'] + .01,
                    'Baseline consumed unfinished gathered activation')
        if bias:
            require(bias[0]['args']['stream'] == wgrad['args']['stream'] and
                    end(wgrad) <= bias[0]['ts'] + .01, 'Column bias reduction moved')
        metrics = dict(forward_cpu_us=forward['dur'], backward_cpu_us=backward['dur'],
                       forward_allgather_us=fag[0]['dur'], dgrad_us=dgrad['dur'], wgrad_us=wgrad['dur'],
                       backward_allgather_us=ag[0]['dur'] if ag else 0.,
                       allgather_tail_after_dgrad_us=max(0., end(ag[0]) - end(dgrad)) if ag else 0.,
                       allgather_dgrad_overlap_us=max(0., min(end(ag[0]), end(dgrad)) - max(ag[0]['ts'], dgrad['ts'])) if ag else 0.,
                       dgrad_end_to_reduce_scatter_us=reduce_scatter['ts'] - end(dgrad),
                       allgather_end_to_reduce_scatter_us=reduce_scatter['ts'] - end(ag[0]) if ag else 0.,
                       gpu_chain_us=max(end(reduce_scatter), end(wgrad)) - min(ag[0]['ts'] if ag else dgrad['ts'], dgrad['ts']))
        records[role].append(dict(metrics=metrics, clears=dict(Counter(fclears) + Counter(bclears))))
    require(set(records) == {'qkv', 'fc1'} and all(len(v) == 128 for v in records.values()),
            'Expected 128 QKV and 128 FC1 calls per rank')
    result = {}
    for role, rows in records.items():
        clears = Counter()
        for row in rows:
            clears.update(row['clears'])
        result[role] = dict(calls=len(rows), forward_allgather_kernels=128,
                            backward_allgather_kernels=0 if candidate else 128,
                            dgrad_reduce_scatter_kernels=128, dgrad_kernels=128, wgrad_kernels=128,
                            bias_gradient_kernels=128 if role == 'qkv' else 0,
                            allowed_workspace_memsets=dict(clears),
                            median_us={key: statistics.median(r['metrics'][key] for r in rows) for key in rows[0]['metrics']},
                            sum_ms={key: sum(r['metrics'][key] for r in rows) / 1000 for key in rows[0]['metrics']})
    return dict(roles=result, column_calls=256, retained_forward_allgathers=256,
                backward_allgathers=0 if candidate else 256, retained_reduce_scatters=256,
                no_gpu_copies=True, original_gemm_order_and_streams=True)


def trace_activation_cache_records(events, arm):
    """Validate actual MA8 marked column paths; no historical fallback is allowed."""
    index = TraceIndex(events)
    return _column_records(index, _marked_column_pairs(index, arm), _arm(arm))


def historical_column_records(events):
    """Diagnostic only: frozen 32-layer, four-microbatch F4 trace without MA8 markers.

    The 65th original column call per microbatch is the vocabulary head and is
    excluded. Forward QKV bias presence independently checks alternating roles.
    """
    index = TraceIndex(events)
    name = 'LinearWithGradAccumulationAndAsyncCommunication'
    forwards = sorted([e for e in events if e.get('cat') == 'cpu_op' and e['name'] == name], key=lambda e: e['ts'])
    backwards = [e for e in events if e.get('cat') == 'cpu_op' and e['name'] == name + 'Backward']
    require(len(forwards) == len(backwards) == 260, 'Historical F4 topology changed')
    by_sequence = {e['args']['Sequence number']: e for e in backwards}
    require(len(by_sequence) == 260, 'Duplicate historical backward sequence')
    pairs = [('qkv' if i % 65 % 2 == 0 else 'fc1', e, by_sequence[e['args']['Sequence number']])
             for i, e in enumerate(forwards) if i % 65 != 64]
    return _column_records(index, pairs, False)


def trace_row_backward_records(events, arm):
    """Confirm unchanged CR3 row backward and classify its cuBLAS clears narrowly."""
    _arm(arm)
    index = TraceIndex(events)
    forwards = [e for e in events if e.get('cat') == 'cpu_op' and e['name'] == 'ConsumerGemmRSFunction']
    backwards = [e for e in events if e.get('cat') == 'cpu_op' and e['name'] == 'ConsumerGemmRSFunctionBackward']
    scopes = [e for e in events if e.get('cat') == 'user_annotation'
              and e['name'] in ('scope.attn.candidate', 'scope.mlp.candidate')]
    require(len(forwards) == len(backwards) == 256, 'Missing row projection autograd coverage')
    roles = {}
    for call in forwards:
        matching = [e for e in scopes if contains(e, call)]
        require(len(matching) == 1, 'Missing or duplicate row forward scope')
        seq = call['args']['Sequence number']
        require(seq not in roles, 'Duplicate row forward sequence')
        roles[seq] = matching[0]['name'].split('.')[1]
    result = defaultdict(list)
    used, seen = set(), set()
    for call in backwards:
        seq = call['args']['Sequence number']
        require(seq in roles and seq not in seen, 'Unmatched row backward')
        seen.add(seq)
        work, launches = index.work(call)
        require(not used.intersection(map(id, work)), 'Row GPU work attributed twice')
        used.update(map(id, work))
        clears = index.clears(call, work, launches)
        kernels = [e for e in work if e['cat'] == 'kernel']
        bias = [e for e in kernels if 'reduce_kernel' in e['name']]
        ag = [e for e in kernels if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        mm = sorted([e for e in kernels if 'gemm' in e['name'].lower()], key=lambda e: e['ts'])
        require(len(kernels) == 4 and len(bias) == len(ag) == 1 and len(mm) == 2,
                'Unchanged row backward kernel coverage differs')
        bias, ag = bias[0], ag[0]
        dgrad, wgrad = mm
        require(bias['args']['stream'] == dgrad['args']['stream'] == wgrad['args']['stream'] and
                ag['args']['stream'] != dgrad['args']['stream'], 'Row backward stream changed')
        require(end(bias) <= ag['ts'] + .01 and end(ag) <= dgrad['ts'] + .01 and
                end(dgrad) <= wgrad['ts'] + .01, 'Row bias, gather, dgrad, wgrad order changed')
        require(end(launches[bias['args']['correlation']]) <= launches[ag['args']['correlation']]['ts'] + .01,
                'Row gather moved ahead of bias')
        result[roles[seq]].append(dict(backward_cpu_us=call['dur'], bias_us=bias['dur'],
            allgather_us=ag['dur'], dgrad_us=dgrad['dur'], wgrad_us=wgrad['dur'],
            cublas_4byte_memsets=clears.get('cublas_4byte', 0)))
    require(set(result) == {'attn', 'mlp'} and all(len(rows) == 128 for rows in result.values()),
            'Expected 128 Attention and 128 MLP row backwards')
    return {role: dict(backward_calls=128, bias_kernels=128, allgather_kernels=128,
                      dgrad_kernels=128, wgrad_kernels=128, gather_schedule='baseline',
                      allowed_cublas_4byte_memsets=sum(r['cublas_4byte_memsets'] for r in rows),
                      no_gpu_copies=True, original_order_and_streams=True,
                      median_us={key: statistics.median(r[key] for r in rows)
                                 for key in rows[0] if key != 'cublas_4byte_memsets'})
            for role, rows in result.items()}
