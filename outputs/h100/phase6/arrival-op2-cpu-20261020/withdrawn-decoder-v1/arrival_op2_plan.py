"""Preserve A's physical plan; fit only the independent v3/1ε arrival diagnostic."""
import copy

import ablation3_plan
import ablation7_new_plan as assignment_plan
import arrival_assignment
import scope_plan
from arrival_op2_protocol import (PROTOCOL, SHAPES, POLICIES, A, D, R, ARRIVAL_OPTIONS,
                                  definition, digest, execution_timing)
from common import require, validate_entry


def frozen_options(record):
    options = arrival_assignment.normalize_options(record.get('arrival_options'))
    require(options == ARRIVAL_OPTIONS, 'Arrival-op2 fixes explicit v3/1-epsilon options')
    return options


def measured_orders(parent, arrival_options=None):
    options = dict(ARRIVAL_OPTIONS) if arrival_options is None else frozen_options(dict(arrival_options=arrival_options))
    return assignment_plan.measured_orders(parent, options)


def fixture_orders(parent, arrival_options=None):
    options = dict(ARRIVAL_OPTIONS) if arrival_options is None else frozen_options(dict(arrival_options=arrival_options))
    return assignment_plan.fixture_orders(parent, options)


def derive(parent, orders, policy):
    roles = copy.deepcopy(ablation3_plan.derive(parent, A)['roles'])
    if policy == R:
        for role, record in roles.items():
            entry = record['entry']
            for field in ('maps', 'coords'):
                entry[field] = copy.deepcopy(orders[role]['entry'][field])
            record['entry_sha256'] = digest(entry)
    for role, record in roles.items():
        validate_entry(record['entry'], dict(parent['roles'][role]['config'], fraction=1), SHAPES[role])
    return dict(definition=definition(policy), roles=roles)


def bundle(parent, source_only=False, verify_evidence=True, arrival_options=None):
    options = dict(ARRIVAL_OPTIONS) if arrival_options is None else frozen_options(dict(arrival_options=arrival_options))
    require(source_only == parent['source_only'], 'Cannot promote a source-only parent')
    for role, record in parent['roles'].items():
        scope_plan.validate(record, role, allow_test=source_only, verify_evidence=verify_evidence)
    orders = fixture_orders(parent, options) if source_only else measured_orders(parent, options)
    result = dict(protocol=PROTOCOL, source_only=source_only, gpu_calibrated=not source_only,
                  execution_timing=execution_timing(), parent=copy.deepcopy(parent),
                  arrival_options=options, orders=orders,
                  variants={p: derive(parent, orders, p) for p in POLICIES})
    return validate(result, verify_evidence=False)


def validate(plans, verify_evidence=True):
    options = frozen_options(plans)
    parent, source_only = plans['parent'], plans['source_only']
    require(plans['protocol'] == PROTOCOL and plans['execution_timing'] == execution_timing() and
            set(plans['variants']) == set(POLICIES) and
            set(plans['orders']) == set(parent['roles']) == set(SHAPES),
            'Wrong arrival-op2 plan set/timing')
    require(type(source_only) is bool and source_only == parent['source_only'] and
            plans['gpu_calibrated'] == (not source_only) == parent['gpu_calibrated'],
            'Plan rehearsal identity changed')
    identities = []
    expected_base = [[((i % 8+r) % 8)*64+i//8 for i in range(512)] for r in range(8)]
    for role, record in parent['roles'].items():
        scope_plan.validate(record, role, allow_test=source_only, verify_evidence=verify_evidence)
        require(record['entry']['base_coords'] == expected_base, 'Parent must use interleaved base')
        if not source_only:
            require(not record['test_only'] and record['origin']['kind'] == 'scope_bf16_calibration',
                    'Both measured role calibrations required')
            identities.append(scope_plan.hardware_identity(record['hardware']))
        order = plans['orders'][role]; entry = order['entry']; diagnostics = entry['arrival_order']
        require(order['kind'] == ('cpu_fixture' if source_only else 'measured_bf16') and
                order['training_passes'] == [0, 1] and order['heldout_passes'] == [2] and
                (source_only or order['inputs']) and
                all(record['inputs'].get(k) == v for k, v in order['inputs'].items()),
                'Arrival evidence or training/heldout split changed')
        validate_entry(entry, record['config'], SHAPES[role])
        require(order['entry_sha256'] == digest(entry) and entry['base_coords'] == expected_base and
                entry['mask'] == record['entry']['mask'], 'Arrival source entry/mask changed')
        require(diagnostics['method'] == arrival_assignment.CONFIGURED_METHOD and
                frozen_options(diagnostics) == options and diagnostics['uncertainty_multiplier'] == 1.0 and
                diagnostics['training_passes'] == [0, 1] and diagnostics['heldout_passes'] == [2] and
                diagnostics['gpu_validated'] is False and diagnostics['exploratory'] is True and
                diagnostics['conditional_error_multiplier'] == 4.0 and
                diagnostics['max_displacement'] == arrival_assignment.MAX_DISPLACEMENT and
                diagnostics['changed_per_rank'] == [sum(i != j for i, j in enumerate(row)) for row in entry['maps']] and
                diagnostics['identity_fallback'] == (not any(diagnostics['changed_per_rank'])) and
                diagnostics['accepted_windows'] == sum(g['status'] == 'candidate' for g in diagnostics['groups']),
                'Arrival assignment method/diagnostics changed')
        candidates = [g['candidate'] for g in diagnostics['groups'] if g['status'] == 'candidate']
        require(diagnostics['accepted_windows_with_positive_conservative_bound'] == sum(
                all(v > arrival_assignment.STRICT_TOLERANCE_US for v in c['conservative_gain_lower_bound_us'])
                for c in candidates), 'Arrival bound diagnostics changed')
        for group in diagnostics['groups']:
            candidate = group['candidate']
            if candidate is not None:
                errors = [sample['max_uncertainty_us'] for sample in group['training']]
                require(len(errors) == len(candidate['gain_us']) == 2 and
                        candidate['conservative_gain_lower_bound_us'] ==
                        [gain-4.0*error for gain, error in zip(candidate['gain_us'], errors)],
                        'Arrival conditional error bounds changed')
    if identities:
        require(identities[0] == identities[1], 'Role hardware/order differs')
    if source_only:
        require(plans['orders'] == fixture_orders(parent), 'CPU fixture order changed')
    elif verify_evidence:
        require(plans['orders'] == measured_orders(parent), 'Arrival mapping differs from raw replay')
    for policy in POLICIES:
        require(plans['variants'][policy] == derive(parent, plans['orders'], policy),
                'Derived order/mask/role identity changed')
    require(plans['variants'][A]['roles'] == plans['variants'][D]['roles'],
            'Decoder candidate must retain the exact baseline physical plans')
    return plans
