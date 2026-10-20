"""All three policies retain the pinned fused-v2 frontend, clone and barriers."""
import ablation4_sources
from common import replace_once


def adapter_source():
    return ablation4_sources.adapter_source()


def taco_source():
    return ablation4_sources.taco_source()


def worker_source(source_only=False):
    source = ablation4_sources.worker_source(source_only)
    source = replace_once(source, "os.environ.get('H100_ABLATION4_EXECUTE')",
                          "os.environ.get('H100_ARRIVAL_OP2_EXECUTE')")
    source = replace_once(source, 'GPU worker requires ablation4 --execute',
                          'GPU worker requires arrival_op2 --execute')
    if source_only:
        source = replace_once(source, 'Source-only scope worker is permanently non-executable',
                              'Source-only arrival_op2 worker is permanently non-executable')
    anchor = "PLAN = json.loads((HERE / 'config.json').read_text())"
    source = replace_once(source, anchor, anchor+'''
from routing import PROTOCOL, execution_timing
execution = execution_timing()
if (PLAN.get('protocol') != PROTOCOL or PLAN.get('execution_timing') != execution or
        any(type(PLAN.get(key)) is not int or PLAN[key] != value for key, value in execution.items())):
    raise ValueError('Arrival-op2 requires frozen 10-warmup / 50-timed-step execution timing')
'''.rstrip())
    return replace_once(source, "report['protocol'] = PLAN['protocol']",
                         "report['execution_timing'] = execution\nreport['protocol'] = PLAN['protocol']")
