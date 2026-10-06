"""Check GPU coordinates, destination/window preservation and physical masks on TP8."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
info = json.loads((root/'submission.json').read_text())
plan = json.loads((root/'scripts/selection-plan.json').read_text())
assert plan['world'] == 8
build = Path(info['artifact_root'])/'build'
out = root/'results/mapping'
checks = []
for policy in info['mappings']:
    base = 'remote_first' if policy == 'remote_arrival' else 'interleaved'
    entry = plan['policies'][base][0]
    assert entry['shape'] == [2048, 2048, 8192]
    for k, mlp in [(2048, False), (8192, True)]:
        run = subprocess.run([str(build/'mapping'/policy), '2048', '2048', str(k), '8', str(int(mlp))], capture_output=True, text=True, check=True)
        (out/f'{policy}-k{k}.csv').write_text(run.stdout)
        (out/f'{policy}-k{k}.log').write_text(run.stderr)
        rows = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(io.StringIO(run.stdout))]
        assert len(rows) == 8*256
        for rank in range(8):
            rr = [r for r in rows if r['rank'] == rank]
            assert [r['index'] for r in rr] == list(range(256))
            coords = [r['m']*16+r['n'] for r in rr]
            assert sorted(coords) == list(range(256))
            if mlp:
                assert coords == entry['coords'][rank]
                previous = entry['base_coords'][rank]
                mapped = entry['maps'][rank]
                assert coords == [previous[j] for j in mapped]
                slots = {dst: [j for j, t in enumerate(previous) if t//32 == dst] for dst in range(8)}
                positions = {slot: (dst, j//plan['window']) for dst, indices in slots.items() for j, slot in enumerate(indices)}
                assert all(positions[i] == positions[j] for i, j in enumerate(mapped))
                mask = entry['mask'][rank]
                assert len(mask) == 256 and not any(mask[rank*32:(rank+1)*32])
                assert sum(mask) <= int(224*plan['budget_fraction'])
                assert {t for t in previous if mask[t]} == {t for t in coords if mask[t]}
        checks.append(dict(policy=policy, shape=[2048, 2048, k], mlp=mlp, passed=True))
(out/'verification.json').write_text(json.dumps(dict(completed=True, world=8, checks=checks), indent=2)+'\n')
