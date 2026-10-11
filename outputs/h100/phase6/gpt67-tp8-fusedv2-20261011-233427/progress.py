"""Read closed rank files without changing the live campaign."""
import json
import os
from pathlib import Path

root = Path(os.environ['H100_LOG'])
state = json.loads((root / 'execution.json').read_text())
dirs = sorted((root / 'compare/results').glob('round*/window-*'))
complete = [d for d in dirs if all((d / f'rank{r}.json').is_file() for r in range(8))]
print('status:', state['status'], 'stage:', state['stage'],
      'completed:', len(complete), '/72', 'started:', len(dirs))
if dirs:
    print('latest:', str(dirs[-1].relative_to(root / 'compare/results')))
if len(complete) >= 2:
    ends = [(d / 'rank7.json').stat().st_mtime for d in complete[-13:]]
    mean = (ends[-1] - ends[0]) / (len(ends) - 1)
    print('recent window wall seconds:', round(mean, 1),
          'remaining minutes:', round(mean * (72 - len(complete)) / 60, 1))
if state['status'] == 'failed':
    print(state['detail'])
