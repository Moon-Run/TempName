import json,sys,collections,statistics,hashlib,csv,math
from pathlib import Path
sys.path.insert(0,str(Path('tools/h100/phase6').resolve()))
import types
aa=types.ModuleType('arrival_assignment_snapshot')
exec(compile(Path('tools/h100/phase6/arrival_assignment.py').read_text(), 'arrival_assignment_snapshot', 'exec'), aa.__dict__)
p=Path('logs/h100/phase6/ablation7-new-4x-50steps-20261019-235424-435064/compare1/scripts/selection-plan.json')
x=json.loads(p.read_text());out={'input':str(p),'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'threshold_sweep':{}}
for mul in (6,4,3,2,1,0.5,0.25,0):
 aa.UNCERTAINTY_MULTIPLIER=mul;roles={}
 for role,data in x['orders'].items():
  rr=[]
  for g in data['entry']['arrival_order']['groups']:
   samples=[dict(arrivals=t['arrivals_us'],baseline=t['baseline_us'],residual=t['residual_us'],uncertainty=t['max_uncertainty_us']) for t in g['training']]
   order,result=aa._fit_window(samples,8)
   rr.append(dict(source=g['source'],destination=g['destination'],status=result['status'],candidate=result['candidate'],order=order,uncertainty_us=[t['uncertainty'] for t in samples]))
  candidates=[r for r in rr if r['status']=='candidate'];roles[role]=dict(status_counts=dict(collections.Counter(r['status'] for r in rr)),accepted_windows=len(candidates),changed_slots=sum(r['candidate']['changed_slots'] for r in candidates),gain_us=[r['candidate']['gain_us'] for r in candidates],min_two_pass_gain_us=[min(r['candidate']['gain_us']) for r in candidates],min_two_pass_gain_over_epsilon=[min(v/e for v,e in zip(r['candidate']['gain_us'],r['uncertainty_us'])) for r in candidates],groups=rr)
 out['threshold_sweep'][str(mul)]=roles
 for r,d in roles.items():
  for field in ('min_two_pass_gain_us','min_two_pass_gain_over_epsilon'):
   values=d[field];d[field+'_range']=[min(values),max(values)] if values else None
 print(json.dumps({'multiplier':mul,'roles':{r:{k:v for k,v in d.items() if k in ('status_counts','accepted_windows','changed_slots') or k.endswith('_range')} for r,d in roles.items()}}),flush=True)
Path('/tmp/op1-threshold-sweep.json').write_text(json.dumps(out,indent=2)+'\n')
