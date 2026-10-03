"""Freeze fresh BF16 arrival calibration for communication-heavy shapes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SHAPES=[[8192,512,2048],[8192,1024,4096],[8192,2048,8192],[8192,4096,2048]]
POLICIES=['remote_first','interleaved','interleaved_remote']


def main(out,sampler,group_sampler=None,large=False):
    repo=Path(__file__).resolve().parents[3]
    out,sampler=out.resolve(),sampler.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'calibration-scripts').mkdir()
    source=repo/'logs/a800/arrival/arrival-v2-tp4-20261003/results/calibration-fit/calibration-scripts/worker.py'
    assert source.is_file()
    shutil.copy2(source,out/'calibration-scripts/worker.py')
    shapes=[[65536,512,2048],[32768,1024,4096],[16384,2048,8192]] if large else SHAPES
    cfg=dict(shapes=shapes,repeats=0,warmup=100,trials=3,iters=50,sample_warmup=10)
    (out/'calibration-scripts/config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    info=dict(repo=str(repo),sampler=str(sampler),shapes=shapes,policies=POLICIES+(['interleaved_remote_group'] if group_sampler else []),
              rationale='Smaller local K increases communication per FLOP; larger M exposes more independent tiles. Last wide-output case is operator-only, not a standard 4H MLP.',
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if group_sampler:info['group_sampler']=str(group_sampler.resolve())
    (out/'plan.json').write_text(json.dumps(info,indent=2)+'\n')
    shutil.copy2(Path(__file__).with_name('calibration_driver.py'),out/'driver.py')
    (out/'manifest.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('out',type=Path);p.add_argument('--sampler',type=Path,required=True)
    p.add_argument('--group-sampler',type=Path)
    p.add_argument('--large',action='store_true')
    a=p.parse_args();main(a.out,a.sampler,a.group_sampler,a.large)
