"""GPU preflight for mixed BF16/FP8 source slots, padding, ordering and lifetimes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import traceback
import torch
import torch.distributed as dist
from flux.gemm_rs_taco import GemmRSTaco
from taco_support import check_taco_output


def main(out, policy, plan_path):
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world == 4 and os.environ.get('SLURM_JOB_ID')
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=5))
    pg=dist.new_group(list(range(world)))
    report=dict(rank=rank,world_size=world,policy=policy,checks=[],completed=False)
    out.mkdir(parents=True,exist_ok=True)
    def save(): (out/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    plan=json.loads(plan_path.read_text())
    try:
        for m,n,k in [(512,128,256),(512,136,256),(512,256,256),(2048,2048,512),(2048,2048,2048)]:
            tiles=(m//128)*((n+127)//128)
            modes=['all','none','checkerboard']+(['calibrated'] if (m,n,k)==(2048,2048,2048) else [])
            for mode in modes:
                mask=None
                if mode!='all':
                    mask=torch.zeros((world,tiles),dtype=torch.uint8)
                    if mode=='calibrated': mask=torch.tensor(plan['policies'][policy]['mask'],dtype=torch.uint8)
                    if mode=='checkerboard':
                        for src in range(world):
                            for tile in range(tiles):
                                mask[src,tile]=int(tile//(tiles//world)!=src and (tile+src)%2==0)
                op=GemmRSTaco(pg,m,n,k,placement='fused',**({'selected':mask} if mask is not None else {}))
                saved=[]
                for case in ('zero','random','spiky','repeat'):
                    torch.manual_seed(73+rank+(case=='repeat'))
                    a=(torch.randn(m,k,device='cuda')*k**-.25).to(torch.bfloat16)
                    b=(torch.randn(n,k,device='cuda')*k**-.25).to(torch.bfloat16)
                    if case=='zero':a.zero_()
                    if case=='spiky':a[:,0]*=10
                    partial=a@b.t()
                    y=op.forward(a,b)
                    torch.cuda.synchronize()
                    error=check_taco_output(partial,y,pg,mask)
                    for previous,copy in saved:assert torch.equal(previous,copy)
                    saved.append((y,y.clone()))
                    report['checks'].append(dict(M=m,N=n,K_local=k,selection=mode,case=case,
                        wire_bytes=op.wire_bytes_per_rank,bf16_wire_bytes=op.bf16_wire_bytes_per_rank,**error))
                    if mode=='none':assert op.wire_bytes_per_rank==op.bf16_wire_bytes_per_rank
                    save()
                dist.barrier(pg);torch.cuda.synchronize()
                del op
        report.update(completed=True,passed=all(c['passed'] for c in report['checks']))
        save()
    except BaseException:
        report.update(passed=False,error=traceback.format_exc())
        save()
        raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('--policy',choices=['remote_first','interleaved'],required=True)
    p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();main(a.out,a.policy,a.plan)
