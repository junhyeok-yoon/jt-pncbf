"""v2.7.1 Stage-1c Phase B — k=5 empty-branch fallback on DI + unicycle SOTA (eval-only, full pool n2000)."""
import csv, json, time
from pathlib import Path
import numpy as np, torch
from src.eval.build_pools import load_pool
from src.eval.evaluate import evaluate
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint
REPO=Path("/home/junhyeok/MIT/jt-pncbf"); SP=Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
SYS={
 "DI_v2.3.0_s42":  (REPO/"data/secured_data/v2.3.0/seed42/checkpoints/best.pt", REPO/"data/secured_data/pools/eval_full_di_n2000_seed23456.pkl"),
 "UNI_v2.2.2_28k": (REPO/"data/previous_runs/v2.2.2__20260619-083424__seed42/checkpoints/best.pt", REPO/"data/secured_data/pools/eval_full_unicycle_n2000_seed23456.pkl"),
}
def boot(v,seed=20260718,n=10000):
    v=np.asarray(v); rng=np.random.default_rng(seed); b=np.array([v[rng.integers(0,len(v),len(v))].mean() for _ in range(n)])
    return float(v.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def run(ckpt,pool,mode,k):
    fw,cfg,ck=load_framework_from_checkpoint(ckpt,config_overrides={"filter":{"empty_fallback":{"mode":mode,"k":k}}})
    t0=time.time(); res=evaluate(fw,pool,cfg,mode="final",step=int(ck.get("step",0)),ckpt_name="best.pt",include_lqr_baseline=False)
    return [r for r in res.episode_rows if r.get("mode")=="final"], time.time()-t0
def summ(rows):
    cps=[float(r["cps_episode"]) for r in rows]; coll=[float(r["collision"]) for r in rows]
    cm,clo,chi=boot(cps); om,olo,ohi=boot(coll)
    return dict(n=len(rows),cps=cm,cps_ci=[clo,chi],collision=om,collision_ci=[olo,ohi],
        reach=float(np.mean([r["reach"] for r in rows])),timeout=float(np.mean([r["timeout"] for r in rows])),
        oob=float(np.mean([r["oob"] for r in rows])),stuck=float(np.mean([r["stuck"] for r in rows])),
        empty=float(np.mean([float(r.get("empty_step_frac",0)) for r in rows])),
        fire=int(sum(1 for r in rows if float(r.get("empty_step_frac",0))>0)))
def flips(b,a):
    bd={int(r["episode_idx"]):r for r in b}; ad={int(r["episode_idx"]):r for r in a}; o=[]
    for i in sorted(bd):
        if bd[i]["outcome"]!=ad[i]["outcome"]: o.append((i,bd[i]["outcome"],ad[i]["outcome"]))
    return o
def chatter(ckpt,pool,coll_ids):
    if not coll_ids: return {}
    scenes=[load_pool(pool).scenes[i] for i in coll_ids]; dev=torch.device("cpu")
    def roll(mode,k):
        fw,cfg,ck=load_framework_from_checkpoint(ckpt,config_overrides={"filter":{"empty_fallback":{"mode":mode,"k":k}}})
        bs=batch_scenes(scenes,device=dev,dtype=torch.float32); x=fw.system.wrap_state(initial_states_from_batch(bs).float())
        ms=int(cfg["eval"]["max_steps"]); dt=float(cfg["env"]["dt"]); us=[]; em=[]
        with torch.no_grad():
            for _ in range(ms):
                un=fw.policy(x,bs); u,_=fw.filter(x,un,bs); le=getattr(fw._filter,"last_empty",None)
                us.append(u.clone()); em.append(le.clone() if le is not None else torch.zeros(x.shape[0],dtype=torch.bool)); x=rk4_step(fw.system,x,u,dt)
        return torch.stack(us,0).numpy(), torch.stack(em,0).numpy()
    r={}
    for mode,k in [("none",5),("kstep",5)]:
        U,E=roll(mode,k); du=np.linalg.norm(U[1:]-U[:-1],axis=2); e=E[1:].astype(bool); de=du[e]
        r[mode]=dict(switch_rate=round(float(((du>1e-3)&e).sum()/max(e.sum(),1)),3),mean_du=round(float(de.mean()) if de.size else 0,4))
    return r
out={}
for tag,(ckpt,pool) in SYS.items():
    nr,wn=run(ckpt,pool,"none",5); kr,wk=run(ckpt,pool,"kstep",5)
    for m,rows in [("none",nr),("k5",kr)]:
        with open(SP/f"stage1c_{tag}_{m}_episodes.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    sn=summ(nr); sn["wall_s"]=round(wn,1); sk=summ(kr); sk["wall_s"]=round(wk,1); fl=flips(nr,kr)
    fixed=sum(1 for i,a,b in fl if a=="collision" and b=="goal"); newc=sum(1 for i,a,b in fl if b=="collision")
    coll_ids=[int(r["episode_idx"]) for r in nr if r["outcome"]=="collision"]
    ch=chatter(ckpt,pool,coll_ids)
    out[tag]=dict(none=sn,k5=sk,n_flips=len(fl),fixed=fixed,new_coll=newc,net=fixed-newc,
                  flip_ids=[(i,a,b) for i,a,b in fl][:40],chatter=ch)
    print(f"{tag}: none cps={sn['cps']:.4f}{[round(x,4) for x in sn['cps_ci']]} coll={sn['collision']:.4f}{[round(x,4) for x in sn['collision_ci']]} | "
          f"k5 cps={sk['cps']:.4f}{[round(x,4) for x in sk['cps_ci']]} coll={sk['collision']:.4f}{[round(x,4) for x in sk['collision_ci']]} | "
          f"flips={len(fl)} fixed={fixed} new={newc} | chatter={ch} | wall none/k5={sn['wall_s']}/{sk['wall_s']}",flush=True)
json.dump(out,open(SP/"stage1c_phaseB.json","w"),indent=2,default=str); print("WROTE",SP/"stage1c_phaseB.json",flush=True)
