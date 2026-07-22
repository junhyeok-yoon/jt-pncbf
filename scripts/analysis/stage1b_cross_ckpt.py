"""v2.7.1 Stage-1b — k=5 empty-branch fallback confirmation across the two secured prior checkpoints (iter-5
3b27d691, iter-1 22b902a6) + v2.7.1 k=5 chattering (S5a). Eval-only; empty_fallback set per-invocation via
config_overrides. Reuses Stage-1 machinery. No training, no config commits."""
import csv, json, time
from pathlib import Path
import numpy as np, torch
from src.eval.build_pools import load_pool
from src.eval.evaluate import evaluate
from src.envs.scene_batch import batch_scenes, initial_states_from_batch
from src.common.rk4 import rk4_step
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint

REPO=Path("/home/junhyeok/MIT/jt-pncbf")
SP=Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
POOL=REPO/"data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.pkl"
CKPTS={
  "iter5_3b27d691": REPO/"data/secured_data/v2.7.0/seed42_iter5/checkpoints/best.pt",
  "iter1_22b902a6": REPO/"data/secured_data/v2.7.0/seed42/checkpoints/best.pt",
}
STORED={
  "iter5_3b27d691": REPO/"data/secured_data/v2.7.0/seed42_iter5/eval_episodes.csv",
  "iter1_22b902a6": REPO/"data/secured_data/v2.7.0/seed42/eval_episodes.csv",
}

def boot(v, seed=20260718, n=10000):
    v=np.asarray(v); rng=np.random.default_rng(seed)
    b=np.array([v[rng.integers(0,len(v),len(v))].mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(b,2.5)), float(np.percentile(b,97.5))

def run(ckpt, mode, k):
    fw,cfg,ck=load_framework_from_checkpoint(ckpt, config_overrides={"filter":{"empty_fallback":{"mode":mode,"k":k}}})
    t0=time.time()
    res=evaluate(fw,POOL,cfg,mode="final",step=int(ck["step"]),ckpt_name="best.pt",include_lqr_baseline=False)
    wall=time.time()-t0
    rows=[r for r in res.episode_rows if r.get("mode")=="final"]
    return rows, wall

def summ(rows):
    cps=[float(r["cps_episode"]) for r in rows]; coll=[float(r["collision"]) for r in rows]
    cm,clo,chi=boot(cps); om,olo,ohi=boot(coll)
    return dict(n=len(rows), cps=cm, cps_ci=[clo,chi], collision=om, collision_ci=[olo,ohi],
        reach=float(np.mean([r["reach"] for r in rows])), timeout=float(np.mean([r["timeout"] for r in rows])),
        empty=float(np.mean([float(r.get("empty_step_frac",0)) for r in rows])),
        firing_episodes=int(sum(1 for r in rows if float(r.get("empty_step_frac",0))>0)))

def flips(b,a):
    bd={int(r["episode_idx"]):r for r in b}; ad={int(r["episode_idx"]):r for r in a}; out=[]
    for i in sorted(bd):
        if bd[i]["outcome"]!=ad[i]["outcome"]:
            out.append(dict(ep=i, frm=bd[i]["outcome"], to=ad[i]["outcome"], had_empty=bool(float(bd[i].get("empty_step_frac",0))>0)))
    return out

def regression(none_rows, stored_csv):
    st={int(r["episode_idx"]):r["outcome"] for r in csv.DictReader(open(stored_csv)) if r.get("mode")=="final"}
    nr={int(r["episode_idx"]):r["outcome"] for r in none_rows}
    common=set(st)&set(nr); dif=[i for i in common if st[i]!=nr[i]]
    return dict(n_common=len(common), n_outcome_flips=len(dif), flip_ids=sorted(dif)[:30])

def chatter(ckpt, coll_ids):
    scenes_all=load_pool(POOL).scenes; scenes=[scenes_all[i] for i in coll_ids]; dev=torch.device("cpu")
    def roll(mode,k):
        fw,cfg,ck=load_framework_from_checkpoint(ckpt, config_overrides={"filter":{"empty_fallback":{"mode":mode,"k":k}}})
        bs=batch_scenes(scenes,device=dev,dtype=torch.float32); x=fw.system.wrap_state(initial_states_from_batch(bs).float())
        ms=int(cfg["eval"]["max_steps"]); dt=float(cfg["env"]["dt"]); us=[]; em=[]
        with torch.no_grad():
            for _ in range(ms):
                un=fw.policy(x,bs); u,_=fw.filter(x,un,bs); le=getattr(fw._filter,"last_empty",None)
                us.append(u.clone()); em.append(le.clone() if le is not None else torch.zeros(x.shape[0],dtype=torch.bool))
                x=rk4_step(fw.system,x,u,dt)
        return torch.stack(us,0).numpy(), torch.stack(em,0).numpy()
    o={}
    for mode,k in [("none",5),("kstep",5)]:
        U,E=roll(mode,k); du=np.linalg.norm(U[1:]-U[:-1],axis=2); e=E[1:].astype(bool)
        de=du[e]; sw=((du>1e-3)&e)
        o[f"{mode}_k{k}"]=dict(empty_steps=int(e.sum()), switch_rate=round(float(sw.sum()/max(e.sum(),1)),3),
                               mean_du=round(float(de.mean()) if de.size else 0,4))
    return o

def main():
    R={}
    for tag,ckpt in CKPTS.items():
        print(f"=== {tag} ===",flush=True)
        none_rows,wn=run(ckpt,"none",5); k5_rows,wk=run(ckpt,"kstep",5)
        for m,rows in [("none",none_rows),("k5",k5_rows)]:
            with open(SP/f"stage1b_{tag}_{m}_episodes.csv","w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        sn=summ(none_rows); sn["wall_s"]=round(wn,1); sk=summ(k5_rows); sk["wall_s"]=round(wk,1)
        fl=flips(none_rows,k5_rows)
        reg=regression(none_rows, STORED[tag])
        coll_ids=[int(r["episode_idx"]) for r in none_rows if r["outcome"]=="collision"]
        ch=chatter(ckpt, coll_ids)
        R[tag]=dict(none=sn, k5=sk, flips=fl, n_flips=len(fl),
                    fixed=sum(1 for f in fl if f["frm"]=="collision" and f["to"]=="goal"),
                    coll_to_other=sum(1 for f in fl if f["frm"]=="collision" and f["to"]!="goal"),
                    new_coll=sum(1 for f in fl if f["to"]=="collision"),
                    regression=reg, chatter=ch)
        print(f"  none: cps={sn['cps']:.4f}{sn['cps_ci']} coll={sn['collision']:.4f}{sn['collision_ci']} (reg flips vs stored={reg['n_outcome_flips']}/{reg['n_common']})",flush=True)
        print(f"  k5:   cps={sk['cps']:.4f}{sk['cps_ci']} coll={sk['collision']:.4f}{sk['collision_ci']} | flips={len(fl)} fixed={R[tag]['fixed']} new_coll={R[tag]['new_coll']}",flush=True)
        print(f"  chatter none->k5: {ch}",flush=True)
    # S5a: v2.7.1 k5 chattering on the M5 collision episodes
    m5=REPO/"data/runs/v2.7.1/set__20260718-110403__seed42/v2.7.1__20260718-114933__seed42/checkpoints/best.pt"
    m5_coll=[int(r["episode_idx"]) for r in csv.DictReader(open(SP/"stage1_none_episodes.csv")) if r["outcome"]=="collision"]
    R["v2.7.1_k5_chatter_S5a"]=chatter(m5, m5_coll)
    print(f"\nS5a v2.7.1 none->k5 chatter: {R['v2.7.1_k5_chatter_S5a']}",flush=True)
    json.dump(R, open(SP/"stage1b_cross_ckpt.json","w"), indent=2, default=str)
    print("WROTE", SP/"stage1b_cross_ckpt.json", flush=True)

if __name__=="__main__":
    main()
