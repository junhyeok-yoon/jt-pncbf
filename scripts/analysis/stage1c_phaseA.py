"""v2.7.1 Stage-1c Phase A — DI + unicycle SOTA mode=none regression + empty-branch rate (eval-only)."""
import csv, json
from pathlib import Path
import numpy as np
from src.eval.evaluate import evaluate
from src.frameworks.jt_pncbf.train import load_framework_from_checkpoint
REPO=Path("/home/junhyeok/MIT/jt-pncbf"); SP=Path("/tmp/claude-1000/-home-junhyeok-MIT-jt-pncbf/31d93785-ac11-4206-bf50-a4c3de145dff/scratchpad")
SYS={
 "DI_v2.3.0_s42":  (REPO/"data/secured_data/v2.3.0/seed42/checkpoints/best.pt",
                    REPO/"data/secured_data/pools/eval_full_di_n2000_seed23456.pkl",
                    REPO/"data/secured_data/v2.3.0/seed42/eval_episodes.csv"),
 "UNI_v2.2.2_28k": (REPO/"data/runs/v2.2.2/v2.2.2__20260619-083424__seed42/checkpoints/best.pt",
                    REPO/"data/secured_data/pools/eval_full_unicycle_n2000_seed23456.pkl", None),
}
def summ(rows):
    import numpy as np
    return dict(n=len(rows), cps=float(np.mean([float(r['cps_episode']) for r in rows])),
        collision=float(np.mean([r['collision'] for r in rows])), reach=float(np.mean([r['reach'] for r in rows])),
        empty=float(np.mean([float(r.get('empty_step_frac',0)) for r in rows])),
        empty_coll=float(np.mean([float(r.get('empty_step_frac',0)) for r in rows if r['outcome']=='collision']) if any(r['outcome']=='collision' for r in rows) else 0),
        fire_eps=int(sum(1 for r in rows if float(r.get('empty_step_frac',0))>0)),
        n_coll=int(sum(1 for r in rows if r['outcome']=='collision')))
out={}
for tag,(ckpt,pool,stored) in SYS.items():
    fw,cfg,ck=load_framework_from_checkpoint(ckpt, config_overrides={"filter":{"empty_fallback":{"mode":"none","k":5}}})
    res=evaluate(fw,pool,cfg,mode="final",step=int(ck.get("step",0)),ckpt_name="best.pt",include_lqr_baseline=False)
    rows=[r for r in res.episode_rows if r.get("mode")=="final"]
    s=summ(rows); s["system"]=cfg["run"]["system"]; s["box_aware"]=cfg["filter"]["hardnet"].get("box_aware")
    # regression vs stored eval_episodes (if present)
    if stored and Path(stored).exists():
        st={int(r["episode_idx"]):r["outcome"] for r in csv.DictReader(open(stored)) if r.get("mode")=="final"}
        nr={int(r["episode_idx"]):r["outcome"] for r in rows}; common=set(st)&set(nr)
        s["regression_flips"]=int(sum(1 for i in common if st[i]!=nr[i])); s["regression_n"]=len(common)
    out[tag]=s
    # save none rows for Phase B flip base
    with open(SP/f"stage1c_{tag}_none_episodes.csv","w",newline="") as f:
        import csv as _c; w=_c.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{tag} [{s['system']}]: cps={s['cps']:.4f} coll={s['collision']:.4f} reach={s['reach']:.4f} box_aware={s['box_aware']} "
          f"empty={s['empty']:.4f} empty@coll={s['empty_coll']:.4f} fire_eps={s['fire_eps']} n_coll={s['n_coll']} "
          f"reg_flips={s.get('regression_flips','n/a')}/{s.get('regression_n','')}", flush=True)
json.dump(out, open(SP/"stage1c_phaseA.json","w"), indent=2, default=str)
print("WROTE", SP/"stage1c_phaseA.json", flush=True)
