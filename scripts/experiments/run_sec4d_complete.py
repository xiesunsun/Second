"""Complete Section IV-D experiments on lab (zero test leakage, zero new LLM API calls):
  1) Component Ablation on all 12 clinical datasets x 20 repeats (n in {3,10,20,30}, NC + LR/RF at n=30):
       - w/o Prior (pure data-driven SP-PD search + R=10 repeated-pairing delivery, phi=0)
       - w/o Search (static 1-call LLM-Select Rank)
       - w/o Repeated Pairing (R=1 single-pairing LOPO, lam=0.5)
       - w/o Prior Reg. (R=10 repeated-pairing LOPO, lam=0.0)
       - Full SP-PD (R=10, lam=0.5)
  2) Parameter Sensitivity on all 12 clinical datasets x 20 repeats (n=30, NC + LR/RF + CPU ms):
       - lambda in {0.0, 0.25, 0.50, 0.75, 1.00, 1.50} at R=10
       - R in {1, 3, 5, 10, 20} at lambda=0.50
  3) 5-Run Stochastic LLM Stability (rep 0..4, Gemini-3.8-Flash T=0 & T=0.3) on 6 clinical datasets (n=3, 30, B=128).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
ROOT = Path("/home/sunxie/paper-experiments-20261124")
sys.path.insert(0, str(ROOT))
import run_clinical_non_llm as clin
from trial import read


def repeated_lopo_per_rep(x, y, masks, rs, max_reps=20):
    """Run up to max_reps stratified LOPO pairings, returning per-rep (max_reps, 2n) correctness & signed margins
    plus cumulative wall-clock time per rep for exact R-sensitivity timing!
    """
    rng = np.random.RandomState(rs + 7777)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    n = len(idx0)
    y_sign = np.where(y == 1, 1.0, -1.0)
    preds = {m: np.empty((max_reps, len(y)), dtype=np.float32) for m in masks}
    smarg = {m: np.empty((max_reps, len(y)), dtype=np.float32) for m in masks}
    cum_ms = np.empty(max_reps, dtype=np.float64)
    t0 = perf_counter()
    for rep in range(max_reps):
        p0 = rng.permutation(idx0)
        p1 = rng.permutation(idx1)
        m_mat = np.zeros_like(x, dtype=float)
        for f in range(n):
            te = np.array([p0[f], p1[f]])
            tr = np.ones(len(y), dtype=bool)
            tr[te] = False
            m_mat[te] = clin.margins(x[tr], y[tr], x[te])
        for m in masks:
            sv = m_mat[:, list(m)].sum(axis=1)
            preds[m][rep] = ((sv > 0) == (y == 1)).astype(np.float32)
            smarg[m][rep] = (sv * y_sign).astype(np.float32)
        cum_ms[rep] = (perf_counter() - t0) * 1000.0
    return preds, smarg, cum_ms


def select_eb_k4(shortlist, preds, smarg, n_reps, lam, p, n, phi_fn):
    dim_factor = np.sqrt(np.log(max(2.0, p / 2.0))) / np.sqrt(2.0 * n)
    best_m, best_key = None, None
    for m in shortlist:
        acc = float(preds[m][:n_reps].mean())
        sm_arr = smarg[m][:n_reps].ravel()
        sm = float(np.tanh(sm_arr / (0.5 * (np.std(sm_arr) + 1e-6))).mean())
        ph = phi_fn(m) if phi_fn is not None else 0.0
        key = (acc + 0.03 * sm + lam * dim_factor * ph, ph)
        if best_key is None or key > best_key:
            best_key = key
            best_m = m
    return best_m


def build_k4_shortlist(bk, path, a_mask, phi_fn):
    eval_ids = list(dict.fromkeys(path))
    y_sign = np.where(bk.y == 1, 1.0, -1.0)
    col_m = bk.margin * y_sign[:, None]
    acc_sc = {j: bk.scores[j]["fold"] for j in eval_ids}
    cont_sc = {}
    for j in eval_ids:
        sv = col_m[:, list(bk.masks[j])].sum(axis=1)
        cont_sc[j] = float(np.tanh(sv / (0.5 * (np.std(sv) + 1e-6))).mean())
    full_ids = [j for j in eval_ids if len(bk.masks[j]) == clin.CAP]
    if not full_ids:
        full_ids = eval_ids
    top_pure = sorted(full_ids, key=lambda j: -(acc_sc[j] + 0.15 * cont_sc[j]))[:10]
    if phi_fn is not None:
        top_tilt = sorted(full_ids, key=lambda j: -(acc_sc[j] + 0.10 * cont_sc[j] + 0.25 * phi_fn(bk.masks[j])))[:6]
        sl = [a_mask] + [tuple(bk.masks[j]) for j in top_pure + top_tilt]
    else:
        sl = [tuple(bk.masks[j]) for j in top_pure]
    return list(dict.fromkeys(sl))


def eval_ablation_and_sens(job):
    t, r = job
    q_outer, v_outer, _ = clin.load_role(t, "outer")
    orders, p = clin.load_gemini_orders(t)
    abl_nc, abl_dn = [], []
    sens_lam, sens_R = [], []

    for n in clin.NS:
        x, y, ids = clin.get_donors(t, r, n)
        rs = clin.seed("clinical-EA-20261123", t, r, n)
        d_outer = clin.margins(x, y, q_outer)
        d_outer_t = np.ascontiguousarray(d_outer.T)
        outer_cache = {}
        def score_ba(mask):
            mask = tuple(mask)
            if mask not in outer_cache:
                pred = d_outer_t[list(mask)].sum(0) > 0
                outer_cache[mask] = float(clin.measures(pred, v_outer)[0])
            return outer_cache[mask]
        down = clin.Downstream(x, y, q_outer, v_outer, clin.seed("clinical-readout-20261123", t, r, n)) if n == 30 else None

        # 1) w/o Prior: pure data-driven SP-PD (order=None) + R=10 repeated-pairing delivery
        t_s0 = perf_counter()
        bk_np, st_np, _, _ = clin.run_frozen_ext(x, y, clin.CAP, 128, None, rs)
        search_ms_np = (perf_counter() - t_s0) * 1000.0
        sl_np = build_k4_shortlist(bk_np, st_np["path"], None, None)
        pr_np, sm_np, _ = repeated_lopo_per_rep(x, y, sl_np, rs, max_reps=10)
        m_wo_prior = select_eb_k4(sl_np, pr_np, sm_np, 10, 0.0, p, n, None)
        ba_wo_prior = score_ba(m_wo_prior)
        for pk in ("correct", "wrong"):
            abl_nc.append((t, r, n, pk, "wo_prior", ba_wo_prior))
        if down is not None:
            ds_np = down.scores(m_wo_prior)
            for pk in ("correct", "wrong"):
                abl_dn.append((t, r, pk, "wo_prior", float(ds_np["lr"][0]), float(ds_np["rf"][0])))

        # 2) Prior-guided variants (Aligned & Misaligned)
        max_R = 20 if n == 30 else 10
        for pk in ("correct", "wrong"):
            order = orders[pk]
            t_s0 = perf_counter()
            bk, st, ret, _ = clin.run_frozen_ext(x, y, clin.CAP, 128, order, rs)
            search_ms = (perf_counter() - t_s0) * 1000.0
            a_mask = tuple(bk.masks[ret["anchor"]])

            topK = set(int(f) for f in order[:clin.CAP])
            top2K = set(int(f) for f in order[:2 * clin.CAP])
            def phi_fullK(m):
                return sum(1.0 if f in topK else (0.5 if f in top2K else 0.0) for f in m) / float(clin.CAP)

            sl = build_k4_shortlist(bk, st["path"], a_mask, phi_fullK)
            preds, smarg, cum_ms = repeated_lopo_per_rep(x, y, sl, rs, max_reps=max_R)

            # Ablation variants
            m_wo_search = a_mask
            m_wo_rep = select_eb_k4(sl, preds, smarg, 1, 0.50, p, n, phi_fullK)
            m_wo_reg = select_eb_k4(sl, preds, smarg, 10, 0.00, p, n, phi_fullK)
            m_full = select_eb_k4(sl, preds, smarg, 10, 0.50, p, n, phi_fullK)

            for vname, m in [
                ("wo_search", m_wo_search),
                ("wo_rep", m_wo_rep),
                ("wo_reg", m_wo_reg),
                ("full_sppd", m_full),
            ]:
                abl_nc.append((t, r, n, pk, vname, score_ba(m)))

            if down is not None:
                # Downstream for ablation variants (cache by mask!)
                dn_cache = {}
                def get_dn(m):
                    if m not in dn_cache:
                        d = down.scores(m)
                        dn_cache[m] = (float(d["lr"][0]), float(d["rf"][0]))
                    return dn_cache[m]
                for vname, m in [
                    ("wo_search", m_wo_search),
                    ("wo_rep", m_wo_rep),
                    ("wo_reg", m_wo_reg),
                    ("full_sppd", m_full),
                ]:
                    lr_v, rf_v = get_dn(m)
                    abl_dn.append((t, r, pk, vname, lr_v, rf_v))

                # Sensitivity over lambda (R=10)
                for lam in (0.0, 0.25, 0.50, 0.75, 1.00, 1.50):
                    ml = select_eb_k4(sl, preds, smarg, 10, lam, p, n, phi_fullK)
                    lr_v, rf_v = get_dn(ml)
                    sens_lam.append((t, r, pk, lam, score_ba(ml), lr_v, rf_v))

                # Sensitivity over R (lambda=0.50) + exact wall-clock ms
                for R_val in (1, 3, 5, 10, 20):
                    mr = select_eb_k4(sl, preds, smarg, R_val, 0.50, p, n, phi_fullK)
                    lr_v, rf_v = get_dn(mr)
                    sens_R.append((t, r, pk, R_val, score_ba(mr), lr_v, rf_v, float(cum_ms[R_val - 1]), search_ms))

    return abl_nc, abl_dn, sens_lam, sens_R


def eval_stability_eb(job):
    """Evaluate 5 independent Gemini-3.8-Flash cards (rep 0=T=0, rep 1..4=T=0.3) on new-6 at n in {3, 30}, B=128."""
    t, r = job
    q_outer, v_outer, _ = clin.load_role(t, "outer")
    frozen_orders, p = clin.load_gemini_orders(t)
    rows = []
    for n in (3, 30):
        x, y, _ = clin.get_donors(t, r, n)
        rs = clin.seed("clinical-EA-20261123", t, r, n)
        d_outer = clin.margins(x, y, q_outer)
        d_outer_t = np.ascontiguousarray(d_outer.T)
        outer_cache = {}
        def score_ba(mask):
            mask = tuple(mask)
            if mask not in outer_cache:
                pred = d_outer_t[list(mask)].sum(0) > 0
                outer_cache[mask] = float(clin.measures(pred, v_outer)[0])
            return outer_cache[mask]
        for pk in ("correct", "wrong"):
            for rep in range(5):
                if rep == 0:
                    ord_vec = frozen_orders[pk]
                else:
                    card = read(ROOT / "llm_cards/stability" / t / pk / f"rep-{rep}" / "card.json")
                    ol = list(map(int, card["order"]))
                    ord_vec = np.array(ol + [j for j in range(p) if j not in set(ol)], dtype=int)
                bk, st, ret, _ = clin.run_frozen_ext(x, y, clin.CAP, 128, ord_vec, rs)
                a_mask = tuple(bk.masks[ret["anchor"]])
                topK = set(int(f) for f in ord_vec[:clin.CAP])
                top2K = set(int(f) for f in ord_vec[:2 * clin.CAP])
                def phi_fullK(m):
                    return sum(1.0 if f in topK else (0.5 if f in top2K else 0.0) for f in m) / float(clin.CAP)
                sl = build_k4_shortlist(bk, st["path"], a_mask, phi_fullK)
                preds, smarg, _ = repeated_lopo_per_rep(x, y, sl, rs, max_reps=10)
                m_sp = select_eb_k4(sl, preds, smarg, 10, 0.50, p, n, phi_fullK)
                rows.append((t, r, n, pk, rep, score_ba(m_sp), score_ba(a_mask)))
    return rows


def w_p(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.all(np.abs(d) < 1e-12):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return 1.0


if __name__ == "__main__":
    print("=== PART 1 & 2: ABLATION + SENSITIVITY (12 DATASETS x 20 REPEATS) ===")
    jobs = [(t, r) for t in clin.ALL_TASKS for r in range(clin.REPEATS)]
    with ProcessPoolExecutor(max_workers=20) as ex:
        res = list(ex.map(eval_ablation_and_sens, jobs))
    abl_nc = pd.DataFrame([x for a, _, _, _ in res for x in a], columns=["t", "r", "n", "prior", "variant", "ba"])
    abl_dn = pd.DataFrame([x for _, b, _, _ in res for x in b], columns=["t", "r", "prior", "variant", "lr", "rf"])
    sl_df = pd.DataFrame([x for _, _, c, _ in res for x in c], columns=["t", "r", "prior", "lam", "nc", "lr", "rf"])
    sr_df = pd.DataFrame([x for _, _, _, d in res for x in d], columns=["t", "r", "prior", "R", "nc", "lr", "rf", "deliv_ms", "search_ms"])

    # Aggregate & save clean TSVs
    agg_abl_nc = abl_nc.groupby(["t", "n", "prior", "variant"], as_index=False)["ba"].mean()
    agg_abl_dn = abl_dn.groupby(["t", "prior", "variant"], as_index=False)[["lr", "rf"]].mean()
    agg_lam = sl_df.groupby(["t", "prior", "lam"], as_index=False)[["nc", "lr", "rf"]].mean()
    agg_R = sr_df.groupby(["t", "prior", "R"], as_index=False)[["nc", "lr", "rf", "deliv_ms", "search_ms"]].mean()

    agg_abl_nc.to_csv(ROOT / "results/sec4d_ablation_nc.tsv", sep="\t", index=False)
    agg_abl_dn.to_csv(ROOT / "results/sec4d_ablation_down.tsv", sep="\t", index=False)
    agg_lam.to_csv(ROOT / "results/sec4d_sens_lambda.tsv", sep="\t", index=False)
    agg_R.to_csv(ROOT / "results/sec4d_sens_R.tsv", sep="\t", index=False)

    # Print Ablation Table (n=30 NC Aln/Mis/Mean + LR Mean + RF Mean + n=3 Aln/Mis + Wilcoxon p vs full_sppd)
    tasks = sorted(clin.ALL_TASKS)
    variants = ["wo_prior", "wo_search", "wo_rep", "wo_reg", "full_sppd"]
    f_nc_a = agg_abl_nc[(agg_abl_nc["variant"]=="full_sppd") & (agg_abl_nc["n"]==30) & (agg_abl_nc["prior"]=="correct")].set_index("t").loc[tasks, "ba"]
    f_nc_m = agg_abl_nc[(agg_abl_nc["variant"]=="full_sppd") & (agg_abl_nc["n"]==30) & (agg_abl_nc["prior"]=="wrong")].set_index("t").loc[tasks, "ba"]
    f_nc_avg = 0.5 * (f_nc_a + f_nc_m)
    f_dn_a = agg_abl_dn[(agg_abl_dn["variant"]=="full_sppd") & (agg_abl_dn["prior"]=="correct")].set_index("t").loc[tasks]
    f_dn_m = agg_abl_dn[(agg_abl_dn["variant"]=="full_sppd") & (agg_abl_dn["prior"]=="wrong")].set_index("t").loc[tasks]
    f_lr_avg = 0.5 * (f_dn_a["lr"] + f_dn_m["lr"])
    f_rf_avg = 0.5 * (f_dn_a["rf"] + f_dn_m["rf"])

    print("\n--- ABLATION SUMMARY (n=30, 12 clinical datasets) ---")
    for v in variants:
        va = agg_abl_nc[(agg_abl_nc["variant"]==v) & (agg_abl_nc["n"]==30) & (agg_abl_nc["prior"]=="correct")].set_index("t").loc[tasks, "ba"]
        vm = agg_abl_nc[(agg_abl_nc["variant"]==v) & (agg_abl_nc["n"]==30) & (agg_abl_nc["prior"]=="wrong")].set_index("t").loc[tasks, "ba"]
        vavg = 0.5 * (va + vm)
        v3a = agg_abl_nc[(agg_abl_nc["variant"]==v) & (agg_abl_nc["n"]==3) & (agg_abl_nc["prior"]=="correct")].set_index("t").loc[tasks, "ba"]
        v3m = agg_abl_nc[(agg_abl_nc["variant"]==v) & (agg_abl_nc["n"]==3) & (agg_abl_nc["prior"]=="wrong")].set_index("t").loc[tasks, "ba"]
        da = agg_abl_dn[(agg_abl_dn["variant"]==v) & (agg_abl_dn["prior"]=="correct")].set_index("t").loc[tasks]
        dm = agg_abl_dn[(agg_abl_dn["variant"]==v) & (agg_abl_dn["prior"]=="wrong")].set_index("t").loc[tasks]
        vlr = 0.5 * (da["lr"] + dm["lr"])
        vrf = 0.5 * (da["rf"] + dm["rf"])
        p_nc = w_p(f_nc_avg, vavg) if v != "full_sppd" else 1.0
        p_lr = w_p(f_lr_avg, vlr) if v != "full_sppd" else 1.0
        p_rf = w_p(f_rf_avg, vrf) if v != "full_sppd" else 1.0
        print(f"{v:12s} | n3(A/M)={v3a.mean()*100:.2f}/{v3m.mean()*100:.2f} | n30 NC(A/M/Mean)={va.mean()*100:.2f}/{vm.mean()*100:.2f}/{vavg.mean()*100:.2f} (p={p_nc:.3f}) | LR_M={vlr.mean()*100:.2f} (p={p_lr:.3f}) | RF_M={vrf.mean()*100:.2f} (p={p_rf:.3f})")

    print("\n--- SENSITIVITY: LAMBDA (R=10, n=30) ---")
    pl = agg_lam.groupby(["lam", "prior"], as_index=False)[["nc", "lr", "rf"]].mean()
    pl_nc = pl.pivot(index="lam", columns="prior", values="nc") * 100
    pl_lr = pl.pivot(index="lam", columns="prior", values="lr") * 100
    pl_rf = pl.pivot(index="lam", columns="prior", values="rf") * 100
    out_l = pd.DataFrame({
        "NC_Aln": pl_nc["correct"], "NC_Mis": pl_nc["wrong"], "NC_Mean": 0.5*(pl_nc["correct"]+pl_nc["wrong"]),
        "LR_Mean": 0.5*(pl_lr["correct"]+pl_lr["wrong"]), "RF_Mean": 0.5*(pl_rf["correct"]+pl_rf["wrong"]),
    })
    print(out_l.round(2).to_string())

    print("\n--- SENSITIVITY: R (lambda=0.50, n=30) ---")
    pr = agg_R.groupby(["R", "prior"], as_index=False)[["nc", "lr", "rf", "deliv_ms", "search_ms"]].mean()
    pr_nc = pr.pivot(index="R", columns="prior", values="nc") * 100
    pr_lr = pr.pivot(index="R", columns="prior", values="lr") * 100
    pr_rf = pr.pivot(index="R", columns="prior", values="rf") * 100
    pr_ms = agg_R.groupby("R")[["deliv_ms", "search_ms"]].mean()
    out_r = pd.DataFrame({
        "NC_Aln": pr_nc["correct"], "NC_Mis": pr_nc["wrong"], "NC_Mean": 0.5*(pr_nc["correct"]+pr_nc["wrong"]),
        "LR_Mean": 0.5*(pr_lr["correct"]+pr_lr["wrong"]), "RF_Mean": 0.5*(pr_rf["correct"]+pr_rf["wrong"]),
        "deliv_ms": pr_ms["deliv_ms"], "search_ms": pr_ms["search_ms"],
    })
    print(out_r.round(2).to_string())

    print("\n=== PART 3: 5-RUN STOCHASTIC LLM STABILITY (NEW-6, B=128, n in {3, 30}) ===")
    sjobs = [(t, r) for t in clin.NEW_SIX for r in range(clin.REPEATS)]
    with ProcessPoolExecutor(max_workers=20) as ex:
        sres = list(ex.map(eval_stability_eb, sjobs))
    sdf = pd.DataFrame([x for sub in sres for x in sub], columns=["t", "r", "n", "prior", "rep", "sppd", "static"])
    # Mean across 20 repeats per (t, n, prior, rep), then mean & SD across 5 independent LLM prompts!
    s_rep = sdf.groupby(["t", "n", "prior", "rep"], as_index=False)[["sppd", "static"]].mean()
    s_agg = s_rep.groupby(["t", "n", "prior"], as_index=False).agg(
        sppd_mean=("sppd", "mean"), sppd_sd=("sppd", "std"),
        static_mean=("static", "mean"), static_sd=("static", "std"),
    )
    for col in ("sppd_mean", "sppd_sd", "static_mean", "static_sd"):
        s_agg[col] *= 100.0
    s_agg.to_csv(ROOT / "results/sec4d_stability_eb.tsv", sep="\t", index=False)
    print(s_agg.groupby(["n", "prior"])[["sppd_mean", "sppd_sd", "static_mean", "static_sd"]].mean().round(2).to_string())
