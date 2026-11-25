"""Generate strict IEEE Transactions (TKDE/TEVC) LaTeX tables with booktabs, threeparttable,
bold/underline ranking highlights, and exact paired significance markers directly from verified TSVs.
"""
from pathlib import Path
import numpy as np
import pandas as pd

EXP_ROOT = Path("/home/sunxie/paper-experiments-20261124")
OLD_CLIN = Path("/home/sunxie/clinical-prospective-20261123.cjUGun3X")
NEW_CLIN = Path("/home/sunxie/clinical-extension-20261124")
ARCH_ROOT = Path("/home/sunxie/paper-archive-20261124")
OUT_DIR = ARCH_ROOT / "ieee_draft/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_all_tables():
    # Load verified TSVs
    df_c = pd.read_csv(ARCH_ROOT / "01_protocols_and_provenance/datasets_clinical_12.tsv", sep="\t")
    df_t2 = pd.read_csv(ARCH_ROOT / "02_tables_data_and_latex/table2_clinical_main_comparison.tsv", sep="\t")
    df_stat = pd.read_csv(ARCH_ROOT / "02_tables_data_and_latex/appendix/tableA4_pairwise_statistics.tsv", sep="\t")
    df_cost = pd.read_csv(ARCH_ROOT / "01_protocols_and_provenance/cost_and_runtime_ledger.tsv", sep="\t")
    df_t4 = pd.read_csv(ARCH_ROOT / "02_tables_data_and_latex/table4_scaling_and_downstream.tsv", sep="\t")
    df_t5 = pd.read_csv(ARCH_ROOT / "02_tables_data_and_latex/table5_numerical_b1000.tsv", sep="\t")

    # Helper to fetch significance marker for SP-PD vs method_b on All-12 B2-16 / B128
    stat_map = {}
    for _, r in df_stat.iterrows():
        if r["panel"] == "all_twelve" and r["method_a"] == "gemini_paired_ext":
            p = float(r["p"])
            hp = float(r["holm_p"]) if not pd.isna(r["holm_p"]) else 1.0
            mk = r"$^{\ddagger}$" if hp < 0.05 else (r"$^{\dagger}$" if p < 0.05 else "")
            stat_map[(r["band"], r["method_b"])] = (mk, int(r["wins"]), int(r["losses"]))

    # ---------------- TABLE I: 12 Clinical Cohorts (double-column compact) ----------------
    t1_lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{Summary of the 12 Prospective Clinical Binary Classification Cohorts ($n{=}3$/class Primary, $K{=}4$, $B \in [2,16]$ Extended to $B{=}128$).}",
        r"\label{tab:datasets_clinical}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5.5pt}",
        r"\begin{tabular}{@{}llrrrrrc@{}}",
        r"\toprule",
        r"Wave & Cohort & Features ($p$) & Eligible & Train Pool & Outer Test & Pos.\ Rate & Zero-History\tnote{a} \\",
        r"\midrule",
    ]
    for i, r in df_c.iterrows():
        if i == 6:
            t1_lines.append(r"\midrule")
        wv = "Earlier-6" if i == 0 else ("New-6" if i == 6 else "")
        zh = r"\checkmark" if r["zero_history"] == "Yes" else r"--"
        t1_lines.append(
            f"{wv} & \\texttt{{{r['task']}}} & {int(r['fields_p'])} & {int(r['total_eligible']):,} & "
            f"{int(r['train_pool']):,} & {int(r['outer_test']):,} & {r['pos_rate']*100:.1f}\\% & {zh} \\\\"
        )
    t1_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\scriptsize",
        r"\item[a] Verified zero file/matrix overlap against 365,793 workspace entries and 8,267 historical matrices (forming the \emph{Zero-History-8} sensitivity panel).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ]
    (OUT_DIR / "tab1_datasets.tex").write_text("\n".join(t1_lines) + "\n")

    # ---------------- TABLE II: Main Clinical Comparison (clean 14-row IEEE table*) ----------------
    t2_by_name = {r["method"]: r for _, r in df_t2.iterrows()}
    main_groups = [
        ("Classical Filter \\& Embedded", [
            ("mRMR", r"mRMR~\cite{peng2005mrmr}", "0", "mrmr_static"),
            ("ReliefF", r"ReliefF~\cite{robnik2003relieff}", "0", "relieff_static"),
            ("L1-LR (LASSO)", r"L1-LR (LASSO)~\cite{tibshirani1996lasso}", "0", "l1_lr_static"),
            ("RF-Imp", r"RF-Imp~\cite{breiman2001rf}", "0", "rf_imp_static"),
        ]),
        ("Standard \\& SOTA Evolutionary FS", [
            ("GA ($B\\le 128$)", r"GA", "0", "ga"),
            ("BPSO ($B\\le 128$)", r"BPSO", "0", "bpso"),
            ("EDA / UMDA ($B\\le 128$)", r"EDA (UMDA)~\cite{larranaga2024edasurvey}", "0", "eda"),
            ("MEL (TKDE, $B\\le 128$)", r"MEL~\cite{wang2024mel}", "0", "mel"),
        ]),
        ("Zero-Shot Static References (No Training Data for FS)", [
            ("LLM-Select (Rank, Static)", r"LLM-Select (Rank)~\cite{jeong2025llmselect}", "12", None),
            ("LLM-Select (Score, Static)", r"LLM-Select (Score)~\cite{jeong2025llmselect}", r"38 (947)\tnote{a}", None),
        ]),
        ("Data-Adaptive Prior-Guided \\& LLM-FS", [
            ("PiBO (ICLR'22, Same Rank Prior)", r"$\pi$BO~\cite{hvarfner2022pibo} (Same Prior)", "12", "pibo_correct"),
            ("LLM-Lasso (Weighted L1)", r"LLM-Lasso~\cite{zhang2025llmlasso}", "38", "llm_lasso_correct"),
            ("LSP-Adapted (Discounted Prior)", r"LSP-Adapted~\cite{skinner2026lsp}", r"38\tnote{b}", "lsp_adapted_correct"),
            ("Ours (Paired Delivery)", r"\textbf{SP-PD (Proposed)}", r"\textbf{12}", None),
            ("Ours + Score Prior (Paired)", r"SP-PD (Score Prior)", r"38\tnote{b}", None),
        ]),
    ]

    # Determine best & second-best among all non-static rows for bold/underline
    adaptive_keys = [k for g, items in main_groups if "Zero-Shot" not in g for k, _, _, _ in items]
    cols_check = ["earlier6_correct_b2_16", "new6_correct_b2_16", "zero8_correct_b2_16",
                  "all12_correct_b2_16", "all12_correct_b128", "all12_wrong_b2_16", "all12_mix_q70"]
    top2 = {}
    for col in cols_check:
        vals = sorted({round(float(t2_by_name[k][col]), 2) for k in adaptive_keys}, reverse=True)
        top2[col] = (vals[0], vals[1] if len(vals) > 1 else -1)

    def fmt_val(v, col, is_static=False, sig=""):
        s = f"{v:.2f}"
        if not is_static:
            rv = round(v, 2)
            if rv == top2[col][0]:
                s = f"\\textbf{{{s}}}"
            elif rv == top2[col][1]:
                s = f"\\underline{{{s}}}"
        return s + sig

    t2_lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{Main Clinical Comparison Across 12 Prospective Cohorts ($n{=}3$/class, $K{=}4$, Outer Balanced Accuracy \%). Best and Second-Best Data-Adaptive Scores Are \textbf{Bolded} and \underline{Underlined}, Respectively.}",
        r"\label{tab:clinical_main}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.8pt}",
        r"\begin{tabular}{@{}lcrrrrrrr@{}}",
        r"\toprule",
        r"Method & LLM Calls & Earlier-6 & New-6 & Zero-Hist-8 & All-12 ($B_{2\text{--}16}$) & All-12 ($B_{128}$) & Misaligned & Mix ($q{=}0.7$) \\",
        r"\midrule",
    ]
    for gi, (gname, items) in enumerate(main_groups):
        if gi > 0:
            t2_lines.append(r"\midrule")
        t2_lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{gname}}}}} \\\\")
        is_stat = "Zero-Shot" in gname
        for key, disp, calls, skey in items:
            r = t2_by_name[key]
            sig16 = stat_map.get(("b2_16", skey), ("", 0, 0))[0] if skey else ""
            sig128 = stat_map.get(("b128", skey), ("", 0, 0))[0] if skey else ""
            t2_lines.append(
                f"{disp} & {calls} & "
                f"{fmt_val(float(r['earlier6_correct_b2_16']), 'earlier6_correct_b2_16', is_stat)} & "
                f"{fmt_val(float(r['new6_correct_b2_16']), 'new6_correct_b2_16', is_stat)} & "
                f"{fmt_val(float(r['zero8_correct_b2_16']), 'zero8_correct_b2_16', is_stat)} & "
                f"{fmt_val(float(r['all12_correct_b2_16']), 'all12_correct_b2_16', is_stat, sig16)} & "
                f"{fmt_val(float(r['all12_correct_b128']), 'all12_correct_b128', is_stat, sig128)} & "
                f"{fmt_val(float(r['all12_wrong_b2_16']), 'all12_wrong_b2_16', is_stat)} & "
                f"{fmt_val(float(r['all12_mix_q70']), 'all12_mix_q70', is_stat)} \\\\"
            )
    t2_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\scriptsize",
        r"\item[a] Batched ($\le 30$ fields/call) requires 38 calls; native per-feature scoring~\cite{jeong2025llmselect} requires $p{=}947$ calls across the 12 cohorts.",
        r"\item[b] Reuses \texttt{LLM-Select (Score)} outputs with zero additional API queries. $^{\dagger}$/$^{\ddagger}$ indicate \textbf{SP-PD} significantly outperforms the baseline on All-12 ($p{<}0.05$ / Holm-adjusted $p{<}0.05$, exact two-sided paired sign-flip test).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ]
    (OUT_DIR / "tab2_clinical_main.tex").write_text("\n".join(t2_lines) + "\n")

    # ---------------- TABLE III: Ablation & Delivery Mechanism (single/double compact table*) ----------------
    abl_rows = [
        ("Numeric Search (Ours w/o Prior)", r"SP-PD-Num (No Semantic Prior)", "0", "numeric_ext"),
        ("Lexical TF-IDF + Paired", r"Lexical TF-IDF + Paired Gate", "0", None),
        ("Ours w/o Delivery Gate (argmax)", r"SP-PD (argmax, w/o Gate)", "12", "gemini_raw_ext"),
        ("Ours Tie-Break Only", r"SP-PD (Tie-Break Only)", "12", "gemini_tied_ext"),
        ("Ours w/o Error Mutation", r"SP-PD (w/o Error Mutation)", "12", None),
        ("Ours (Paired Delivery)", r"\textbf{SP-PD (Full Paired Gate)}", r"\textbf{12}", None),
    ]
    t3_lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{Component Ablation Across All 12 Clinical Cohorts ($n{=}3, K{=}4, B \in [2,16]$, Outer BA \%).}",
        r"\label{tab:ablation}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Variant & Aligned & Misaligned & $\Delta$ vs.\ Full & W/L\tnote{a} \\",
        r"\midrule",
    ]
    full_c = float(t2_by_name["Ours (Paired Delivery)"]["all12_correct_b2_16"])
    for key, disp, _, skey in abl_rows:
        r = t2_by_name[key]
        ac = float(r["all12_correct_b2_16"])
        aw = float(r["all12_wrong_b2_16"])
        if key == "Ours (Paired Delivery)":
            t3_lines.append(f"{disp} & \\textbf{{{ac:.2f}}} & \\textbf{{{aw:.2f}}} & -- & -- \\\\")
        else:
            delta = ac - full_c
            if skey and ("b2_16", skey) in stat_map:
                mk, w, l = stat_map[("b2_16", skey)]
                wl = f"{w}/{l}{mk}"
            else:
                wl = "--"
            t3_lines.append(f"{disp} & {ac:.2f} & {aw:.2f} & {delta:+.2f} & {wl} \\\\")
    t3_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\scriptsize",
        r"\item[a] Cohort wins/losses of Full \textbf{SP-PD} over the ablated variant on Aligned Prior ($^{\ddagger}$Holm $p{<}0.05$).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (OUT_DIR / "tab3_ablation.tex").write_text("\n".join(t3_lines) + "\n")

    # ---------------- TABLE IV: LLM API Cost Ledger (compact single-column IEEE table!) ----------------
    t4_lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{Offline \texttt{gemini-3.8-flash} API Cost Across All 12 Clinical Cohorts ($p{=}947$ Fields, Amortized Over 20 Splits).}",
        r"\label{tab:llm_cost}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Method & Calls\tnote{a} & Out.\ Tok. & Think Tok. & Total Tok. & Time (s) \\",
        r"\midrule",
    ]
    order_cost = [
        ("ours_top8_rank", r"\textbf{SP-PD (Top-8)}"),
        ("llm_select_score", r"LLM-Select (Score)"),
        ("llm_lasso", r"LLM-Lasso"),
    ]
    cost_by_m = {r["method"]: r for _, r in df_cost.iterrows()}
    for mk, lbl in order_cost:
        r = cost_by_m[mk]
        cstr = f"{int(r['batched_calls'])}/{int(r['native_calls'])}"
        if mk == "ours_top8_rank":
            t4_lines.append(
                f"{lbl} & \\textbf{{{cstr}}} & \\textbf{{{int(r['output_tokens']):,}}} & "
                f"\\textbf{{{int(r['thoughts_tokens']):,}}} & \\textbf{{{int(r['total_tokens']):,}}} & \\textbf{{{r['api_seconds']:.1f}}} \\\\"
            )
        else:
            t4_lines.append(
                f"{lbl} & {cstr} & {int(r['output_tokens']):,} & "
                f"{int(r['thoughts_tokens']):,} & {int(r['total_tokens']):,} & {r['api_seconds']:.1f} \\\\"
            )
    t4_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\scriptsize",
        r"\item[a] Batched / native protocol calls. Online iterative LLM methods~\cite{yang2024icesearch} scale as $O(B \cdot R)$ calls per task.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (OUT_DIR / "tab4_llm_cost.tex").write_text("\n".join(t4_lines) + "\n")

    # ---------------- TABLE V: Scaling & Downstream Classifier Transfer (table*) ----------------
    name_map_t4 = {
        "LLM-Select (Rank, Static)": r"LLM-Select (Rank, Static)",
        "LLM-Select (Score, Static)": r"LLM-Select (Score, Static)",
        "Ours (Paired, 1 call/task)": r"\textbf{SP-PD (Proposed)}",
        "Ours + Score Prior (Paired)": r"SP-PD (Score Prior)",
        "PiBO (ICLR'22)": r"$\pi$BO~\cite{hvarfner2022pibo}",
        "LLM-Lasso (2025)": r"LLM-Lasso~\cite{zhang2025llmlasso}",
        "LSP-Adapted (2026)": r"LSP-Adapted~\cite{skinner2026lsp}",
        "ReliefF": r"ReliefF",
        "MEL (TKDE)": r"MEL~\cite{wang2024mel}",
        "Numeric Search": r"SP-PD-Num",
        "GA": r"GA",
        "BPSO": r"BPSO",
    }
    non_static_t4 = df_t4[~df_t4["method"].str.contains("Static")]
    t4_cols = ["n3_b2_16", "n3_b128", "n10_b2_16", "n10_b128", "lr_16", "lr_128", "rf_16", "rf_128"]
    t4_top2 = {c: sorted({round(float(v), 2) for v in non_static_t4[c]}, reverse=True)[:2] for c in t4_cols}

    t5_lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{Sample-Size Scaling ($n{=}3$ vs.\ $n{=}10$) and Independent Downstream Classifier Transfer (LR and RF at $n{=}10$) Across All 12 Clinical Cohorts (Outer BA \%).}",
        r"\label{tab:scaling_downstream}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5.2pt}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{$n{=}3$ (Centroid)} & \multicolumn{2}{c}{$n{=}10$ (Centroid)} & \multicolumn{2}{c}{$n{=}10$ (LR Transfer)} & \multicolumn{2}{c}{$n{=}10$ (RF Transfer)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Method & $B_{2\text{--}16}$ & $B_{128}$ & $B_{2\text{--}16}$ & $B_{128}$ & $B_{16}$ & $B_{128}$ & $B_{16}$ & $B_{128}$ \\",
        r"\midrule",
    ]
    for idx, r in df_t4.iterrows():
        if idx == 2:
            t5_lines.append(r"\midrule")
        is_st = "Static" in r["method"]
        cells = [name_map_t4.get(r["method"], r["method"])]
        for c in t4_cols:
            v = float(r[c])
            s = f"{v:.2f}"
            if not is_st:
                if round(v, 2) == t4_top2[c][0]:
                    s = f"\\textbf{{{s}}}"
                elif round(v, 2) == t4_top2[c][1]:
                    s = f"\\underline{{{s}}}"
            cells.append(s)
        t5_lines.append(" & ".join(cells) + r" \\")
    t5_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\scriptsize",
        r"\item Note: Top two rows are zero-shot static references; \textbf{bold} and \underline{underline} mark best and second-best among data-adaptive selectors.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ]
    (OUT_DIR / "tab5_scaling_transfer.tex").write_text("\n".join(t5_lines) + "\n")

    # ---------------- TABLE VI: Numerical Benchmark B=1..1000 (table*) ----------------
    num_lbl = {
        "tree_union": r"\textbf{SP-PD-Num (Proposed)}",
        "mel": r"MEL~\cite{wang2024mel}",
        "numeric": r"MW-Half Baseline",
        "iid": r"Strong-Init IID",
        "mrmr_static": r"mRMR~\cite{peng2005mrmr}",
        "l1_lr_static": r"L1-LR~\cite{tibshirani1996lasso}",
        "bpso": r"BPSO",
        "ga": r"GA",
    }
    t6_lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{High-Dimensional Tight-$K$ ($K \le 8$) Non-Semantic Numerical Benchmark Extended to $B{=}1000$ Evaluations (Source-Equal Weighted \% and Single-Core CPU Time).}",
        r"\label{tab:numerical_b1000}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.8pt}",
        r"\begin{tabular}{@{}llrrrrrrrr@{}}",
        r"\toprule",
        r"Suite & Method & Prec ($B_{8\text{--}128}$) & Prec@$128$ & Prec@$500$ & Prec@$1000$ & BA@$1000$ & F1@$1000$ & Pool Ceil & CPU (s) \\",
        r"\midrule",
    ]
    for si, (suite, slbl) in enumerate([("prospective_10", "Zero-Hist 10"), ("author_13", "Author 13")]):
        if si > 0:
            t6_lines.append(r"\midrule")
        sub = df_t5[df_t5["suite"] == suite]
        n_cols = ["p_8_128", "p_128", "p_500", "p_1000", "ba_1000", "f1_1000"]
        ntop = {c: sorted({round(float(v), 2) for v in sub[c]}, reverse=True)[:2] for c in n_cols}
        for ri, (_, r) in enumerate(sub.iterrows()):
            cells = [slbl if ri == 0 else "", num_lbl[r["method"]]]
            for c in n_cols:
                v = float(r[c])
                s = f"{v:.2f}"
                if round(v, 2) == ntop[c][0]:
                    s = f"\\textbf{{{s}}}"
                elif round(v, 2) == ntop[c][1]:
                    s = f"\\underline{{{s}}}"
                cells.append(s)
            cells.append(f"{float(r['ceil_1000']):.2f}")
            cells.append(f"{float(r['cpu_1000']):.2f}")
            t6_lines.append(" & ".join(cells) + r" \\")
    t6_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\scriptsize",
        r"\item Note: Zero-History 10 comprises 10 strictly deduplicated unseen sources (30 folds); Author 13 comprises 13 established sources across 14 tables (42 folds).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ]
    (OUT_DIR / "tab6_numerical_b1000.tex").write_text("\n".join(t6_lines) + "\n")
    print("ALL 6 STRICT IEEE LATEX TABLES GENERATED IN", OUT_DIR)


if __name__ == "__main__":
    build_all_tables()
