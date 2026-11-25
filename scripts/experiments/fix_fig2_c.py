from pathlib import Path
p = Path("/home/sunxie/paper-experiments-20261124/plot_tevc_fig2.py")
txt = p.read_text()
old = """prior_pts = [
    ("SP-PD", "gemini_paired_tau15", "wrong_paired_tau15", (-22, 6)),
    ("LLM-Select (Rank)", "gemini_static", "wrong_static", (4, -10)),
    ("LLM-Select (Score)", "llm_select_score_correct", "llm_select_score_wrong", (4, 3)),
    ("LLM-Lasso", "llm_lasso_correct", "llm_lasso_wrong", (-38, 6)),
    ("LSP", "lsp_adapted_correct", "lsp_adapted_wrong", (-22, -11)),
    (r"$\\pi$BO", "pibo_correct", "pibo_wrong", (-20, 3)),
]
short_lbl = {
    "SP-PD": "SP-PD",
    "LLM-Select (Rank)": "Rank",
    "LLM-Select (Score)": "Score",
    "LLM-Lasso": "LLM-Lasso",
    "LSP": "LSP",
    r"$\\pi$BO": r"$\\pi$BO",
}"""
new = """prior_pts = [
    ("SP-PD", "gemini_paired_tau15", "wrong_paired_tau15", (-18, 7)),
    ("LLM-Select (Rank)", "gemini_static", "wrong_static", (5, -3)),
    ("LLM-Select (Score)", "llm_select_score_correct", "llm_select_score_wrong", (5, -2)),
    ("LLM-Lasso", "llm_lasso_correct", "llm_lasso_wrong", (-46, 5)),
    ("LSP", "lsp_adapted_correct", "lsp_adapted_wrong", (-23, -10)),
    (r"$\\pi$BO", "pibo_correct", "pibo_wrong", (-24, -4)),
]
short_lbl = {
    "SP-PD": "SP-PD",
    "LLM-Select (Rank)": "Select (Rank)",
    "LLM-Select (Score)": "Select (Score)",
    "LLM-Lasso": "LLM-Lasso",
    "LSP": "LSP",
    r"$\\pi$BO": r"$\\pi$BO",
}"""
assert old in txt, "old block not found"
txt = txt.replace(old, new)
old2 = """ax.annotate("Non-LLM\\nbaselines", (65.25, 65.15), xytext=(-26, -24), textcoords="offset points",
            fontsize=6.4, color="#636363",
            arrowprops=dict(arrowstyle="-", color="#969696", lw=0.6))"""
new2 = """ax.annotate("Non-LLM baselines", (65.05, 65.12), xytext=(-68, -2), textcoords="offset points",
            fontsize=6.5, color="#636363",
            arrowprops=dict(arrowstyle="->", color="#969696", lw=0.6))"""
assert old2 in txt, "old2 block not found"
txt = txt.replace(old2, new2)
p.write_text(txt)
