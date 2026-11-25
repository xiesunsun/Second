# SP-PD (IEEE TEVC Working Manuscript & Experiment Archive)

面向有限评价预算特征选择的离线语义先验引导进化搜索与重复配对交付框架（**SP-PD**）论文源码、写作规范与完整实验归档（私有仓库）。

## 目录结构

- `main_tevc.tex` / `main_tevc.pdf`：*IEEE Transactions on Evolutionary Computation*（`IEEEtran` 双栏）主手稿与最新编译样稿
- `sections/`：各节中英文手稿与附录（已完成 Section IV-A 至 IV-D 与 Appendices A–C）
- `tables/`：LaTeX 表格源码（`Table I` 至 `Table VII`）
- `figures/`：STIX 衬线纯矢量图（`Fig. 1` 至 `Fig. 3`）
- `writing_guides/`：论文撰写纪律（`writing_essay.md`）与全局锁定术语表（`terminology.md`）
- `scripts/core/`：`SP-PD` 核心算法源码（`selector.py`、`handoff.py`、`engine.py`、`strong.py`、`clinical_trial.py`）
- `scripts/experiments/`：在远程服务器 `lab` 上运行的全部高维基准、临床实验、消融/敏感性与矢量图绘制脚本
- `results/`：全部已固化汇总结果（`.tsv` / `.json`）
- `llm_cards/`：`gemini-3.8-flash` 离线缓存卡片（Rank / Score / Lasso / 5-run Stability）
- `reference_papers/`：对标顶刊参考论文（仅私有留存）

## 本地编译 PDF

```bash
pdflatex main_tevc.tex
bibtex main_tevc
pdflatex main_tevc.tex
pdflatex main_tevc.tex
```
