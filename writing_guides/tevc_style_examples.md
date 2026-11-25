# IEEE TEVC 写作参照样例（`tevc_style_examples.md`）

本文件汇总已下载 IEEE TEVC / TKDE 范文中的真实结构与句式模板，并与初版草稿的反面教训对照，供每节撰稿与自审直接参照。

---

## 一、Section IV 开篇导语参照（硬上限：3–4 句，≤ 85 词）

**真刊正面样例（`Feng2024_TEVC_MOFSEMT`, 65 词）**：
> In this section, we present the experimental study conducted to investigate the effectiveness and efficiency of the proposed MO-FSEMT framework in addressing the high-dimensional FS problems. Specifically, we first describe the experimental setup, including the real-world datasets, comparison methods, and the adopted parameter settings. Then, a detailed comparison and analysis of the experimental results are provided. Finally, the effectiveness of the strategies designed in the MO-FSEMT framework is validated through ablation experiments.

**真刊正面样例（`Li2023_TEVC_MFCSO`, 73 词）**：
> This section presents a large number of experiments to empirically validate the performance of our proposed MF-CSO in processing eighteen real-world high-dimensional datasets with the number of features ranging from 2000 to 12000. In addition, ablation experiments further demonstrate the effectiveness of our designed strategies in MF-CSO. All the simulations are performed by using MATLAB R2020a on a 64-bit Windows 10 personal computer with 24 GB RAM and an Intel Core-i7 3.6 GHz processor.

**初版反面教训（必须避免）**：
- 初版写了 180 词，把软件小版本号（NumPy 1.20 / SciPy 1.7 / single-threaded BLAS）、LLM 存档、所有小节参数范围全堆进导语，头重脚轻。

---

## 二、数据集表（Table I）排版参照

**真刊正面样例（`Feng2024_TEVC_MOFSEMT` Table I & `Li2023_TEVC_MFCSO` Table I）**：
- 单栏规整表格（仅占单栏约 35%–45% 高度），5–6 列数值对齐（`Dataset | #Features | #Instances | #Classes | Min/Max (%)`）。
- 数据集名称统一用**正体（Roman）**，不使用 `\texttt{}`。
- 辅助数据集或扩展套件（如我们的 Suite II 与 Suite III）不与主表硬拼成 13 列怪表，直接分流至 **Appendix / Supplementary Material**。

**初版反面教训（必须避免）**：
- 把 6 列高维表与 7 列临床表硬拼成一个 13 列超大跨栏表；右侧空缺处塞入项目符号文本块；使用 `\texttt{}`；出现断裂横线。

---

## 三、对比算法与参数设置参照

**真刊正面样例（`Li2023_TEVC_MFCSO` Table II & `Wang2024_TKDE_MEL` Appendix B）**：
- 正文用 **1–2 段流畅散文**按类别交代对比算法及其代表性（**不对算法名加粗**，不滥用项目符号与冒号清单）。
- 主文只交代所有算法共享的核心实验协议（分类器、评价预算、划分方式、公平预处理与统计检验）；**各对比算法的冗长细粒度超参表放入附录 Table S-II**（且单元格内仅写紧凑公式，不写多句散文）。

---

## 四、句式、标点与格式细节红线

| 检查项 | 正确 IEEE TEVC 写法 | 错误写法（严禁出现） |
|---|---|---|
| 正文算法名 | `SP-PD`, `MEL`, `BPSO`, `LLM-Lasso`（正体） | `\textbf{SP-PD}`, `\textbf{MEL}`（通篇加粗刷屏） |
| 数据集与模型名 | `arcene`, `DLBCL`, `Gemini-3.8-Flash`（正体） | `\texttt{arcene}`, `\texttt{gemini-3.8-flash}` |
| 破折号与插入语 | 用逗号或简洁从句连接，保持自然节奏 | 频繁使用 `--- ... ---` 长插入语 |
| 数学间距 | `$K = 8$`, `$B = 1{,}000$`, `$\alpha = 0.05$` | `$K{=}8$`, `$B{=}1{,}000$`（正文挤压） |
| 语气用词 | `evaluate`, `compare`, `demonstrate`, `Table I summarizes` | `rigorously`, `without cherry-picking`, `crucially`, `comprehensive empirical evaluation` |
