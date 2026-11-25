# 专业学术术语与符号总表（`terminology.md`）

> 遵循 [writing_essay.md](file:///Users/ssunxie/essay/second/tevc-experiments-20261124/writing_guides/writing_essay.md) 规则 1 与规则 9：**随章节渐进累积、全篇唯一锁定**。每个概念与数学符号在整个手稿（方法章 Section III、实验章 Section IV 及附录）中只允许使用表中唯一固定表达，严禁同义异形替换、自造包装词或工程黑话。

## 一、全篇核心数学符号锁定（Section III 与 Section IV 共享）

| 符号 | 中文定义 | 英文固定定义 | 使用约束 |
|---|---|---|---|
| $p$ | 特征维数 / 特征数 | number of features | Table I 表头用 `#Features`，公式与正文统一用 $p$ |
| $N$ | 样本总数 / 实例数 | number of instances | Table I 表头用 `#Instances`，每类训练样本数用 $n$ |
| $C$ | 类别数 | number of classes | Table I 表头用 `#Classes` |
| $B$ | 评价预算 | evaluation budget | 指外层验证前允许的内层候选子集评价总次数 |
| $K$ | 最大特征子集规模 | maximum subset size / subset size limit | 直接写具体取值（$K = \lfloor 0.15p \rfloor$、$K = 8$、$K = 4$），禁用 `Relaxed/Compact Limit` |
| $R, \lambda$ | 重复配对次数与先验正则系数 | repeated pairings / prior-regularization weight | 临床配对交付默认取 $R = 10, \lambda = 0.5$（高维无先验时 $\phi \equiv 0$） |
| $n$ | 每类训练样本数 | training instances per class | 临床小样本设定 $n \in \{3, 10, 20, 30\}$ |

---

## 二、通用与 Section IV-A（实验设置）核心术语

| 中文固定译名 | 英文固定术语 | 严禁混用的同义变体 / 自造词 / 黑话 | 对标来源 |
|---|---|---|---|
| 本文方法 | SP-PD | SP-PD-Num, 无先验模式, prior-free mode（全篇统一单一框架名 SP-PD） | 本文统一命名 |
| 离线语义先验 | offline semantic prior | LLM brief, semantic hint, 工单 | 本文方法（固定） |
| 进化搜索与配对交付 | evolutionary search and paired delivery | dual-stage oracle, paired handoff | 本文方法（固定） |
| 特征选择 | feature selection (FS) | attribute selection, variable screening | Li2023 (TEVC), Feng2024 (TEVC) |
| 高维基准数据集 | high-dimensional benchmarks / benchmark datasets | high-dim track, core benchmarks, numerical track | Li2023 (TEVC), Wang2024 (TKDE) |
| 真实临床数据集 | real-world clinical datasets | clinical track, medical tables, clinical waves, zero-history cohorts | 本文应用验证（固定） |
| 特征（数） | features ($p$) | attributes, variables, predictors, covariates, columns | Feng2024 (TEVC) |
| 样本（数） | instances ($N$) | samples, records, donors, subjects | Li2023 (TEVC), Feng2024 (TEVC) |
| 类别（数） | classes ($C$) | categories, labels | Li2023 (TEVC), Wang2024 (TKDE) |
| 对比算法 / 基线 | comparison algorithms / baselines | competitors, counterparts, rival methods | Li2023 (TEVC), Wang2024 (TKDE) |
| 进化特征选择算法 | evolutionary feature selection algorithms | evolutionary and swarm wrappers, EC wrappers | Li2023 (TEVC), Xue2016 (TEVC) |
| 过滤式与嵌入式方法 | filter and embedded methods | classical filter selectors, regularized selectors | Xue2016 (TEVC), Li2023 (TEVC) |
| 基于大语言模型的特征选择方法 | LLM-based feature selection methods | LLM-informed selectors, semantic methods | Jeong2024, Zhang2025 |
| 交叉验证 | cross-validation (CV) | inner fold check, CV oracle | Li2023 (TEVC), Feng2024 (TEVC) |
| 最近质心分类器 | nearest-centroid (NC) classifier | Primary readout, primary evaluator | 本文临床主分类器（固定） |
| 一致先验 / 偏差先验 | aligned prior / misaligned prior | correct/wrong brief, 工单 | 本文先验设定（固定） |
| 宏平均精确率 | macro-precision（表题用 Macro-Precision） | macro prec, precision（正文不单写 precision 替代） | 本文指标（固定） |
| 平衡准确率 | balanced accuracy (BA)（表题用 Balanced Accuracy） | balanced error, bAcc | Feng2024 (TEVC), Xue2016 (TEVC) |
| 宏平均 F1 分数 | macro-F1（表题用 Macro-F1） | F1-score, F-measure | 本文指标（固定） |
| Friedman 平均秩 | Friedman average rank（窄列头用 `Rank`） | mean rank, rank score | Demšar2006 (JMLR) |
| Wilcoxon 符号秩检验 | Wilcoxon signed-rank test | exact paired sign-flip test | Li2023 (TEVC), Feng2024 (TEVC) |

---

## 三、Section IV-B（高维基准对比）新增术语

| 中文固定译名 | 英文固定术语 | 严禁混用的同义变体 / 自造词 / 黑话 | 对标来源 |
|---|---|---|---|
| 特征子集规模 / 选入特征数 | subset size / number of selected features（表行统一 `Mean #Feat`） | subset width, active features, `#F_sel`, `Avg. #Feat` 混用 | Li2023 (TEVC), Feng2024 (TEVC), Wang2024 (TKDE) |
| 均值行 / 精确 $p$ 值 | `Mean` / `$p$-value`（表内统一） | `Mean Prec. (%)`, `Avg. Rank` 混用，或借用 `+/=/-`、`†` | Li2023 (TEVC), Wang2024 (TKDE) |
| 收敛曲线 | convergence curves | anytime curves, budget trajectories | Wang2024 (TKDE), Li2023 (TEVC) |
| 百分点差值 | percentage points | points, -point lead | IEEE 规范表达 |
| 内层验证过拟合 | inner-validation overfitting | CV overfitting, wrapper memorization | 标准机器学习表达 |

## 四、Section IV-C（真实临床数据集对比）新增术语

| 中文固定译名 | 英文固定术语 | 严禁混用的同义变体 / 自造词 / 黑话 | 对标来源 |
|---|---|---|---|
| 跨分类器泛化 | cross-classifier generalization | transferability to downstream models | Jeong2024, Xue2016 (TEVC) |
| 零系数退化 | zero-coefficient degeneration | L1 collapse, empty model failure | Zhang2025, 本文临床诊断 |
| 零样本提示方法 | zero-shot prompting methods (`LLM-Select (Rank/Score)`) | static LLM oracle, prompt-only selector | Jeong2024 |
| 先验偏差恢复 / 纠偏 | recovery under misaligned priors | adversarial rescue, wrong-brief fix | Zhang2025 |

## 五、Section IV-D（消融、敏感性与效率分析）新增术语

| 中文固定译名 | 英文固定术语 | 严禁混用的同义变体 / 自造词 / 黑话 | 对标来源 |
|---|---|---|---|
| 消融实验 | ablation study | lesion study, strip-down test | Li2023 (TEVC), Feng2024 (TEVC) |
| 参数敏感性分析 | parameter sensitivity analysis | hyperparameter tuning sweep | Li2023 (TEVC) |
| 重复配对交付 | repeated-pairing prior-regularized delivery | EB-RPD, multi-pair vote | 本文方法（固定） |
| 错例引导搜索 | error-guided search | mistake-driven heuristic | 本文方法（固定） |
| 树并集特征提议 | tree-union proposal | tree-union menu | 本文方法（固定） |
| 跨提示标准差 | across-prompt standard deviation (SD) | prompt jitter, LLM noise | 本文稳定性指标（固定） |
