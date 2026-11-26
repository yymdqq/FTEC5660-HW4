# 任务5：执行、文件和评分说明

本包依据老师PDF中任务5的四项要求准备：完整消融表6分；每次只改一个组件并解释5分；错误分析2分；流程图和说明2分。

**已实际完成10组离线消融实验、逐题证据、流程图、英文报告及自动生成正式报告的脚本。**
**这里没有你的DeepSeek密钥，各变体的端到端准确率没有实测，因此预览报告明确标为草稿。**
你之前最终方案60/60正确、75/75分、7.5秒的日志，只证明最终方案，不能替代其他消融组的准确率。
本包没有将该日志或估计值填入其他组。

## 一、文件是什么

| 文件/目录 | 用途 |
|---|---|
| `task5_ablation.py` | 任务5完整实验和报告脚本；在Python 3.11下验证 |
| `README_Task5_Preview.md` | 已填写真实离线数据的英文报告预览；在线列尚未测量 |
| `public_results/` | 本次10组离线实验的真实CSV、元数据和逐题原文证据 |
| `Task5_Rubric_Check_CN.md` | 逐项对照15分评分标准，区分已完成和待测内容 |

## 二、你只需执行这些操作

将本包解压，确保此文件直接位于：

`C:\Users\于沈豪\Desktop\HW4_Task5\task5_ablation.py`

打开PowerShell，执行：

```powershell
Set-Location "C:\Users\于沈豪\Desktop\FTEC5660-HW4-main"

Copy-Item "C:\Users\于沈豪\Desktop\HW4_Task5\task5_ablation.py" ".\task5_ablation.py"

& ".\.venv\Scripts\python.exe" -u task5_ablation.py --project . --online --write-readme
```

继续使用你原有的Python 3.11虚拟环境、老师的原始依赖和本地`.env`，不用修改依赖或密钥。

脚本会顺序运行B0、V1、……、V9。每组内部仍使用老师原有的生成器及其8个工作线程。
按照当前数据和配置，预计有586次生成器请求，不含重试；完整在线运行会产生API费用。

### 正常完成后自动生成什么

- 根目录`README.md`中的`Homework 4 solution`部分：流程图、设计说明、实验配置、完整消融表、逐项分析、错误分析、复现方法；
- `task5_results/ablation_summary.csv`：10组结果总表；
- `task5_results/error_breakdown.csv`：各组检索错误、误拒答和生成错误计数；
- `task5_results/B0/results.csv`至`task5_results/V9/results.csv`：老师原有运行程序生成的真实逐题结果；
- 每组的`metrics.json`、`evidence.json`和`run_log.txt`；
- `task5_results/experiment_manifest.json`：控制变量、源码和数据指纹；
- `task5_results/run_status.json`：全部实验是否正常完成。

### 怎么确认任务5在线实验完成

全部实验成功后，`task5_results/run_status.json`中应出现：

```json
"online_complete": true
```

报告中的`Answer accuracy`和`Online s`会自动变成真实数值，不再显示`Not measured`。
如果API报错或实验超时，不会标为全部完成。可以执行：

```powershell
& ".\.venv\Scripts\python.exe" -u task5_ablation.py --project . --online --write-readme --reuse
```

`--reuse`会复用源码、数据、脚本指纹一致且已正常完成的实验，只重跑未完成或报API错误的组。
发生代码或数据改动时，旧结果不会被当作同一实验直接复用。

## 三、严格控制每次只改变一个组件

| 组别 | 相对上一组唯一改变的组件 |
|---|---|
| B0 | 老师规定的简单RAG基线 |
| V1 | HTML表格转换为保留行名、列名和数值的文本 |
| V2 | 分词器：财务词汇、数字、停用词及复数处理 |
| V3 | 查询解析和元数据候选过滤，作为一个自查询检索组件 |
| V4 | 分块组件：改为HTML段落和表格行 |
| V5 | 文本块内部的上下文头部 |
| V6 | 财务指标排序；不改变最大返回块数 |
| V7 | 比较题按公司轮流分配证据；候选得分和最大块数保持不变 |
| V8 | 证据数量上限：单家公司最多3块，两家公司各最多2块 |
| V9 | 证据校验和拒答，移除全局检索兜底 |

B0严格使用BeautifulSoup的`get_text()`、RecursiveCharacterTextSplitter、规定元数据、全语料BM25 top-5、小写空白分词，不解析查询且不拒答。

固定设置：BM25的k1=1.4、b=0.65；递归分块阶段chunk_size=1200、chunk_overlap=150；相同公开文档和问题；相同模型、提示词、评分函数和最多5块限制。
只有到V4才切换分块方式。

生成器原有的元数据头部一直保留。V5测试的是文本块内部额外的上下文头部，没有改动老师的生成器。
V9会断言全部60题检索结果与提交的`hw4.py`一致，确保最后一组是真实最终方案。

## 四、已测到的真实离线结果

| 组别 | Recall@5 | 可回答题保留率 | 无法回答题拒答率 |
|---|---:|---:|---:|
| B0 | 28.3% | 100% | 0% |
| V1 | 26.1% | 100% | 0% |
| V2 | 21.7% | 100% | 0% |
| V3 | 43.5% | 100% | 0% |
| V4 | 47.8% | 100% | 0% |
| V5 | 82.6% | 100% | 0% |
| V6 | 100% | 100% | 0% |
| V7 | 100% | 100% | 0% |
| V8 | 100% | 100% | 0% |
| V9 | 100% | 100% | 100% |

报告保留了表格转换和分词后的下降，也明确写出V7、V8没有额外召回率提升，不能为了好看将数据改成单调上升。
有正确证据不代表生成器一定答对，所以不能用Recall@5冒充答案准确率。

若想自己重新跑不调用API的离线实验：

```powershell
& ".\.venv\Scripts\python.exe" -u task5_ablation.py --project . --output task5_offline
```

这个命令不修改README，也不覆盖在线实验输出。

## 五、修改范围和30分钟限制

- `hw4.py`不修改，脚本会校验其完整文件指纹；
- 老师的模型、生成器、提示词、main和评分函数不修改；
- 实验仅在各自进程的内存中替换5个允许编写的学生函数，不改变磁盘上的提交实现；
- 不修改`.env`、`requirements.txt`或公开数据；
- 不覆盖根目录原有的`results.csv`，每组结果保存在自己的目录；
- README只新增或更新脚本标记的任务5报告部分，已有其他内容保留；若README原来已有另一份同名任务5草稿，可在提交前手动删除旧的重复草稿；
- 修改已有README前会在`task5_results/README_before_task5.md`保存原内容；
- 每组从进程启动起最多允许29分钟，包含导入、诊断和官方完整运行；超时会停止并标为未完成；
- 10组总时间可能超过30分钟，老师限制的是每次运行；
- 不能把被停止的实验当作满足30分钟要求的成功实验。

## 六、最后提交哪些内容

保留之前的`hw4.py`、`requirements.txt`和根目录`results.csv`，并将以下任务5内容加入仓库：

1. `task5_ablation.py`；
2. 自动更新的`README.md`；
3. `task5_results/`中的实验结果CSV、实验配置、日志和分析依据。

不提交`.env`、`.venv`、`__pycache__`或预建索引。任务6的反思仍单独完成。

当前报告预览包含未测列，因此不能直接说任务5已拿满分。在线实验成功完成后，报告才具备完整的端到端对比数据。
