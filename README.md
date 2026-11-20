# Muon：矩阵更新机制、随机非凸收敛分析与仿真

优化与博弈课程大作业。报告研究精确极因子 Muon 的确定性与随机收敛，并用 CPU 实验比较有限步正交化和微型语言模型训练。

## 阅读入口

- 报告 PDF：[`output/pdf/main.pdf`](output/pdf/main.pdf)
- LaTeX 源码：[`report/main.tex`](report/main.tex)
- 原始实验记录：[`output/results`](output/results)
- 汇总与审计：[`output/audit.json`](output/audit.json)
- 依赖版本：[`requirements-lock.txt`](requirements-lock.txt)
- 第三方来源及许可：[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

代码版本使用 `v1.0` 标签。报告不包含姓名、学号；提交者可以在本地 LaTeX 源码的作者栏补充。Canvas 提交由提交者本人完成。

## 实验与理论的对应

1. **确定性矩阵问题**：`32×16` 参数，Hessian 条件数为 1、10、100，比较梯度下降、EMA 动量和精确 Muon，运行四个预算。
2. **随机矩阵问题**：在精确梯度上叠加已知方差的高斯噪声，检验动量跟踪误差和显式有限步界。
3. **正交化测量**：比较三种形状、三种奇异值谱和四个 Newton–Schulz 深度。
4. **微型语言模型**：两层、宽度 64 的因果 Transformer，比较全参数 AdamW、Muon-SVD+AdamW 与 Muon-NS+AdamW。

正式实验种子为 0、1、2，试跑选学习率使用种子 42。报告中的误差为样本标准差，不代表总体置信区间。理论只直接覆盖无衰减、无 Nesterov 的精确版本；实际语言模型是带衰减和辅助 AdamW 的混合配置，不声称其满足该精确版本定理。

## Windows 一键复现

需要 Windows、64 位 Python 3.12，以及联网下载依赖和数据。报告编译需要包含中文宏包的 TeX Live（XeLaTeX、latexmk、BibTeX）；PDF 校验和渲染另需 Poppler 的 `pdftotext`、`pdftoppm` 加入 PATH。在项目目录运行：

```powershell
.\scripts\setup.ps1
.\scripts\reproduce.ps1
```

安装脚本优先使用已安装 Python，也兼容当前 Codex 桌面环境中的 Python 运行时。依赖安装使用 PyPI 和 PyTorch 官方 CPU wheel 源；不会使用 GPU 或云服务。

构建脚本会将 LaTeX 源码和图表复制到临时英文路径编译，再把 PDF 与日志复制回来，以兼容本机 TeX Live 对中文绝对输出路径的限制；构建结束后清理本次临时目录。

已有同配置记录会跳过。若要全部重做并覆盖生成结果：

```powershell
.\scripts\reproduce.ps1 -Overwrite
```

## 单独运行

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m experiments.run --suite all --device cpu --threads 2
.\.venv\Scripts\python.exe -m experiments.summarize
.\scripts\build_report.ps1
```

构建后可运行 `.\.venv\Scripts\python.exe scripts/verify_artifacts.py` 核对 201 份正式运行日志、原始记录摘要和 PDF 引用。运行 `.\.venv\Scripts\python.exe scripts/render_review.py` 使用 Poppler 逐页渲染 PDF，在 `tmp/pdfs` 中生成页面图和检查拼图；视觉检查仍需阅读渲染结果。

`--suite` 可以选择 `deterministic`、`stochastic`、`orthogonalization`、`synthetic`、`language` 或 `profile`。默认正式种子为 `--seeds 0 1 2`；默认输出为 `output/results`，可用 `--output` 指定独立目录。同一文件名对应不同配置时程序会拒绝静默混用。

汇总器针对报告的完整协议检查运行数：144 次确定性运行、48 次随机运行和 9 次正式语言模型训练。如果只运行部分实验，不应把其输出当作完整报告结果。微基准记录在 `output/results/language/profile.csv`，它不属于正式学习率选择。

## 数据与计时

Tiny Shakespeare 固定到 `karpathy/char-rnn` 的提交 `370cbcd448eb7daf32f21a6be560b70e0b33c4e3`。首次运行下载约 1.1 MB 文本，随后校验 `data/manifest.json` 中的 SHA256。训练、验证、测试按原文顺序划为 90%/5%/5%，窗口不跨集合。词表仅使用公开语料的字符集合。

每种语言模型配置先试跑三个学习率，各 300 步；正式训练 1500 步。按固定验证批量选择检查点，完整测试集仅评价所选检查点。相同正式种子共享初始化和训练批次，方法执行顺序随种子轮换。

训练计算耗时包含取批量、前向、反向和参数更新，不包含验证、测试、下载和检查点写入。测试指标从各次验证最优检查点取得，表中时间对应完整的 1500 步预算。运行时长受机器和负载影响；CPU 时间不能外推为 GPU 大模型加速比。

本地检查点位于 `output/results/checkpoints`，不上传 GitHub；重新训练可生成。全部逐步 CSV、配置 JSON 和测试汇总保留在仓库中。图表完全由这些记录生成，不依赖人工录入成绩或指标。

## 文件校验与重跑差异

`output/audit.json` 保存原始记录集合的 SHA256 及主要检查量。修改结果 CSV/JSON 后重新汇总会改变校验值。由于计时随机器负载变化，即使训练数值相同，重跑后的记录集合摘要也可能不同；摘要用来标识本次交付，不能当作跨机器位级复现的要求。

## GitHub 发布

目标仓库为 `https://github.com/TrueEway/muon-course-project`。`scripts/publish_github.py` 默认只读取本地 Git Credential Manager 的登录状态；只有显式传入 `--publish` 才会创建公开仓库并推送已准备的 `main` 和 `v1.0`。脚本不会打印或写入令牌，也不会替换已有的不同远程地址。
