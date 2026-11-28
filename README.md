# Muon 研究大作业（v1.1）

修订版补入含谱范数近似误差的条件收敛定理、条件数 1e4 的非凸矩阵分解、完整 ResNet-18 的 CIFAR-10 比较，以及 K=0/1/3/5/7 的三种子训练消融。历史 v1.0 标签和原始实验保持不变。

## 阅读入口

- 提交版 PDF：[output/pdf/Muon_Report.pdf](output/pdf/Muon_Report.pdf)
- 主文件与分章节源码：[report/main.tex](report/main.tex)、[report/sections](report/sections)
- 四张新增矢量图：[output/figures](output/figures) 中的 fig1.pdf 至 fig4.pdf
- 新原始日志：[output/revision](output/revision)，旧日志：[output/results](output/results)
- 新旧审计：[output/revision_audit.json](output/revision_audit.json)、[output/audit.json](output/audit.json)
- 新版运行环境：[output/revision/environment.json](output/revision/environment.json)；旧 CPU 依赖：[requirements-lock.txt](requirements-lock.txt)
- 第三方来源：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

报告不包含姓名、学号，可在作者栏补充后由提交者本人上传 Canvas。

## 本次结果

H20 上三种子、3 epoch 的完整 ResNet-18 比较中，SGD+M、AdamW、Muon（K=5）的选定测试准确率分别为 **63.12±0.39%、69.21±0.11%、74.57±0.27%**；单步活跃时间约 **9.3、9.9、26.8 ms**。三者都在首次 epoch 检查达到预设 40% 验证阈值，达标步数比均为 1。该实验展示短程准确率与计算开销的权衡，不构成训练加速或 K=5 普遍最优的证明。

## 科学表述的修正

1. 谱范数单位球上的最小化方向是 -UV^T。报告定义 O*=UV^T，沿 -O* 更新。
2. 显式含 delta 的非渐近界采用谱范数误差。统一 delta<1 已是本证明路线的充分条件；固定五步 NS 不自动保证全程误差界。
3. 非凸因子分解只有局部光滑性，不能不加条件地验证全局光滑定理；原有受控二次实验保留在附录。
4. 更新谱更均衡不能直接证明表示特征坍缩消除，K=5 也不预设为最佳。
5. 阈值只由验证集定义，未达标不伪造加速比。分别记录进程峰值 RSS、PyTorch 峰值已分配显存与保留显存。

## 新增冻结协议

完整 ResNet-18 使用 [2,2,2,2] 基本块、[64,128,256,512] 通道，共 **11,181,642** 参数。保留原始 7x7/stride-2 stem 和 max-pool，原生 32x32 输入，只将 head 改为 10 类；没有缩窄通道或删层。这与常见 3x3 stem 的 CIFAR 变体不同，不与另一结构的公开成绩直接比较。

官方训练数据按类别划分为 45,000 训练、5,000 验证，官方测试集 10,000。划分种子 20261128，来源、官方 MD5、SHA256 及索引摘要见 data/cifar10/manifest.json。二进制包校验后解析，不使用 pickle；约 170 MB 下载包不上传 GitHub。

所有正式训练使用 batch=128、3 epoch/1056 次更新、50 步预热和余弦衰减；这是完整数据上的短程 GPU 比较。增强为补零 4、随机裁剪和水平翻转。相同种子共享初始化、批次和增强随机序列。

SGD、AdamW、Muon 分别独立试跑三个 LR，各 96 步，种子 42，按验证交叉熵选择。候选依次为 0.03/0.1/0.3、0.0003/0.001/0.003、0.005/0.02/0.05。辅助 AdamW 共享已选基线 LR。

正式种子 0/1/2。三个主方法 9 次训练，五个 K 各三个种子共 15 次，其中 K=5 与主比较共享，因此共 **21 次正式训练**。K 消融只改变迭代次数，其余配置固定为 K=5 所选配置。K=0 表示仅做 Frobenius 归一化，不是零更新，也不是普通动量 SGD。

每 epoch 验证并保存检查点，按验证 CE 选择；全部训练后才离线评价测试曲线，不用测试选择 LR。验证准确率阈值固定 40%，每 epoch 检查，步数分辨率 352。只有同种子双方均达标时计算相对 SGD 的步数比；配对不完整时明确标记。

采用单张 H20-3e、CUDA float32、channels-last、4 个绑定 CPU 线程，TF32 关闭。服务器为双路 Xeon 6759P-C，共 240 个逻辑核；本任务只用其中四个。复用服务器已有 PyTorch 2.8.0a0/CUDA 12.9，没有重装训练框架。数据、缓存、检查点都位于项目目录，不使用服务器根分区存放大文件。迁移前未完成的本机 CPU 记录单独归档，不混入 H20 统计。

每次训练使用新子进程且串行执行。PyTorch 分配器上限为单卡显存的 4%，每批训练和评价后主动等待，计算占空目标为 35%；这不是硬件算力硬上限。定期检查全卡剩余显存，低于 12 GiB 时退出自身任务。CPU/I/O 优先级降低；不更改 GPU 全局设置。

单步活跃时间在 CUDA 同步后测量，含批次预处理/传输与优化器更新，不含等待、验证与谱诊断。训练墙钟包含验证、谱诊断、主动等待及训练期检查点写入，排除进程启动、下载解压和训练后测试。墙钟反映共享负载及等待策略，不能当作独占 H20 吞吐量。峰值 RSS 包含数据、模型、优化器和临时空间；CUDA 已分配量不含驱动上下文，缓存保留量另报。详细配置见 output/revision/protocol.json，短测试见 output/revision/h20_smoke_test.json。

非凸分解优化 0.5*||UV^T-A||F^2，U/V/A 均为 32x32，A 的真实条件数为 1e4。三个方法各试跑三个 LR/300 步，再按三个种子各训练 3000 步，共 9 次正式运行。所有衰减为 0，报告真实梯度核范数及重构误差，不预设 Muon 胜出。

## Linux / H20 复现

需要 Python 3.12、兼容驱动的 CUDA PyTorch、NumPy、Matplotlib，以及编译报告用的中文 TeX Live（XeLaTeX/BibTeX）和 Poppler。项目自行实现完整 ResNet-18，无需 torchvision。先检查当前负载，再选择一个可用 GPU 和四个逻辑核；下面的 GPU/CPU 编号仅为例子，其他服务器需要自行调整。请勿用历史 requirements-lock.txt 的 CPU PyTorch 覆盖服务器环境。

```bash
export MUON_GPU=2 MUON_CPUSET=119,95,25,72
bash scripts/run_h20.sh scripts/profile_h20.py
bash scripts/run_h20.sh -m unittest discover -s tests -v
bash scripts/run_h20.sh -m experiments.factorization
bash scripts/run_h20.sh train.py
bash scripts/run_h20.sh plot.py
bash scripts/build_report.sh
python scripts/verify_artifacts.py  # 需要 pypdf
python scripts/render_review.py    # 需要 Pillow 与 pypdf
```

若服务器仅缺 PDF 校验依赖，可使用 `python -m pip install --no-deps --target .cache/audit pypdf`，再设置 `PYTHONPATH="$PWD/.cache/audit"`；该目录不进入仓库。`scripts/finalize_h20.sh` 自动识别这一本地目录，依次捕获环境、绘图、编译、校验和渲染。`scripts/reproduce_h20.sh` 提供完整顺序入口；原始记录已齐全时可直接运行 finalize。

muon.py 提供导入，train.py 运行冻结协议，plot.py 独立读取完整日志。绘图器要求 21 次图像正式运行和 9 次分解运行齐全，拒绝将部分协议生成终稿。已有同配置运行跳过，配置变更拒绝静默混用。重跑请用独立副本，保留已发布版本。

Linux 构建按 XeLaTeX → BibTeX → XeLaTeX → XeLaTeX 执行。Windows 的 scripts/build_report.ps1 仍可仅重新编译已有图表，在临时英文路径完成依赖链，规避中文绝对输出路径限制。Muon_Report.pdf 是提交名称，main.pdf 为内容相同的兼容名称。Windows 的 reproduce.ps1 仅汇总已有新版记录；新增训练请使用 Linux CUDA 入口。

## 记录与图表

output/revision/vision/pilot 保存所有候选，formal 保存逐步 CSV、逐 epoch 指标、配置、资源及更新谱 NPZ。验证集最优检查点保留在本地 output/revision/checkpoints，不上传 GitHub。

四张主图对应病态分解、CIFAR 动态、真实更新谱和 K 权衡。代表层固定为 layer1.0.conv2.weight，展平 64x576，记录第 1 步及 epoch 末，不按结果挑层。有效条件数阈值 max(1e-6*s1,1e-12)。误差为样本标准差，不是总体置信区间。

## v1.0 历史复现

旧版 144 次确定性二次实验、48 次随机矩阵实验和 9 次正式语言模型记录在 output/results，双线程 CPU 时间只用于对应附录，不与新增 GPU 时间混合。requirements-lock.txt 与 setup.ps1 仅属于这个历史 CPU 环境。

```powershell
.\.venv\Scripts\python.exe -m experiments.run --suite all --device cpu --threads 2
.\.venv\Scripts\python.exe -m experiments.summarize
```

新旧审计摘要分开保存。计时受负载影响，跨机器重跑的集合摘要通常不同。PDF 经逐页渲染复核后发布；生成物、原始记录、LaTeX 和完整运行说明均随版本交付。

environment.json 中的源码摘要是训练启动快照；核验器检查列明的七个训练相关文件，绘图与排版代码的后续修订由发布标签固定。原始结果的集合摘要覆盖 CSV、JSON 和 NPZ，包含环境及短测试记录。

## 发布和打包

新版本使用 v1.1，不替换旧标签。scripts/publish_github.py --publish --tag v1.1 使用已登录的 Git Credential Manager，不输出或写出令牌。scripts/package_project.py --tag v1.1 打包跟踪文件，排除环境、语料与本地检查点。
