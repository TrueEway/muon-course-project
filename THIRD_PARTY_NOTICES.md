# 第三方来源

`experiments/optim.py` 中 Newton–Schulz 五次多项式、EMA/Nesterov 约定和矩阵形状缩放参考并改写自 Keller Jordan 的 Muon 实现：

- 原项目：https://github.com/KellerJordan/Muon
- 固定参考版本：`64560829f70a216f8263ea8ae0fefeac8b451d36`
- 改动：采用 CPU float32、移除分布式代码、增加精确紧致极因子分支、避免原地修改梯度。

以下许可适用于上述参考实现及其改写部分：

```text
MIT License

Copyright (c) 2024 Keller Jordan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Tiny Shakespeare 文本来自 Andrej Karpathy 的 `char-rnn` 仓库，固定到首次提交中的数据文件。实际下载文件不纳入本仓库，来源、版本、SHA256 和划分信息见 `data/manifest.json`，运行时按原链接下载。语料为莎士比亚作品节选，本项目不声称拥有语料版权。

Transformer 架构、AdamW 及理论文献的出处见报告参考文献；本项目的实验证据均由随附程序生成。
