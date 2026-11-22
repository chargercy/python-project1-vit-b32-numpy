# 用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器

**项目 1 提交** · 陈煜 · 2025681052 · 人工智能

本仓库只用 **NumPy** 实现 OpenCLIP ViT-B/32 的图像前向计算：输入 `[B,3,224,224]`
的预处理图像，输出每条图像一个 L2 单位长度的 `[B,512]` 向量。152 个视觉权重由课程
提供，全程不训练；完成的编码器随后作为期末项目 `szu_python_openclip` 的图像塔复用。

- 实验报告（封面 + 4 页正文）：[`REPORT/report.pdf`](REPORT/report.pdf)
- 检查输出：[`results/check_all.txt`](results/check_all.txt) · 误差数据：[`results/stage_metrics.csv`](results/stage_metrics.csv)
- 代码仓库：`https://github.com/<username>/python-project1-vit-b32-numpy`（占位链接，发布方法见文末）

## 结果一览

`python check.py --stage all` 全部通过（详见 `results/check_all.txt`）：

| 阶段 | 形状 | max &#124;Δ&#124; | mean &#124;Δ&#124; | atol | 结果 |
| --- | --- | --- | --- | --- | --- |
| preprocess | 10×3×224×224 | 0 | 0 | 2e-6 | PASS |
| patches | 2×49×3072 | 0 | 0 | 2e-4 | PASS |
| tokens | 2×50×768 | 4.768e-05 | 1.296e-06 | 2e-4 | PASS |
| block0 | 2×50×768 | 6.586e-06 | 3.252e-07 | 2e-4 | PASS |
| last_tokens | 2×50×768 | 4.864e-05 | 2.696e-06 | 2e-4 | PASS |
| output | 10×512 | 6.631e-07 | 8.289e-08 | 2e-5 | PASS |
| batch（乱序 + 重复 `[1,0,1]`） | 3×512 | 5.066e-07 | 7.768e-08 | 2e-5 | PASS |

`preprocess` 与 `patches` 的最大误差为 0：patchify 只做索引与轴重排，没有任何算术。
`output` 误差比容差小两个数量级；`batch` 用乱序且重复的索引重算，说明实现与批内位置
无关，也没有返回参考行。每条输出向量的平方和为 1（`check.py` 断言）。10 张图 × 12
个 block 的前向在本机 CPU 上约 9.5 s（含读取 326 MB 权重）。

## 快速复现

要求 Python ≥ 3.9，只需 NumPy 与 Pillow（`requirements.txt` 另含课堂用的 imageio）：

```bash
python -m pip install -r requirements.txt
python check.py --stage all
python run.py --output output_tokens.npz
```

`run.py` 会打印 `Saved (10, 512) to output_tokens.npz`，并与 `results/output_tokens.npz`
一致。**课程附件中的两个大文件未随仓库提交**：`weights/image_encoder.npz`（326 MB）与
`assets/reference_outputs.npz`（7.7 MB）。从作业附件把它们放回原位即可直接复现。

也可以逐阶段检查、观察第一处不一致：

```bash
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

## 我实现了什么

只修改了 `image_encoder.py`，六个 TODO 全部用 NumPy 的 reshape / transpose / 矩阵乘法 /
归约 / 广播完成：

| 任务 | 位置 | 做法 |
| --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | `[B,3,224,224]` → reshape `[B,3,7,32,7,32]` → transpose `(0,2,4,1,3,5)` → `[B,49,3072]`；patch 行优先、块内按 C,H,W |
| TODO-02 | `MultiHeadAttention.split_heads` | `[B,50,768]` → `[B,50,12,64]` → transpose `(0,2,1,3)` → `[B,12,50,64]` |
| TODO-03 | `MultiHeadAttention.__call__` | `q @ kᵀ / √64`，加权求和后 `transpose(0,2,1,3)` 合并，再乘 `out_proj` |
| TODO-04 | `TransformerBlock.__call__` | 两次残差；MLP = GELU(`u @ W1ᵀ + b1`) @ W2ᵀ + b2 |
| TODO-05 | `ImageEncoder.embed` | patch 投影无 bias；CLS 前置后加 `[50,768]` 位置编码 |
| TODO-06 | `ImageEncoder.__call__` | 取 CLS 行 → `ln_post` → 乘 `[768,512]` 的 `visual.proj` → 单位化 |

一个值得记录的调试：合并 head 时若直接 `context.reshape(B,50,768)`，block0 误差为
**1.907**；先 `transpose(0,2,1,3)` 再 reshape 后降到 **6.586e-06**。对比脚本
`experiments/head_merge_demo.py` 可复现这两个数字。

## 目录结构

```
image_encoder.py               # 唯一修改的代码文件（六个 TODO）
helpers.py / preprocess_image.py / runtime.py / run.py / check.py   # 课程提供
experiments/head_merge_demo.py # head 合并顺序的错误/正确对比
results/                       # check_all.txt、stage_metrics.csv、output_tokens.npz
REPORT/report.pdf              # 实验报告（封面 + 4 页）
REPORT/figures/                # 四张插图
REPORT/tools/                  # 生成图与 PDF 的脚本
docs/course-README.md          # 课程原始说明（保留备查）
```

## 发布到 GitHub

1. 在 GitHub 新建空仓库，名字例如 `python-project1-vit-b32-numpy`（不要勾选 README）。
2. 在本目录执行：

```bash
git init
git add .
git commit -m "Project 1: NumPy OpenCLIP ViT-B/32 image encoder"
git branch -M main
git remote add origin https://github.com/<username>/python-project1-vit-b32-numpy.git
git push -u origin main
```

3. 把 `<username>` 换成自己的用户名，并同步修改报告封面（`REPORT/report.pdf`）里的链接；
   重新生成报告：`python REPORT/tools/build_report_pdf.py`。

`.gitignore` 已排除 326 MB 的权重、参考输出与缓存，仓库体积保持在几 MB。
