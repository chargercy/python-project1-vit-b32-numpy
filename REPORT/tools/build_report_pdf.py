"""Build the illustrated four-page lab report (cover + 4 pages) as a PDF.

The layout is built with ReportLab platypus; Chinese text uses Microsoft YaHei,
code uses Consolas. Red text marks the parts implemented in image_encoder.py.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle,
                                XPreformatted)
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "report.pdf"

# --- fonts -----------------------------------------------------------------
FONTS = Path(r"C:\Windows\Fonts")
pdfmetrics.registerFont(TTFont("Sans", FONTS / "msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("Sans-Bold", FONTS / "msyhbd.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("Mono", FONTS / "consola.ttf"))
pdfmetrics.registerFont(TTFont("Mono-Bold", FONTS / "consolab.ttf"))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold",
                              italic="Sans", boldItalic="Sans-Bold")

NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2F6FB3")
RED = colors.HexColor("#C00000")
TEAL = colors.HexColor("#21867A")
GRAY = colors.HexColor("#5B6B7C")
LINE = colors.HexColor("#C3D0DE")
PALE = colors.HexColor("#F5F8FC")

PAGE_W, PAGE_H = A4
LM = RM = 1.95 * cm
TM = 2.0 * cm
BM = 1.85 * cm
FRAME_W = PAGE_W - LM - RM
FRAME_H = PAGE_H - TM - BM

STUDENT = {"name": "陈煜", "sid": "2025681052", "class": "人工智能",
           "date": "2026 年 11 月 22 日",
           "github": "https://github.com/<username>/python-project1-vit-b32-numpy"}

# --- styles ----------------------------------------------------------------
body = ParagraphStyle("body", fontName="Sans", fontSize=9.5, leading=14.7,
                      alignment=TA_JUSTIFY, wordWrap="CJK", spaceAfter=3.2,
                      textColor=colors.HexColor("#1F2A37"))
lead = ParagraphStyle("lead", parent=body, fontSize=9.7, leading=15.0)
h1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=13.2, leading=17,
                    textColor=NAVY, spaceBefore=8, spaceAfter=1, wordWrap="CJK")
h2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=10.6, leading=14,
                    textColor=NAVY, spaceBefore=5, spaceAfter=2, wordWrap="CJK")
caption = ParagraphStyle("caption", fontName="Sans", fontSize=8.2, leading=11,
                         textColor=GRAY, alignment=TA_CENTER, spaceBefore=2,
                         spaceAfter=7, wordWrap="CJK")
code = ParagraphStyle("code", fontName="Mono", fontSize=7.6, leading=10.2,
                      textColor=colors.HexColor("#1F2A37"))
code_red = ParagraphStyle("code_red", parent=code, textColor=RED)
note = ParagraphStyle("note", fontName="Sans", fontSize=8.6, leading=12.4,
                      textColor=GRAY, wordWrap="CJK", spaceBefore=2, spaceAfter=6)
cell = ParagraphStyle("cell", fontName="Sans", fontSize=8.4, leading=11.4,
                      wordWrap="CJK")
cell_c = ParagraphStyle("cell_c", parent=cell, alignment=TA_CENTER)
head_cell = ParagraphStyle("head_cell", fontName="Sans-Bold", fontSize=8.4,
                           leading=11.4, textColor=colors.white, alignment=TA_CENTER)
th = ParagraphStyle("th", parent=head_cell, alignment=TA_LEFT)


def P(text, style=body):
    return Paragraph(text, style)


def code_block(lines, red=None, width=FRAME_W):
    """A light code panel; lines listed in `red` are the implemented ones."""
    red = set(red or [])
    rows = []
    for i, line in enumerate(lines):
        style = code_red if i in red else code
        rows.append([XPreformatted(line.replace(" ", "&nbsp;"), style)])
    table = Table(rows, colWidths=[width])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F9FC")),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 0.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6),
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 5),
    ]))
    return KeepTogether(table)


def figure(path, width_cm, cap):
    from PIL import Image as PILImage
    with PILImage.open(path) as probe:
        w, h = probe.size
    height = width_cm * cm * h / w
    return [Image(str(path), width=width_cm * cm, height=height),
            P(cap, caption)]


def rule(color=NAVY, thickness=1.1, space=2):
    table = Table([[""]], colWidths=[FRAME_W], rowHeights=[0.1])
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, -1), thickness, color),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return [table, Spacer(1, space)]


# --- page furniture --------------------------------------------------------
def cover_page(cv, doc):
    cv.saveState()
    cv.setFillColor(NAVY)
    cv.rect(0, PAGE_H - 26 * mm, PAGE_W, 26 * mm, stroke=0, fill=1)
    cv.setFillColor(colors.white)
    cv.setFont("Mono-Bold", 9)
    cv.drawString(LM, PAGE_H - 11 * mm, "PYTHON PROGRAMMING  ·  PROJECT 1")
    cv.drawRightString(PAGE_W - RM, PAGE_H - 11 * mm, "IMAGE ENCODER")

    cv.setFillColor(colors.HexColor("#EAF1F9"))
    cv.rect(LM, 120 * mm, 3.2 * mm, 132 * mm, stroke=0, fill=1)

    cv.setFillColor(NAVY)
    cv.setFont("Sans-Bold", 25)
    cv.drawString(LM + 11 * mm, 237 * mm, "用 NumPy 复现")
    cv.setFont("Sans-Bold", 21)
    cv.drawString(LM + 11 * mm, 223 * mm, "OpenCLIP ViT-B/32 图像编码器")
    cv.setFillColor(RED)
    cv.setFont("Sans-Bold", 14)
    cv.drawString(LM + 11 * mm, 208 * mm, "实 验 报 告")
    cv.setFillColor(GRAY)
    cv.setFont("Sans", 9)
    cv.drawString(LM + 11 * mm, 196 * mm,
                  "Project 1 · NumPy-only forward pass · 152 visual tensors · output [B, 512]")

    cv.setStrokeColor(LINE)
    cv.setLineWidth(0.6)
    cv.line(LM + 11 * mm, 180 * mm, PAGE_W - RM, 180 * mm)

    def field(x, y, label, value):
        cv.setFillColor(colors.HexColor("#F3F6FA"))
        cv.roundRect(x, y - 10 * mm, 74 * mm, 13 * mm, 2 * mm, stroke=0, fill=1)
        cv.setFillColor(GRAY)
        cv.setFont("Sans", 8.2)
        cv.drawString(x + 4 * mm, y - 6.2 * mm, label)
        cv.setFillColor(colors.HexColor("#1F2A37"))
        cv.setFont("Sans", 10.5)
        cv.drawString(x + 21 * mm, y - 6.4 * mm, value)

    field(LM + 11 * mm, 162 * mm, "姓　　名", STUDENT["name"])
    field(LM + 99 * mm, 162 * mm, "学　　号", STUDENT["sid"])
    field(LM + 11 * mm, 144 * mm, "班　　级", STUDENT["class"])
    field(LM + 99 * mm, 144 * mm, "提交日期", "2026-11-22")

    cv.setFillColor(GRAY)
    cv.setFont("Sans", 8.6)
    cv.drawString(LM + 11 * mm, 126 * mm, "代码仓库")
    cv.setFillColor(BLUE)
    cv.setFont("Mono", 8.6)
    cv.drawString(LM + 28 * mm, 126 * mm, STUDENT["github"])

    cv.setStrokeColor(LINE)
    cv.line(LM, 64 * mm, PAGE_W - RM, 64 * mm)
    cv.setFillColor(colors.HexColor("#8A98A8"))
    cv.setFont("Sans", 7.8)
    cv.drawString(LM, 55 * mm,
                  "封面不计入正文页数；正文 4 页。红色文字表示本人在 TODO 处补充的代码与结论。")
    cv.restoreState()


def body_page(cv, doc):
    cv.saveState()
    cv.setStrokeColor(LINE)
    cv.setLineWidth(0.5)
    cv.line(LM, PAGE_H - 14 * mm, PAGE_W - RM, PAGE_H - 14 * mm)
    cv.setFont("Sans", 7.6)
    cv.setFillColor(GRAY)
    cv.drawString(LM, PAGE_H - 12 * mm, "Python Programming · 项目 1")
    cv.drawRightString(PAGE_W - RM, PAGE_H - 12 * mm,
                       "NumPy 复现 OpenCLIP ViT-B/32 图像编码器")
    cv.setStrokeColor(LINE)
    cv.line(LM, BM - 4 * mm, PAGE_W - RM, BM - 4 * mm)
    cv.restoreState()


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._states = []

    def showPage(self):
        self._states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._states)
        for state in self._states:
            self.__dict__.update(state)
            if self._pageNumber > 1:
                self.setFont("Sans", 7.6)
                self.setFillColor(GRAY)
                self.drawCentredString(PAGE_W / 2, BM - 9 * mm,
                                       "第 %d 页 / 共 %d 页" % (self._pageNumber - 1, total - 1))
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)


# --- content ---------------------------------------------------------------
def cover_story():
    return [Spacer(1, 1)]


def content_story():
    s = []
    # 1 -------------------------------------------------------------------
    s.append(P("1　项目目标", h1))
    s.append(P(
        "本实验只使用 <b>NumPy</b> 复现 OpenCLIP ViT-B/32 图像塔的前向计算。输入是经课程 "
        "<font face='Mono' size='9'>preprocess_image.py</font> 处理后的 <font face='Mono'>[B,3,224,224]</font> "
        "图像批，输出是每张图像一个 L2 单位长度的 <font face='Mono'>[B,512]</font> 向量；152 个视觉权重由课程提供且全程固定，"
        "不涉及任何训练。完成的编码器随后作为期末项目 <font face='Mono'>szu_python_openclip</font> 的图像塔直接复用，"
        "因此接口（<font face='Mono'>ImageEncoder(weights)</font>、<font face='Mono'>LayerNorm</font>、"
        "<font face='Mono'>TransformerBlock</font>）保持不变。", lead))

    # 2 -------------------------------------------------------------------
    s.append(P("2　整体方案", h1))
    s.append(P(
        "图像被切成 7×7 个 32×32 的 patch，每个 patch 按 C,H,W 展平成 3072 维，经<b>无偏置</b>线性投影得到 49 个 patch token；"
        "与可学习 CLS 拼接并加上位置编码后得到 <font face='Mono'>[B,50,768]</font>，经 <font face='Mono'>ln_pre</font> 送入 12 个结构相同的 "
        "Transformer block。每个 block 先做 LN₁ + 12 头自注意力并残差相加，再做 LN₂ + 768→3072→768 的 MLP 并再次残差相加。"
        "最后取 CLS 行，经 <font face='Mono'>ln_post</font>、768→512 投影与逐行 L2 归一化得到输出。图 1 给出完整数据流与 block 内部结构。"))
    s += figure(FIG / "fig1_architecture.png", 16.4,
                "图 1　完整前向数据流（上）与每个 Transformer block 的内部结构（下）。图中红色加号表示两次残差相加，"
                "它们读取的都是未归一化的张量。")

    # 2.1 -----------------------------------------------------------------
    s.append(P("2.1　模型约定与实现接口", h2))
    s.append(P("课程提供的 checkpoint 有若干与教科书不同的约定，直接决定矩阵乘法的方向，见表 1。"))
    header = [P("项目", th), P("本实验遵循的约定", th)]
    rows = [
        ("线性层权重", "<font face='Mono'>[out,in]</font>（PyTorch nn.Linear），前向使用 <font face='Mono'>x @ W.T + b</font>"),
        ("patch 投影", "<font face='Mono'>visual.conv1.weight</font> 视为 <font face='Mono'>[768,3072]</font>，<b>没有 bias</b>"),
        ("输出投影", "<font face='Mono'>visual.proj</font> 已是 <font face='Mono'>[in,out]=[768,512]</font>，<b>不再转置</b>"),
        ("归一化 / 激活", "LayerNorm ε=1e-5；GELU 用课程提供的精确 erf 形式"),
        ("残差与缩放", "每个 block 两次残差，均加在未归一化输入上；<b>没有 LayerScale</b>"),
    ]
    data = [header] + [[P(a, cell), P(b, cell)] for a, b in rows]
    table = Table(data, colWidths=[3.0 * cm, FRAME_W - 3.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    s.append(table)
    s.append(P("表 1　checkpoint 的关键约定。红色文字在正文中表示本人在 TODO 处补充的实现。", caption))

    # 3 -------------------------------------------------------------------
    s.append(P("3　实现细节（红色为本人在 TODO 处补充的代码）", h1))
    s.append(P("3.1　patchify：先分轴、再换序、最后展平", h2))
    s.append(P(
        "224/32 = 7，因此第一步用一次 <font face='Mono'>reshape</font> 把两个空间轴拆成“网格轴 + 块内轴”，得到 "
        "<font face='Mono'>[B,3,7,32,7,32]</font>；第二步 <font face='Mono'>transpose(0,2,4,1,3,5)</font> 把轴顺序调成 "
        "<font face='Mono'>[B,7,7,3,32,32]</font>，使 patch 以“先左右、后上下”的行优先顺序排列；第三步 flatten 成 "
        "<font face='Mono'>[B,49,3072]</font>，块内保持 C,H,W。patch 边长由权重列数 3072=3·32² 反推，而不是硬编码。"
        "图 2 用第一张图像展示了 7×7 网格与行 0 的对应关系。"))
    s.append(P(
        "这一步只做索引与轴重排，不改变任何像素值，也不重新缩放图像（预处理已由课程脚本完成）；"
        "正因为没有算术运算，check.py 中 patches 阶段的最大误差恰好为 0，可以用来单独确认 patch 顺序是否正确。"))
    s.append(code_block([
        "batch, channels, height, width = images.shape",
        "patch = self.patch_size",
        "grid_h, grid_w = height // patch, width // patch   # 224/32 -> 7 x 7",
        "grid = images.reshape(batch, channels, grid_h, patch, grid_w, patch)",
        "grid = grid.transpose(0, 2, 4, 1, 3, 5)           # [B,7,7,3,32,32]",
        "return grid.reshape(batch, grid_h * grid_w, channels * patch * patch)",
    ], red={0, 1, 2, 3, 4, 5}))
    s.append(Spacer(1, 3))

    s.append(P("3.2　多头注意力：拆分、打分、合并", h2))
    s.append(P(
        "<font face='Mono'>split_heads</font> 把最后一维 768 拆成 (12, 64) 并把 head 轴移到 batch 之后："
        "<font face='Mono'>[B,50,768] → [B,50,12,64] → [B,12,50,64]</font>，第 h 个 head 独占 64h…64h+63 的连续特征。"
        "打分时 q 与 k 的最后两轴相乘得到 <font face='Mono'>[B,12,50,50]</font>，再除以 √64 = 8；softmax 与加权由 "
        "<font face='Mono'>helpers.py</font> 提供；合并时先把 head 轴放回 token 轴之后（transpose 0,2,1,3）再 reshape，"
        "最后用 <font face='Mono'>out_proj</font> 做一次跨 768 维的线性混合。图 3 给出拆分/合并过程，并画出了块 0 第 2 个 head 的真实注意力矩阵。"))
    s.append(code_block([
        "grouped = x.reshape(batch, tokens, self.heads, self.head_dim)",
        "return grouped.transpose(0, 2, 1, 3)                # [B,H,N,D/H]",
        "...",
        "scores = q @ k.swapaxes(-1, -2)                     # [B,H,N,N]",
        "scores = scores * np.float32(1.0 / np.sqrt(self.head_dim))",
        "...",
        "context = coefficients @ v                          # [B,H,N,D/H]",
        "merged = context.transpose(0, 2, 1, 3).reshape(batch, tokens, self.width)",
        "return merged @ self.out_weight.T + self.out_bias",
    ], red={0, 1, 3, 4, 6, 7, 8}))
    s += figure(FIG / "fig3_attention.png", 16.2,
                "图 3　768 维特征按 12 个连续 64 维切片分给 12 个 head，各自计算 50×50 的注意力后再按原顺序合并。"
                "右下角是块 0 第 2 个 head 的真实注意力矩阵（颜色越暖权重越大）。")

    # 3.3 -----------------------------------------------------------------
    s.append(P("3.3　MLP 与两次残差", h2))
    s.append(P(
        "每个 block 有两次残差：<font face='Mono'>residual = x + attn(LN₁(x))</font>、"
        "<font face='Mono'>output = residual + MLP(LN₂(residual))</font>。MLP 先用 <font face='Mono'>c_fc</font> 把 768 扩到 3072，"
        "经精确 erf GELU 后再用 <font face='Mono'>c_proj</font> 压回 768；两次残差都加在未归一化的张量上，形状始终保持 "
        "<font face='Mono'>[B,50,768]</font>。"))
    s.append(code_block([
        "residual = x + update                                # TODO-04a",
        "hidden = u @ self.fc1_weight.T + self.fc1_bias       # 768 -> 3072",
        "activated = self.gelu(hidden)",
        "mlp_update = activated @ self.fc2_weight.T + self.fc2_bias",
        "return residual + mlp_update                         # TODO-04b",
    ], red={0, 1, 2, 3, 4}))

    # 4 -------------------------------------------------------------------
    s.append(P("4　数值验证", h1))
    s.append(P(
        "按课程 <font face='Mono'>check.py</font> 的阶段顺序检查，全部通过（表 2、图 4）。<b>patches</b> 与 <b>preprocess</b> "
        "的最大误差为 0，说明 reshape/transpose 没有引入额外舍入；<b>output</b> 的最大误差 6.63e-07，比 2e-5 的 atol 小两个数量级以上；"
        "<b>batch</b> 用乱序且重复的索引 [1,0,1] 重算，误差 5.07e-07，说明实现与批内位置无关，也没有返回缓存的参考行。"
        "check.py 还会断言每条输出向量的平方和为 1。全流程（10 张图 × 12 个 block）在本机 CPU 上约 9.5 s，含读取 326 MB 权重。"))
    header = [P("阶段", th), P("形状", th), P("max |Δ|", th), P("mean |Δ|", th),
              P("atol", th), P("结果", th)]
    metrics = [
        ("preprocess", "10×3×224×224", "0", "0", "2e-6"),
        ("patches", "2×49×3072", "0", "0", "2e-4"),
        ("tokens", "2×50×768", "4.77e-05", "1.30e-06", "2e-4"),
        ("block0", "2×50×768", "6.59e-06", "3.25e-07", "2e-4"),
        ("last_tokens", "2×50×768", "4.86e-05", "2.70e-06", "2e-4"),
        ("output", "10×512", "6.63e-07", "8.29e-08", "2e-5"),
        ("batch [1,0,1]", "3×512", "5.07e-07", "7.77e-08", "2e-5"),
    ]
    data = [header]
    for name, shape, mx, mean, atol in metrics:
        data.append([P(name, cell), P(shape, cell_c), P(mx, cell_c), P(mean, cell_c),
                     P(atol, cell_c), P("<font color='#21867A'>PASS</font>", cell_c)])
    table = Table(data, colWidths=[3.1 * cm, 3.6 * cm, 3.0 * cm, 3.0 * cm, 2.2 * cm, 2.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
    ]))
    table.repeatRows = 1
    s.append(KeepTogether([table,
                           P("表 2　各检查阶段与独立 FP32 参考的最大/平均绝对误差（rtol = 2e-4）。", caption)]))
    s += figure(FIG / "fig4_errors.png", 14.6,
                "图 4　各阶段最大绝对误差与验收容差（对数坐标）；前两个阶段逐位一致。")

    # 5 -------------------------------------------------------------------
    s.append(P("5　调试记录：一个 shape 变化与一个 bug", h1))
    s.append(P("5.1　shape 变化：patchify 中的四步轴变换", h2))
    s.append(P(
        "<font face='Mono'>[B,3,224,224]</font> 到 <font face='Mono'>[B,49,3072]</font> 并不是一次 reshape 能做到的：如果直接 "
        "<font face='Mono'>reshape(B,49,3072)</font>，内存会按 <font face='Mono'>C,H,W</font> 的顺序切分，得到的“行”既不是 patch，"
        "也不是 C,H,W 展平。正确做法是把两类信息分两步处理——先用 <font face='Mono'>reshape</font> 把空间轴拆成 "
        "<font face='Mono'>(7,32)</font>，让网格与块内互不干扰；再用 <font face='Mono'>transpose</font> 把轴顺序换成 "
        "<font face='Mono'>[B,7,7,3,32,32]</font>，此时每个 patch 的 3×32×32 个值在内存中连续；最后 flatten。"
        "图 2 中最上方的红色行就是网格第 0 行第 0 列的 patch，三段颜色依次对应 C=0/1/2 的 1024 个值。"))
    s += figure(FIG / "fig2_patchify.png", 15.0,
                "图 2　patchify 的形状变化：224×224 图像 → 7×7 网格 → 49×3072 的 patch 矩阵，行 0 即左上角 patch。")
    s.append(P("5.2　一个 bug：合并 head 时直接 reshape", h2))
    s.append(P(
        "第一版合并写成 <font face='Mono'>context.reshape(B, 50, 768)</font>，<b>block0 的 max |Δ| = 1.907</b>，远超 2e-4 的容差。"
        "原因是 <font face='Mono'>context</font> 的形状是 <font face='Mono'>[B,12,50,64]</font>，内存按“head → token → 64 维”排列，"
        "直接 reshape 会把第 50 行跨到下一个 head，token 顺序被彻底打乱；而正确语义要求每个 head 的 64 维回到它在 768 维中的位置。"
        "改成先 <font face='Mono'>transpose(0,2,1,3)</font>（把 head 轴放回 token 轴之后）再 reshape 后，误差降到 "
        "<b>6.586e-06</b>。对比脚本 <font face='Mono'>experiments/head_merge_demo.py</font> 可复现这两个数字。"
        "另一个同源陷阱是把 k 的转置误写成 reshape：q/k 是 4D，必须用 <font face='Mono'>swapaxes(-1,-2)</font> 只交换最后两轴。"))
    rows = [
        ("<font face='Mono'>context.reshape(B,50,768)</font>", "1.907e+00", "错误：head 与 token 交错"),
        ("<font face='Mono'>context.transpose(0,2,1,3).reshape(…)</font>", "6.586e-06", "正确：与参考一致"),
    ]
    data = [[P("合并写法", th), P("block0 max |Δ|", th), P("结论", th)]]
    data += [[P(a, cell), P(b, cell_c), P(c, cell)] for a, b, c in rows]
    table = Table(data, colWidths=[8.4 * cm, 3.4 * cm, 5.3 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    s.append(Spacer(1, 2))
    s.append(KeepTogether([table,
                           P("表 3　两种 head 合并写法在 block0 上的误差对比（同一份权重、同一批图像）。", caption)]))

    # 6 -------------------------------------------------------------------
    s.append(P("6　复现步骤与代码链接", h1))
    s.append(P(
        "在课程附件目录中执行下列命令即可复现全部数值结果（Python ≥ 3.9，只需 NumPy 与 Pillow）："))
    s.append(code_block([
        "python -m pip install -r requirements.txt",
        "python check.py --stage all",
        "python run.py --output output_tokens.npz",
    ], red=set()))
    s.append(Spacer(1, 4))
    s.append(P(
        "GitHub 仓库：<font face='Mono' color='#2F6FB3'>%s</font>。仓库包含 "
        "<font face='Mono'>image_encoder.py</font>、<font face='Mono'>experiments/</font>、"
        "<font face='Mono'>results/</font> 与本报告的源文件/图表；按课程要求未提交 326 MB 权重与参考输出。" %
        STUDENT["github"]))
    s.append(P("7　贡献与致谢", h1))
    s.append(P(
        "本人完成：六个 TODO（patchify、split_heads、attention、block、embed、CLS 输出）的 NumPy 实现与逐行注释，"
        "批量/乱序检查，head 合并顺序的对比实验，报告中的四张图与全部测量。"
        "使用的帮助：课程提供的 <font face='Mono'>helpers.py</font>、<font face='Mono'>preprocess_image.py</font>、"
        "<font face='Mono'>check.py</font>、权重与参考输出；课堂上 <font face='Mono'>transformer_steps.py</font> 的迷你算例"
        "（用三行、四特征核对过 Norm/注意力/MLP 的公式）；NumPy 官方文档中 reshape/transpose 的轴语义。"
        "未修改任何提供的文件，也未使用参考输出作为编码器输入。"))
    s.append(P("8　小结", h1))
    s.append(P(
        "六个 TODO 全部通过课程的多阶段数值检查，输出误差比容差小两个数量级；实现只依赖 reshape/transpose/矩阵乘法与广播，"
        "没有使用任何深度学习框架。最大的收获是理解了两件事：数组形状变化必须和内存顺序一起考虑（patch 的行优先顺序、"
        "head 轴的往返），以及残差连接要读未归一化的张量。该编码器可直接作为期末项目图像塔，与共享的 TransformerBlock "
        "一起复用。"))
    s.append(P("9　心得体会", h1))
    s.append(P(
        "动手前我以为最难的是注意力公式，实际卡住最久的却是“这个数组现在是什么形状、在内存里怎么排”。把每一行的形状写进"
        "注释、每完成一个 TODO 就跑一次 check.py，是这次最有效的两个习惯。另一个体会是分阶段定位：先用 block0 这种小阶段"
        "找到第一处不一致，比盯着最终的 512 维输出猜要快得多——误差从 1.9 降到 1e-6 的那次修复，就是靠 block0 的 "
        "max |Δ| 一眼看出问题出在合并 head，而不是权重方向。"))
    s.append(P("附录 A　提交文件说明", h1))
    s.append(P(
        "<font face='Mono'>image_encoder.py</font>：唯一修改的代码文件，含六个 TODO 的实现与逐行注释；"
        "<font face='Mono'>experiments/head_merge_demo.py</font>：表 3 的对比脚本；"
        "<font face='Mono'>results/check_all.txt</font>：一次 <font face='Mono'>check.py --stage all</font> 的完整输出；"
        "<font face='Mono'>results/stage_metrics.csv</font>：表 2/图 4 的数据；"
        "<font face='Mono'>REPORT/report.pdf</font>、<font face='Mono'>REPORT/figures/</font>：本报告与四张图。"
        "按课程要求，<font face='Mono'>weights/</font> 与 <font face='Mono'>assets/reference_outputs.npz</font> "
        "不随仓库提交（课程附件提供）。"))
    return s


def build():
    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
                          title="用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器 — 实验报告",
                          author=STUDENT["name"], subject="Python Programming Project 1")
    cover_frame = Frame(LM, BM, FRAME_W, FRAME_H, id="cover",
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    body_frame = Frame(LM, BM, FRAME_W, FRAME_H, id="body",
                       leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[cover_frame], onPage=cover_page),
        PageTemplate(id="Body", frames=[body_frame], onPage=body_page),
    ])
    story = []
    story += cover_story()
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())
    story += content_story()
    doc.build(story, canvasmaker=NumberedCanvas)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
