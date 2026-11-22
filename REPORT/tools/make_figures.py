"""Draw the report figures for the NumPy ViT-B/32 image encoder project.

Each figure is drawn with Pillow on an oversized canvas (SS times the final
pixel size) and then resampled with LANCZOS, which gives smooth edges and
readable text at 300 dpi print size.

Run from this folder:  python make_figures.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

SS = 3  # supersampling factor
TEXT_SCALE = 1.18  # all labels are enlarged so they stay readable in print

FONT_DIR = Path(r"C:\Windows\Fonts")
F_UI = FONT_DIR / "segoeui.ttf"
F_UI_B = FONT_DIR / "segoeuib.ttf"
F_CN = FONT_DIR / "msyh.ttc"
F_CN_B = FONT_DIR / "msyhbd.ttc"
F_MONO = FONT_DIR / "consola.ttf"
F_MONO_B = FONT_DIR / "consolab.ttf"

# palette
INK = "#1F2A37"
NAVY = "#1F3864"
BLUE = "#2F6FB3"
BLUE_L = "#E8F1FB"
TEAL = "#21867A"
TEAL_L = "#E3F3F0"
ORANGE = "#C9721F"
ORANGE_L = "#FCEFE1"
RED = "#C00000"
RED_L = "#FBE7E7"
GRAY = "#5B6B7C"
GRAY_L = "#F2F5F9"
LINE = "#C3D0DE"


def font(path, size):
    return ImageFont.truetype(str(path), int(round(size * TEXT_SCALE * SS)))


class Fig:
    def __init__(self, width, height, bg="white"):
        self.w, self.h = width, height
        self.img = Image.new("RGB", (width * SS, height * SS), bg)
        self.d = ImageDraw.Draw(self.img)

    # --- primitives -----------------------------------------------------
    def rect(self, box, fill=None, outline=None, width=2, radius=0):
        x0, y0, x1, y1 = [v * SS for v in box]
        if radius:
            self.d.rounded_rectangle((x0, y0, x1, y1), radius=radius * SS,
                                     fill=fill, outline=outline, width=int(width * SS))
        else:
            self.d.rectangle((x0, y0, x1, y1), fill=fill, outline=outline,
                             width=int(width * SS))

    def line(self, pts, fill=LINE, width=2):
        self.d.line([(x * SS, y * SS) for x, y in pts], fill=fill, width=int(width * SS))

    def text(self, xy, s, f, fill=INK, anchor="mm", spacing=6):
        self.d.multiline_text((xy[0] * SS, xy[1] * SS), s, font=f, fill=fill,
                              anchor=anchor, align="center", spacing=int(spacing * SS))

    def arrow(self, x0, y0, x1, y1, color=BLUE, width=3, head=9):
        self.line([(x0, y0), (x1, y1)], fill=color, width=width)
        dx, dy = x1 - x0, y1 - y0
        n = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / n, dy / n
        px, py = -uy, ux
        bx, by = x1 - ux * head, y1 - uy * head
        self.d.polygon([(x1 * SS, y1 * SS),
                        ((bx + px * head * 0.45) * SS, (by + py * head * 0.45) * SS),
                        ((bx - px * head * 0.45) * SS, (by - py * head * 0.45) * SS)],
                       fill=color)

    def save(self, path):
        out = self.img.resize((self.w, self.h), Image.LANCZOS)
        out.save(path, "PNG", optimize=True)
        print("saved", path, out.size)


def preprocessed_rgb(index=0):
    """Rebuild the uint8 RGB view of a supplied preprocessed image."""
    with np.load(PROJECT / "assets/reference_outputs.npz", allow_pickle=False) as ref:
        chw = ref["preprocessed"][index]
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
    pixels = np.clip(chw * std + mean, 0.0, 1.0)
    return Image.fromarray((pixels.transpose(1, 2, 0) * 255).round().astype(np.uint8))


def block0_attention(head=0):
    """Attention coefficients of block 0, head `head`, first image."""
    from runtime import load_weights, load_images
    from image_encoder import ImageEncoder
    from helpers import attention_weights

    encoder = ImageEncoder(load_weights())
    images = load_images()[:1]
    x = encoder.pre_norm(encoder.embed(images))
    attn = encoder.blocks[0].attention
    qkv = x @ attn.qkv_weight.T + attn.qkv_bias
    w = attn.width
    q = attn.split_heads(qkv[:, :, :w])
    k = attn.split_heads(qkv[:, :, w:2 * w])
    scores = (q @ k.swapaxes(-1, -2)) * np.float32(1.0 / np.sqrt(attn.head_dim))
    return attention_weights(scores)[0, head]


# ---------------------------------------------------------------------------
# Figure 1: encoder pipeline and the block it repeats twelve times
# ---------------------------------------------------------------------------
def figure_architecture(path):
    W, H = 2000, 780
    fig = Fig(W, H)
    f_card = font(F_UI_B, 26)
    f_small = font(F_UI, 20)
    f_shape = font(F_MONO, 21)
    f_note = font(F_UI, 21)

    cards = [
        ("Image input", "[B,3,224,224]", "224 × 224 RGB"),
        ("Patchify  7×7", "[B,49,3072]", "32×32 patches, C,H,W"),
        ("Patch projection", "[B,49,768]", "linear, no bias"),
        ("+ CLS + positions", "[B,50,768]", "1 + 49 token rows"),
        ("12 × block", "[B,50,768]", "after LN_pre"),
        ("CLS → output", "[B,512]", "LN_post · proj · L2"),
    ]
    top, card_h, card_w, gap = 48, 148, 272, 42
    x0 = (W - (len(cards) * card_w + (len(cards) - 1) * gap)) / 2
    for i, (title, shape, detail) in enumerate(cards):
        x = x0 + i * (card_w + gap)
        accent = ORANGE if i == len(cards) - 1 else BLUE
        fill = ORANGE_L if i == len(cards) - 1 else BLUE_L
        fig.rect((x, top, x + card_w, top + card_h), fill=fill, outline=accent, width=2, radius=15)
        fig.d.ellipse([(x + 14) * SS, (top + 12) * SS, (x + 36) * SS, (top + 34) * SS],
                      fill=accent)
        fig.text((x + 25, top + 23), str(i + 1), font(F_UI_B, 14), "white")
        fig.text((x + card_w / 2, top + 48), title, f_card, NAVY if i < 5 else ORANGE)
        fig.text((x + card_w / 2, top + 94), shape, f_shape, GRAY)
        fig.text((x + card_w / 2, top + 124), detail, f_small, GRAY)
        if i < len(cards) - 1:
            ax0 = x + card_w + 7
            fig.arrow(ax0, top + card_h / 2, ax0 + gap - 14, top + card_h / 2, BLUE, 3, 9)

    # --- block panel ------------------------------------------------------
    px0, px1 = 60, W - 60
    py0 = 248
    chain_y = py0 + 148
    chip_h = 70
    fig.rect((px0, py0, px1, py0 + 430), fill="#FBFCFE", outline=LINE, width=2, radius=18)
    fig.text((px0 + 34, py0 + 32),
             "Transformer block  × 12   —   D = 768, H = 12, head_dim = 64, MLP width = 3072",
             font(F_UI_B, 23), NAVY, anchor="lm")

    def chip(x, y, w, title, sub, fill, outline, tsize=21):
        fig.rect((x, y - chip_h / 2, x + w, y + chip_h / 2), fill=fill, outline=outline,
                 width=2, radius=10)
        fig.text((x + w / 2, y - 9), title, font(F_UI_B, tsize), INK)
        fig.text((x + w / 2, y + 17), sub, font(F_MONO, 15), GRAY)
        return x + w

    def plus_chip(x, y, w, sub, outline):
        fig.rect((x, y - chip_h / 2, x + w, y + chip_h / 2), fill="white", outline=outline,
                 width=2, radius=10)
        cx, cy, r = x + w / 2, y - 9, 15
        fig.d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
                      outline=outline, width=int(2 * SS))
        fig.line([(cx - r + 5, cy), (cx + r - 5, cy)], fill=outline, width=2)
        fig.line([(cx, cy - r + 5), (cx, cy + r - 5)], fill=outline, width=2)
        fig.text((x + w / 2, y + 19), sub, font(F_MONO, 14), GRAY)

    # fixed chain pieces
    ln1_w, ln2_w, add_w, gap = 112, 112, 86, 26
    attn_w, mlp_w = 790, 452
    total = ln1_w + attn_w + add_w + ln2_w + mlp_w + add_w + 5 * gap
    cx = (W - total) / 2
    y_in = chain_y

    # LN1
    ln1_x = cx
    chip(ln1_x, y_in, ln1_w, "LN₁", "768", TEAL_L, TEAL)

    # attention group
    ag_x = ln1_x + ln1_w + gap
    fig.rect((ag_x, py0 + 58, ag_x + attn_w, py0 + 236), fill="#F4F9F7", outline=TEAL,
             width=2, radius=14)
    fig.text((ag_x + 18, py0 + 80), "multi-head self-attention", font(F_UI_B, 18), TEAL,
             anchor="lm")
    inner = [("QKV", "2304", 88), ("12 heads", "50×64", 104), ("QKᵀ/√64", "50×50", 112),
             ("softmax", "50×50", 100), ("× V", "50×64", 84), ("merge", "768", 92),
             ("proj", "768", 80)]
    inner_gap = (attn_w - 36 - sum(w for _, _, w in inner)) / (len(inner) - 1)
    ax = ag_x + 18
    for j, (t, s, w) in enumerate(inner):
        chip(ax, y_in + 20, w, t, s, "white", TEAL, 18)
        ax += w
        if j < len(inner) - 1:
            fig.arrow(ax + 3, y_in + 20, ax + inner_gap - 4, y_in + 20, TEAL, 2.4, 6)
            ax += inner_gap

    # residual add 1 (attention)
    add1_x = ag_x + attn_w + gap
    plus_chip(add1_x, y_in, add_w, "x + attn", ORANGE)

    # LN2
    ln2_x = add1_x + add_w + gap
    chip(ln2_x, y_in, ln2_w, "LN₂", "768", TEAL_L, TEAL)

    # MLP group
    mg_x = ln2_x + ln2_w + gap
    fig.rect((mg_x, py0 + 58, mg_x + mlp_w, py0 + 236), fill="#FDF6EF", outline=ORANGE,
             width=2, radius=14)
    fig.text((mg_x + 18, py0 + 80), "MLP", font(F_UI_B, 18), ORANGE, anchor="lm")
    mp = [("FC 768→3072", "hidden", 152), ("GELU", "erf", 84), ("FC 3072→768", "768", 152)]
    mgap = (mlp_w - 36 - sum(w for _, _, w in mp)) / (len(mp) - 1)
    mx = mg_x + 18
    for j, (t, s, w) in enumerate(mp):
        chip(mx, y_in + 20, w, t, s, "white", ORANGE, 17)
        mx += w
        if j < len(mp) - 1:
            fig.arrow(mx + 3, y_in + 20, mx + mgap - 4, y_in + 20, ORANGE, 2.4, 6)
            mx += mgap

    # residual add 2 (MLP)
    add2_x = mg_x + mlp_w + gap
    plus_chip(add2_x, y_in, add_w, "mlp", ORANGE)

    # group-to-group arrows on the main axis
    fig.arrow(ln1_x + ln1_w + 4, y_in, ag_x - 4, y_in, NAVY, 2.6, 7)
    fig.arrow(ag_x + attn_w + 4, y_in, add1_x - 4, y_in, NAVY, 2.6, 7)
    fig.arrow(add1_x + add_w + 4, y_in, ln2_x - 4, y_in, NAVY, 2.6, 7)
    fig.arrow(ln2_x + ln2_w + 4, y_in, mg_x - 4, y_in, NAVY, 2.6, 7)

    # residual rails below the chain
    rail1 = y_in + 112
    rail2 = y_in + 152
    fig.line([(ln1_x - 26, y_in), (ln1_x - 26, rail1)], fill=ORANGE, width=2)
    fig.line([(ln1_x - 26, rail1), (add1_x + add_w / 2, rail1)], fill=ORANGE, width=2)
    fig.arrow(add1_x + add_w / 2, rail1, add1_x + add_w / 2, y_in + chip_h / 2 + 8, ORANGE, 2, 6)
    fig.d.ellipse([(ln1_x - 31) * SS, (y_in - 5) * SS, (ln1_x - 21) * SS, (y_in + 5) * SS],
                  fill=ORANGE)
    fig.line([(add1_x + add_w / 2, y_in + chip_h / 2), (add1_x + add_w / 2, rail2)],
             fill=ORANGE, width=2)
    fig.line([(add1_x + add_w / 2, rail2), (add2_x + add_w / 2, rail2)], fill=ORANGE, width=2)
    fig.arrow(add2_x + add_w / 2, rail2, add2_x + add_w / 2, y_in + chip_h / 2 + 8, ORANGE, 2, 6)
    fig.text((ln1_x + 100, rail1 - 16), "x", font(F_UI_B, 20), ORANGE, anchor="lm")
    fig.text((add1_x + add_w + 60, rail2 - 16), "residual", font(F_UI, 18), ORANGE, anchor="lm")

    fig.text((W / 2, py0 + 322),
             "each + reads the unnormalized block input, so both residual paths keep shape [B,50,768]",
             font(F_UI, 19), GRAY)
    fig.text((W / 2, py0 + 360),
             "patch projection has no bias · linear weights stored [out,in] · visual.proj is [in,out]",
             f_note, GRAY)
    fig.save(path)




# ---------------------------------------------------------------------------
# Figure 2: patchify and the resulting 49 x 3072 token rows
# ---------------------------------------------------------------------------
def block0_attention_all():
    """Attention coefficients of block 0 for all 12 heads, first image."""
    from runtime import load_weights, load_images
    from image_encoder import ImageEncoder
    from helpers import attention_weights

    encoder = ImageEncoder(load_weights())
    images = load_images()[:1]
    x = encoder.pre_norm(encoder.embed(images))
    attn = encoder.blocks[0].attention
    qkv = x @ attn.qkv_weight.T + attn.qkv_bias
    w = attn.width
    q = attn.split_heads(qkv[:, :, :w])
    k = attn.split_heads(qkv[:, :, w:2 * w])
    scores = (q @ k.swapaxes(-1, -2)) * np.float32(1.0 / np.sqrt(attn.head_dim))
    return attention_weights(scores)[0]


def vtext(fig, xy, s, f, fill):
    """Paste a 90-degree rotated text label (used for the heat-map axis)."""
    probe = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    width, height = ImageDraw.Draw(probe).textbbox((0, 0), s, font=f)[2:]
    tile = Image.new("RGBA", (width + 8, height + 8), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((4, 4), s, font=f, fill=fill)
    tile = tile.rotate(90, expand=True)
    fig.d._image.paste(tile, (int(xy[0] * SS), int(xy[1] * SS)), tile)


# ---------------------------------------------------------------------------
# Figure 2: patchify and the resulting 49 x 3072 patch rows
# ---------------------------------------------------------------------------
def figure_patchify(path):
    W, H = 1700, 620
    fig = Fig(W, H)
    f_title = font(F_UI_B, 23)
    f_body = font(F_UI, 19)
    f_shape = font(F_MONO, 18)
    f_tiny = font(F_UI, 16)
    f_mono_s = font(F_MONO, 14)

    # --- real preprocessed image with the 7x7 patch grid -------------------
    img = preprocessed_rgb().resize((300 * SS, 300 * SS), Image.LANCZOS)
    img_x, img_y, size = 50, 110, 300
    fig.d._image.paste(img, (img_x * SS, img_y * SS))
    cell = size / 7
    for i in range(1, 7):
        fig.line([(img_x + i * cell, img_y), (img_x + i * cell, img_y + size)],
                 fill=(255, 255, 255), width=1)
        fig.line([(img_x, img_y + i * cell), (img_x + size, img_y + i * cell)],
                 fill=(255, 255, 255), width=1)
    fig.rect((img_x, img_y, img_x + cell, img_y + cell), outline=RED, width=3)
    fig.rect((img_x, img_y, img_x + size, img_y + size), outline=NAVY, width=2)
    fig.text((img_x + size / 2, img_y - 30), "224 × 224 RGB input", f_title, NAVY)
    fig.text((img_x + size / 2, img_y + size + 28), "[B,3,224,224]", f_shape, GRAY)

    # --- patchify arrow ----------------------------------------------------
    fig.arrow(372, 260, 462, 260, BLUE, 3, 10)
    fig.text((417, 228), "patchify", font(F_MONO_B, 15), BLUE)

    # --- the 49 x 3072 patch matrix ---------------------------------------
    mx, my, mw, mh = 500, 110, 700, 380
    rows = 49
    row_h = mh / rows
    seg_cols = ["#F3D8D8", "#FBE7C8", "#D9E9F7"]
    fig.rect((mx, my, mx + mw, my + mh), fill="#FFFFFF", outline=LINE, width=2)
    for r in range(1, rows):
        fig.line([(mx, my + r * row_h), (mx + mw, my + r * row_h)], fill="#E6ECF3", width=1)
    for c in range(1, 3):
        fig.line([(mx + c * mw / 3, my), (mx + c * mw / 3, my + mh)], fill=LINE, width=1)
    for i, colour in enumerate(seg_cols):
        fig.rect((mx + i * mw / 3, my, mx + (i + 1) * mw / 3, my + row_h), fill=colour)
    fig.rect((mx - 2, my - 2, mx + mw + 2, my + row_h + 2), outline=RED, width=2)
    fig.text((mx + mw / 2, my - 38), "49 patch rows × 3072 values", f_title, NAVY)
    fig.text((mx + mw / 2, my + mh + 26), "[B,49,3072]", f_shape, GRAY)
    fig.text((mx, my + mh + 54), "row 0 = patch (0,0)",
             font(F_UI_B, 15), RED, anchor="lm")

    # --- legend for the three channel segments of one row -----------------
    lx = 1260
    fig.text((lx, 120), "one row, left to right:", font(F_UI_B, 18), NAVY, anchor="lm")
    entries = [("C = 0  ·  32×32 = 1024", "#F3D8D8"),
               ("C = 1  ·  32×32 = 1024", "#FBE7C8"),
               ("C = 2  ·  32×32 = 1024", "#D9E9F7")]
    for i, (label, colour) in enumerate(entries):
        y = 145 + i * 54
        fig.rect((lx, y, lx + 330, y + 40), fill=colour, outline=LINE, width=1, radius=8)
        fig.text((lx + 165, y + 20), label, f_mono_s, INK)
    fig.text((lx, 340),
             "inside one patch the values are\nflattened C, H, W; the 7×7 cells are "
             "then\nnumbered left→right, top→bottom.",
             f_body, GRAY, anchor="la", spacing=10)

    # --- the three shapes at the bottom -----------------------------------
    fig.text((W / 2, 586),
             "[B,3,224,224]  →reshape→  [B,3,7,32,7,32]  →transpose (0,2,4,1,3,5)→  "
             "[B,7,7,3,32,32]  →reshape→  [B,49,3072]",
             font(F_MONO, 14), GRAY)
    fig.save(path)


# ---------------------------------------------------------------------------
# Figure 3: splitting and merging the 12 attention heads
# ---------------------------------------------------------------------------
STEM_COLOURS = ["#2F6FB3", "#3C8DBC", "#4AA8C4", "#57B8B4", "#6CBFA3", "#8CC292",
                "#B4C47F", "#D4BE72", "#E5AE6A", "#E8946A", "#DE7A74", "#C96A86"]
STEM_LIGHT = ["#DCE8F5", "#DFEEF6", "#E0F2F2", "#E2F4EE", "#E6F4E9", "#EAF5E6",
              "#F0F5DF", "#F5F3DA", "#F8EBDD", "#FAE6DD", "#F9DEDC", "#F7DEE4"]


def figure_heads(path):
    W, H = 1700, 660
    fig = Fig(W, H)
    f_title = font(F_UI_B, 23)
    f_body = font(F_UI, 18)
    f_shape = font(F_MONO, 17)
    f_tiny = font(F_UI, 15)
    f_arrow = font(F_MONO_B, 14)

    def band_matrix(box, nbands, palette, radius=6):
        x0, y0, x1, y1 = box
        fig.rect(box, fill="#FFFFFF", outline=LINE, width=2, radius=radius)
        bw = (x1 - x0) / nbands
        for i in range(nbands):
            fig.rect((x0 + i * bw, y0, x0 + (i + 1) * bw, y1), fill=palette[i],
                     outline="#FFFFFF", width=1)
        fig.rect(box, outline=LINE, width=2, radius=radius)

    # --- x [B,50,768] ------------------------------------------------------
    AX0, AY0, AX1, AY1 = 55, 170, 315, 470
    band_matrix((AX0, AY0, AX1, AY1), 12, STEM_LIGHT)
    fig.text((AX0 + 130, AY0 - 32), "x   [B,50,768]", f_shape, NAVY)
    fig.text((AX0 + 130, AY1 + 28), "50 tokens × 768 features", f_body, GRAY)
    fig.text((AX0 + 130, AY1 + 52), "12 contiguous 64-wide slices", f_tiny, GRAY)

    # --- split arrow -------------------------------------------------------
    fig.arrow(330, 320, 412, 320, BLUE, 3, 9)
    fig.text((371, 292), "split", f_arrow, BLUE)
    fig.text((371, 348), "reshape", font(F_MONO, 12), GRAY)
    fig.text((371, 366), "(B,50,12,64)", font(F_MONO, 12), GRAY)
    fig.text((371, 392), "transpose", font(F_MONO, 12), GRAY)
    fig.text((371, 410), "(0,2,1,3)", font(F_MONO, 12), GRAY)

    # --- head stack --------------------------------------------------------
    BX0, BY0 = 432, 190
    for k in range(3):
        x0, y0 = BX0 + k * 70, BY0
        fig.rect((x0, y0, x0 + 190, y0 + 270), fill=STEM_LIGHT[k * 4], outline=STEM_COLOURS[k * 4],
                 width=2, radius=8)
        for r in range(1, 12):
            fig.line([(x0 + 4, y0 + r * 270 / 12), (x0 + 186, y0 + r * 270 / 12)],
                     fill="#FFFFFF", width=1)
        fig.text((x0 + 14, y0 + 18), ["head 0", "head 1", "head 11"][k], font(F_UI_B, 15),
                 STEM_COLOURS[k * 4], anchor="lm")
    fig.text((BX0 + 130, BY0 - 32), "12 independent heads", f_title, NAVY)
    fig.text((BX0 + 130, BY0 + 330), "[B,12,50,64]", f_shape, GRAY)
    fig.text((BX0 + 130, BY0 + 354), "each head: 50 tokens × 64 features", f_tiny, GRAY)

    # --- attention map of the most structured head -------------------------
    coef_all = block0_attention_all()
    entropies = []
    for h in range(12):
        rows = coef_all[h]
        entropies.append(float(-(rows * np.log(rows + 1e-12)).sum(-1).mean()))
    head = int(np.argmax(entropies))
    coef = coef_all[head]

    HX, HY, HS = 820, 190, 270
    fig.text((HX + HS / 2, HY - 66), "attention of one head", f_title, NAVY)
    fig.text((HX + HS / 2, HY - 40), "softmax(QKᵀ/√64)", f_shape, GRAY)
    fig.text((HX + HS / 2, HY - 18), "head %d, block 0, 50 × 50 (most spread of the 12)"
             % head, f_tiny, GRAY)

    def cmap(t):
        stops = [(0.0, (247, 251, 255)), (0.25, (198, 219, 239)), (0.45, (116, 173, 209)),
                 (0.62, (254, 196, 120)), (0.80, (244, 120, 80)), (1.0, (150, 20, 40))]
        t = min(max(t, 0.0), 1.0)
        for i in range(len(stops) - 1):
            a, ca = stops[i]
            b, cb = stops[i + 1]
            if a <= t <= b:
                u = (t - a) / (b - a)
                return tuple(int(ca[j] + (cb[j] - ca[j]) * u) for j in range(3))
        return stops[-1][1]

    values = (coef / max(coef.max(), 1e-12)) ** 0.35
    cell = HS / 50
    for r in range(50):
        for c in range(50):
            fig.d.rectangle([(HX + c * cell) * SS, (HY + r * cell) * SS,
                             (HX + (c + 1) * cell) * SS, (HY + (r + 1) * cell) * SS],
                            fill=cmap(values[r, c]))
    fig.rect((HX - 1, HY - 1, HX + HS + 1, HY + HS + 1), outline=LINE, width=2)
    fig.text((HX + HS / 2, HY + HS + 22), "key token (49 patches + CLS)", f_tiny, GRAY)
    vtext(fig, (HX - 60, HY + HS / 2 - 70), "query token", font(F_UI, 15), GRAY)

    # --- merge back --------------------------------------------------------
    MX0, MY0, MX1, MY1 = 1250, 170, 1510, 470
    fig.arrow(1130, 320, 1216, 320, BLUE, 3, 9)
    fig.text((1150, 288), "merge", f_arrow, RED)
    fig.text((1150, 352), "transpose (0,2,1,3)", font(F_MONO, 13), RED)
    fig.text((1150, 372), "reshape (B,50,768)", font(F_MONO, 13), RED)
    band_matrix((MX0, MY0, MX1, MY1), 12, STEM_LIGHT)
    fig.text((MX0 + 130, MY0 - 32), "merged  [B,50,768]", f_shape, NAVY)
    fig.text((MX0 + 130, MY1 + 28), "out_proj → [B,50,768]", f_body, GRAY)

    fig.text((W / 2, 600),
             "the head axis must return inside the feature axis before the final reshape — "
             "a direct reshape of [B,12,50,64] would interleave tokens",
             f_tiny, RED)
    fig.save(path)


# ---------------------------------------------------------------------------
# Figure 4: per-stage numerical error against the acceptance tolerance
# ---------------------------------------------------------------------------
def figure_errors(path, metrics):
    W, H = 1700, 560
    fig = Fig(W, H)
    f_title = font(F_UI_B, 23)
    f_axis = font(F_UI, 18)
    f_val = font(F_MONO_B, 15)
    f_tiny = font(F_UI, 16)

    left, right, top, bottom = 130, 1660, 90, 440
    names = [m["stage"] for m in metrics]
    values = [m["max"] for m in metrics]
    tols = [m["atol"] for m in metrics]

    lo, hi = 1e-8, 1e-2
    import math

    def ypos(v):
        v = max(v, lo)
        return bottom - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (bottom - top)

    # grid + axis
    for exponent in range(-8, -1):
        y = ypos(10.0 ** exponent)
        fig.line([(left, y), (right, y)], fill="#E7ECF2", width=1)
        fig.text((left - 12, y), "1e%d" % exponent, f_axis, GRAY, anchor="rm")
    fig.line([(left, top - 20), (left, bottom)], fill=GRAY, width=2)
    fig.line([(left, bottom), (right, bottom)], fill=GRAY, width=2)

    # tolerance step line
    slot = (right - left) / len(names)
    points = []
    for i, tol in enumerate(tols):
        x0 = left + i * slot
        points += [(x0, ypos(tol)), (x0 + slot, ypos(tol))]
    fig.line(points, fill=RED, width=2)
    fig.text((right - 10, ypos(1e-3) + 26), "acceptance tolerance  |Δ| ≤ atol + rtol·|ref|",
             font(F_UI_B, 16), RED, anchor="rm")

    # bars
    for i, (name, value) in enumerate(zip(names, values)):
        cx = left + i * slot + slot / 2
        bw = slot * 0.52
        y = ypos(value)
        fig.rect((cx - bw / 2, y, cx + bw / 2, bottom), fill=TEAL, outline=TEAL, width=1,
                 radius=4)
        label = "0 (exact)" if value == 0 else "%.2e" % value
        fig.text((cx, y - 14), label, f_val, TEAL)
        fig.text((cx, bottom + 24), name, f_axis, INK)

    fig.text((left, 40), "Maximum absolute error per check stage (log scale)", f_title, NAVY,
             anchor="lm")
    fig.text((right, 40), "all 7 numerical stages PASS", font(F_UI_B, 20), TEAL, anchor="rm")
    fig.text((left + 6, bottom + 58), "rtol = 2e-4 for every stage; zero error means bitwise-identical float32 output",
             f_tiny, GRAY, anchor="lm")
    fig.save(path)


METRICS = [
    {"stage": "preprocess", "max": 0.0, "mean": 0.0, "atol": 2e-6},
    {"stage": "patches", "max": 0.0, "mean": 0.0, "atol": 2e-4},
    {"stage": "tokens", "max": 4.768e-05, "mean": 1.296e-06, "atol": 2e-4},
    {"stage": "block0", "max": 6.586e-06, "mean": 3.252e-07, "atol": 2e-4},
    {"stage": "last_tokens", "max": 4.864e-05, "mean": 2.696e-06, "atol": 2e-4},
    {"stage": "output", "max": 6.631e-07, "mean": 8.289e-08, "atol": 2e-5},
    {"stage": "batch", "max": 5.066e-07, "mean": 7.768e-08, "atol": 2e-5},
]

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "figures"
    out.mkdir(exist_ok=True)
    figure_architecture(out / "fig1_architecture.png")
    figure_patchify(out / "fig2_patchify.png")
    figure_heads(out / "fig3_attention.png")
    figure_errors(out / "fig4_errors.png", METRICS)
