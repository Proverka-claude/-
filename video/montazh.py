"""Вертикальный ролик 9:16 с презентации волонтёрского клуба.

Склейки стоят в сетке 124 BPM, дроп музыки — 3.87 с (начало 3-го такта).
Переходы — из video/perehody.py (zoom in, zoom out, flash), без тряски.
Финал — все машут в камеру (на весь экран), надпись «Волонтёрский клуб» и уход в чёрный.

Запуск: python montazh.py <папка с IMG_*.mov> <выход.mp4>
"""
import math, os, subprocess, sys
import numpy as np
from PIL import Image
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
FF = imageio_ffmpeg.get_ffmpeg_exe()
SRC, OUT = sys.argv[1], sys.argv[2]

# --- переходы из perehody.py (без демо-рендера), в разрешении 1080x1920 ---
code = open(os.path.join(HERE, "perehody.py"), encoding="utf-8").read()
P = {}
exec(code.split("TRANS = [")[0], P)
W, H, FPS = 1080, 1920, 30
P["W"], P["H"] = W, H
P["yy"], P["xx"] = np.mgrid[0:H, 0:W]
TR = {"zoom_in": P["t_zoom_in"], "zoom_out": P["t_zoom_out"], "spin": P["t_spin"],
      "flash": P["t_flash"], "punch": P["t_punch"]}

BEAT = 60 / 124
TRF = 10  # длина перехода в кадрах (~1/3 с), по центру склейки

# (исходник, старт в сек, длина в битах, зум-кроп, центр x, центр y, переход В этот кусок)
EDIT = [
    # --- часть 1: презентация, зал смотрит и слушает ---
    ("IMG_4546", 4.00, 4, 1.0, .5, .5, None),         # спикеры у экрана, начало презентации
    ("IMG_4548", 0.00, 4, 1.0, .5, .5, None),         # зрители слушают
    ("IMG_4547", 1.60, 6, 1.5, .62, .42, "zoom_in"),  # ДРОП 3.87 с: спикер рассказывает, жестикулирует
    ("IMG_4546", 11.00, 2, 1.0, .5, .5, None),        # зал внимательно смотрит
    # --- часть 2: раздача подарков и значков ---
    ("IMG_4557", 0.30, 5, 1.0, .5, .5, "flash"),      # раздаёт подарки, из зала тянутся руки
    ("IMG_4558", 6.00, 5, 1.0, .5, .5, None),         # девушка смеётся в камеру
    ("IMG_4557", 14.20, 4, 1.0, .5, .5, None),        # вручает значки
    ("IMG_4557", 19.60, 3, 1.0, .5, .5, None),        # идёт в камеру с улыбкой
    ("IMG_4546", 21.20, 3, 1.0, .5, .5, "zoom_out"),  # «Как стать волонтёром?» + QR
    # --- финал: все машут в камеру (на весь экран), надпись, уход в чёрный ---
    ("IMG_4571", 0.95, 6, 1.0, .5, .5, "flash"),
]
LANDSCAPE = {"IMG_4571"}   # горизонтальное видео: на весь экран, плавная панорама по группе
LW = 3414                  # ширина горизонтального кадра при высоте 1920
PAN = (650, 1750)          # панорама слева направо (x левого края кропа)
FADE = 0.75                # уход в чёрный в конце, сек
TONE = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
        "zscale=t=bt709:m=bt709:r=tv,format=rgb24")


def load(name, t0, dur, z, cx, cy):
    """кусок исходника с запасом по краям: HDR->SDR, кроп, 1080x1920, 30 к/с"""
    pre = TRF / 2 / FPS + 0.05
    ss = max(0.0, t0 - pre)
    if name in LANDSCAPE:
        vf = f"scale={LW}:{H}:flags=lanczos,{TONE},fps={FPS},eq=saturation=1.12:contrast=1.04"
    else:
        cw, ch = 2160 / z, 3840 / z
        x = min(max(cx * 2160 - cw / 2, 0), 2160 - cw)
        y = min(max(cy * 3840 - ch / 2, 0), 3840 - ch)
        vf = (f"crop={cw:.0f}:{ch:.0f}:{x:.0f}:{y:.0f},scale={W}:{H}:flags=lanczos,"
              f"{TONE},fps={FPS},eq=saturation=1.12:contrast=1.04")
    raw = subprocess.run([FF, "-loglevel", "error", "-ss", f"{ss:.3f}",
                          "-i", os.path.join(SRC, name + ".mov"), "-t", f"{dur + 2 * pre:.3f}",
                          "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, H, LW if name in LANDSCAPE else W, 3)
    return fr, int(round((t0 - ss) * FPS))  # кадры и индекс точки входа


def zoom(img, s):
    """плавный наезд из центра"""
    if s <= 1.001: return img
    w, h = int(W / s), int(H / s)
    x, y = (W - w) // 2, (H - h) // 2
    return img.crop((x, y, x + w, y + h)).resize((W, H), Image.BILINEAR)


# --- надпись «ВОЛОНТЁРСКИЙ КЛУБ»: буквы «наливаются» сверху вниз по очереди ---
from PIL import ImageDraw, ImageFont, ImageFilter
FONT = os.path.join(HERE, "fonts", "Montserrat-Black.ttf")


def fit_font(text, width, size):
    while ImageFont.truetype(FONT, size).getlength(text) > width: size -= 2
    return ImageFont.truetype(FONT, size)


def glyphs(text, font, cy, grad=None, color=(255, 255, 255), shadow=0.6):
    """список (x, y, RGBA-слой буквы с тенью) для строки по центру экрана"""
    total = font.getlength(text); x = (W - total) / 2
    asc, desc = font.getmetrics(); hh = asc + desc
    out = []
    for ch in text:
        adv = font.getlength(ch)
        if ch.strip():
            pad = 30; cw, chh = int(adv) + 2 * pad, hh + 2 * pad
            m = Image.new("L", (cw, chh), 0)
            ImageDraw.Draw(m).text((pad, pad), ch, font=font, fill=255)
            if grad:
                g = np.linspace(0, 1, chh)[:, None, None]
                col = (np.array(grad[0]) * (1 - g) + np.array(grad[1]) * g) * np.ones((1, cw, 1))
                fill = Image.fromarray(col.astype(np.uint8))
            else:
                fill = Image.new("RGB", (cw, chh), color)
            layer = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
            sh = m.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * shadow))
            layer.paste((0, 0, 0, 255), (0, 6), sh)
            layer.paste(fill, (0, 0), m)
            out.append((x - pad, cy - hh / 2 - pad, layer))
        x += adv
    return out


L1, L2 = "ВОЛОНТЁРСКИЙ", "КЛУБ"
f1 = fit_font(L1, W - 110, 140)
f2 = fit_font(L2, W - 300, 250)
TEXT_Y = 470               # на белом экране проектора, над головами
GLYPHS = (glyphs(L1, f1, TEXT_Y, color=(28, 28, 48), shadow=0.25) +
          glyphs(L2, f2, TEXT_Y + 175, grad=((255, 90, 60), (235, 30, 100)), shadow=0.3))


def title(img, t):
    """t — сек от начала появления надписи"""
    if t <= 0: return img
    base = img.convert("RGBA")
    for k, (x, y, layer) in enumerate(GLYPHS):
        q = min(max((t - k * 0.055) / 0.5, 0), 1)
        if q <= 0: continue
        e = 1 - (1 - q) ** 3
        lw, lh = layer.size
        mask = np.zeros((lh, 1)); edge = e * lh * 1.25   # фронт «жидкости» идёт сверху вниз
        mask[:, 0] = np.clip((edge - np.arange(lh)) / 40, 0, 1)
        a = np.asarray(layer.getchannel("A"), float) * mask
        l2 = layer.copy(); l2.putalpha(Image.fromarray(a.astype(np.uint8)))
        if q < 1: l2 = l2.filter(ImageFilter.GaussianBlur(6 * (1 - e)))
        base.alpha_composite(l2, (int(x), int(y - 28 * (1 - e))))
    return base.convert("RGB")


# границы кусков в кадрах, строго по сетке битов
starts, b = [], 0
for e in EDIT:
    starts.append(int(round(b * BEAT * FPS))); b += e[2]
TOTAL = int(round(b * BEAT * FPS))
starts.append(TOTAL)
print(f"длительность {TOTAL / FPS:.2f} с, склеек {len(EDIT) - 1}")

cache = {}


def clip(i):
    if i not in cache:
        n, t0, beats, z, cx, cy, _ = EDIT[i]
        cache[i] = load(n, t0, beats * BEAT, z, cx, cy)
        for k in list(cache):
            if k < i - 1: del cache[k]
    return cache[i]


def frame(i, f):
    """кадр куска i, f — кадры от его начала (может быть <0 или за концом)"""
    fr, in0 = clip(i)
    a = fr[min(max(in0 + f, 0), len(fr) - 1)]
    if a.shape[1] != W:  # горизонтальный кадр: плавная панорама на весь экран
        q = min(max(f / (starts[i + 1] - starts[i]), 0), 1)
        x = int(PAN[0] + (PAN[1] - PAN[0]) * (3 * q * q - 2 * q ** 3))
        a = a[:, x:x + W]
    return Image.fromarray(a)


out = OUT + ".noaudio.mp4"
enc = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                        "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", out],
                       stdin=subprocess.PIPE)
half = TRF // 2
LAST = len(EDIT) - 1
for g in range(TOTAL):
    i = max(k for k in range(len(EDIT)) if starts[k] <= g)
    f = g - starts[i]
    nxt = i + 1 if i + 1 < len(EDIT) and EDIT[i + 1][6] and starts[i + 1] - g <= half else None
    if f < half and i > 0 and EDIT[i][6]:            # вторая половина перехода в кусок i
        p = (half + f + 1) / (TRF + 1)
        img = TR[EDIT[i][6]](frame(i - 1, starts[i] - starts[i - 1] + f), frame(i, f), p)
    elif nxt is not None:             # первая половина перехода в следующий кусок
        d = starts[nxt] - g           # 1..half
        p = (half - d + 1) / (TRF + 1)
        img = TR[EDIT[nxt][6]](frame(i, f), frame(nxt, -d), p)
    else:
        img = frame(i, f)
        if i == LAST - 1:             # слайд с QR — медленный наезд
            img = zoom(img, 1 + 0.05 * f / (starts[i + 1] - starts[i]))
    if i == LAST:
        img = title(img, (f - half) / FPS - 0.1)
        left = (TOTAL - g) / FPS      # уход в чёрный
        if left < FADE:
            k = max(0.0, (left - 0.12) / (FADE - 0.12))
            img = Image.fromarray((np.asarray(img, float) * k ** 1.5).astype(np.uint8))
    enc.stdin.write(img.convert("RGB").tobytes())
    if g % 30 == 0: print(f"{g / FPS:5.1f} с", flush=True)
enc.stdin.close(); enc.wait()

dur = TOTAL / FPS
subprocess.run([FF, "-y", "-loglevel", "error", "-i", out, "-i", os.path.join(HERE, "muzyka_klub.m4a"),
                "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-af", f"atrim=0:{dur:.3f},afade=t=out:st={dur - FADE - 0.3:.3f}:d={FADE + 0.2:.3f}",
                "-t", f"{dur:.3f}", "-movflags", "+faststart", OUT], check=True)
os.remove(out)
print("готово", OUT)
