"""Вертикальный ролик 9:16 с презентации волонтёрского клуба.

Склейки стоят в сетке 124 BPM, дроп музыки — 3.87 с (начало 3-го такта).
Переходы — только из video/perehody.py: zoom in, zoom out, spin, flash, punch.

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
    ("IMG_4557", 0.55, 3, 1.0, .5, .5, None),        # ХУК: раздача подарков, руки тянутся
    ("IMG_4547", 1.60, 2, 1.8, .62, .40, "zoom_in"),  # спикер рассказывает, жестикулирует
    ("IMG_4557", 2.40, 3, 1.0, .5, .5, "spin"),       # бросок в зал, ловят и улыбаются
    ("IMG_4558", 6.85, 3, 1.15, .65, .60, "punch"),   # ДРОП 3.87 с: девушка смеётся в камеру
    ("IMG_4557", 11.45, 2, 1.0, .5, .5, "zoom_out"),  # парень проходит сквозь кадр
    ("IMG_4557", 5.15, 2, 1.0, .5, .5, "zoom_in"),    # волонтёр показывает на зрителей, раздаёт
    ("IMG_4557", 14.30, 2, 1.0, .5, .5, "flash"),     # волонтёр с пакетом подарков
    ("IMG_4559", 0.60, 2, 1.6, .33, .38, "spin"),     # вручает в руки
    ("IMG_4547", 4.30, 2, 1.8, .60, .42, "punch"),    # объясняет, активные жесты
    ("IMG_4557", 18.50, 2, 1.0, .5, .5, "zoom_out"),  # раздача, реакция
    ("IMG_4558", 3.90, 2, 1.0, .5, .5, "zoom_in"),    # волонтёр идёт по залу
    ("IMG_4557", 20.50, 3, 1.0, .5, .5, "spin"),      # идёт в камеру с улыбкой
    ("IMG_4546", 21.20, 5, 1.0, .5, .5, "flash"),     # финал: «Как стать волонтёром?» + QR
]
DROP_BEAT = 8


def load(name, t0, dur, z, cx, cy):
    """кусок исходника с запасом по краям: HDR->SDR, кроп, 1080x1920, 30 к/с"""
    pre = TRF / 2 / FPS + 0.05
    ss = max(0.0, t0 - pre)
    cw, ch = 2160 / z, 3840 / z
    x = min(max(cx * 2160 - cw / 2, 0), 2160 - cw)
    y = min(max(cy * 3840 - ch / 2, 0), 3840 - ch)
    vf = (f"crop={cw:.0f}:{ch:.0f}:{x:.0f}:{y:.0f},scale={W}:{H}:flags=lanczos,"
          "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
          f"zscale=t=bt709:m=bt709:r=tv,format=rgb24,fps={FPS},eq=saturation=1.12:contrast=1.04")
    raw = subprocess.run([FF, "-loglevel", "error", "-ss", f"{ss:.3f}",
                          "-i", os.path.join(SRC, name + ".mov"), "-t", f"{dur + 2 * pre:.3f}",
                          "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    return fr, int(round((t0 - ss) * FPS))  # кадры и индекс точки входа


def pulse(img, s):
    """лёгкий зум-«качок» на сильную долю"""
    if s <= 1.001: return img
    w, h = int(W / s), int(H / s)
    x, y = (W - w) // 2, (H - h) // 2
    return img.crop((x, y, x + w, y + h)).resize((W, H), Image.BILINEAR)


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
    return Image.fromarray(fr[min(max(in0 + f, 0), len(fr) - 1)])


out = OUT + ".noaudio.mp4"
enc = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                        "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", out],
                       stdin=subprocess.PIPE)
half = TRF // 2
for g in range(TOTAL):
    i = max(k for k in range(len(EDIT)) if starts[k] <= g)
    f = g - starts[i]
    nxt = i + 1 if i + 1 < len(EDIT) and starts[i + 1] - g <= half else None
    if f < half and i > 0:            # вторая половина перехода в кусок i
        p = (half + f + 1) / (TRF + 1)
        img = TR[EDIT[i][6]](frame(i - 1, starts[i] - starts[i - 1] + f), frame(i, f), p)
    elif nxt is not None:             # первая половина перехода в следующий кусок
        d = starts[nxt] - g           # 1..half
        p = (half - d + 1) / (TRF + 1)
        img = TR[EDIT[nxt][6]](frame(i, f), frame(nxt, -d), p)
    else:
        img = frame(i, f)
        if i == 0 and f < 8:          # удар-вход на хуке
            img = pulse(img, 1 + 0.18 * (1 - f / 8) ** 2)
        elif g >= DROP_BEAT * BEAT * FPS:  # после дропа — качок на каждую долю
            ph = (g / FPS / BEAT) % 1
            img = pulse(img, 1 + 0.035 * math.exp(-ph * 9))
        if i == len(EDIT) - 1:        # финальный слайд — медленный наезд
            img = pulse(img, 1 + 0.06 * f / (starts[-1] - starts[i]))
    enc.stdin.write(img.convert("RGB").tobytes())
    if g % 30 == 0: print(f"{g / FPS:5.1f} с", flush=True)
enc.stdin.close(); enc.wait()

dur = TOTAL / FPS
subprocess.run([FF, "-y", "-loglevel", "error", "-i", out, "-i", os.path.join(HERE, "muzyka_klub.m4a"),
                "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-af", f"atrim=0:{dur:.3f},afade=t=out:st={dur - 0.8:.3f}:d=0.8",
                "-t", f"{dur:.3f}", "-movflags", "+faststart", OUT], check=True)
os.remove(out)
print("готово", OUT)
