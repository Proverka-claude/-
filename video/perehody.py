import math, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import imageio_ffmpeg

W, H, FPS = 540, 960, 30
HOLD, TR = 0.9, 0.5
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F_BIG = ImageFont.truetype(FB, 96)
F_T = ImageFont.truetype(FB, 38)
F_S = ImageFont.truetype(FB, 24)

PALS = [((255, 94, 58), (255, 42, 104)), ((20, 30, 120), (0, 200, 255)),
        ((10, 120, 80), (180, 255, 60)), ((120, 20, 160), (255, 120, 220)),
        ((255, 170, 0), (255, 60, 0)), ((0, 70, 90), (0, 230, 170))]
yy, xx = np.mgrid[0:H, 0:W]


def scene(k, t, label):
    c1, c2 = (np.array(c, float) for c in PALS[k % len(PALS)])
    g = ((xx / W + yy / H) / 2 + 0.15 * math.sin(t * 2))[..., None].clip(0, 1)
    img = Image.fromarray((c1 * (1 - g) + c2 * g).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    for i in range(6):
        r = 40 + 25 * i
        cx = W / 2 + math.cos(t * 1.7 + i * 1.1) * (120 + 20 * i)
        cy = H / 2 + math.sin(t * 1.3 + i * 0.9) * (260 + 15 * i)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 45))
    for gx in range(0, W, 90):
        d.line([(gx, 0), (gx, H)], fill=(255, 255, 255, 25), width=2)
    d.text((W / 2, H / 2), label, font=F_BIG, fill="white", anchor="mm",
           stroke_width=4, stroke_fill=(0, 0, 0))
    return img


def tiled(img):
    a = np.asarray(img)
    fx, fy = a[:, ::-1], a[::-1]
    fxy = a[::-1, ::-1]
    row_m = np.hstack([fx, a, fx]); row_e = np.hstack([fxy, fy, fxy])
    return Image.fromarray(np.vstack([row_e, row_m, row_e]))


def xf(img, s=1.0, ang=0.0, dx=0.0, dy=0.0):
    """scale/rotate around center with mirrored edges"""
    t = tiled(img)
    ca, sa = math.cos(ang), math.sin(ang)
    a, b = ca / s, sa / s
    c, d_ = -sa / s, ca / s
    cx, cy = 1.5 * W + dx, 1.5 * H + dy
    coef = (a, b, cx - a * W / 2 - b * H / 2, c, d_, cy - c * W / 2 - d_ * H / 2)
    return t.transform((W, H), Image.AFFINE, coef, resample=Image.BILINEAR)


def zblur(img, s0, s1, ang0=0.0, ang1=0.0, n=6):
    acc = np.zeros((H, W, 3), float)
    for i in range(n):
        f = i / max(n - 1, 1)
        acc += np.asarray(xf(img, s0 + (s1 - s0) * f, ang0 + (ang1 - ang0) * f), float)
    return Image.fromarray((acc / n).astype(np.uint8))


def mix(a, b, f):
    return Image.blend(a, b, max(0.0, min(1.0, f)))


ein = lambda p: p ** 3
eout = lambda p: 1 - (1 - p) ** 3


# each transition: fn(A, B, p) with p in [0,1]
def t_zoom_in(A, B, p):
    if p < 0.5:
        q = ein(p * 2); return zblur(A, 1 + 1.2 * q, 1 + 1.6 * q)
    q = eout((p - 0.5) * 2); s = 2.2 - 1.2 * q
    return zblur(B, s, s + 0.4 * (1 - q))


def t_zoom_out(A, B, p):
    if p < 0.5:
        q = ein(p * 2); s = 1 - 0.55 * q
        return zblur(A, s, s + 0.15 * q)
    q = eout((p - 0.5) * 2); s = 1.8 - 0.8 * q
    return zblur(B, s, s - 0.2 * (1 - q))


def t_spin(A, B, p):
    if p < 0.5:
        q = ein(p * 2); return zblur(A, 1 + 0.6 * q, 1 + 0.8 * q, 0, math.pi * q * 0.9, 7)
    q = eout((p - 0.5) * 2); a = -math.pi * 0.9 * (1 - q)
    return zblur(B, 1.6 - 0.6 * q, 1.8 - 0.8 * q, a, a * 0.6, 7)


def t_flash(A, B, p):
    white = Image.new("RGB", (W, H), "white")
    if p < 0.4:
        q = p / 0.4; return mix(xf(A, 1 + 0.08 * q), white, q ** 2)
    q = (p - 0.4) / 0.6; return mix(white, xf(B, 1.08 - 0.08 * eout(q)), eout(q))


def t_glitch(A, B, p):
    rng = np.random.default_rng(int(p * 1000))
    src = np.asarray(A if p < 0.5 else B).copy()
    amp = 1 - abs(p - 0.5) * 2
    off = int(40 * amp)
    out = src.copy()
    out[..., 0] = np.roll(src[..., 0], off, 1)
    out[..., 2] = np.roll(src[..., 2], -off, 1)
    for _ in range(int(10 * amp) + 1):
        y0 = rng.integers(0, H - 60); h = rng.integers(8, 60)
        out[y0:y0 + h] = np.roll(out[y0:y0 + h], rng.integers(-80, 80), 1)
    noise = rng.integers(0, 60, (H, W, 1)) * amp
    return Image.fromarray(np.clip(out + noise, 0, 255).astype(np.uint8))


def t_pixel(A, B, p):
    src = A if p < 0.5 else B
    amp = 1 - abs(p - 0.5) * 2
    blk = max(1, int(2 + 60 * amp ** 1.5))
    small = src.resize((max(1, W // blk), max(1, H // blk)), Image.BILINEAR)
    return small.resize((W, H), Image.NEAREST)


def t_blur(A, B, p):
    amp = 1 - abs(p - 0.5) * 2
    r = 30 * amp
    a = A.filter(ImageFilter.GaussianBlur(r)); b = B.filter(ImageFilter.GaussianBlur(r))
    return mix(xf(a, 1 + 0.1 * p), xf(b, 1.1 - 0.1 * p), (p - 0.3) / 0.4)


def t_leak(A, B, p):
    base = mix(A, B, (p - 0.35) / 0.3)
    amp = 1 - abs(p - 0.5) * 2
    cx = -200 + (W + 400) * p
    g = np.exp(-(((xx - cx) / 260) ** 2 + ((yy - H * 0.4) / 520) ** 2))[..., None]
    col = np.array([255, 150, 40], float)
    arr = np.asarray(base, float)
    arr = arr + (col * g * 1.6 + 60) * amp
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def t_punch(A, B, p):
    if p < 0.35:
        q = p / 0.35; return xf(A, 1 + 0.12 * ein(q))
    q = (p - 0.35) / 0.65
    shake = (1 - q) ** 2
    rng = np.random.default_rng(int(p * 999))
    return xf(B, 1.2 - 0.2 * eout(q), 0.04 * shake * rng.uniform(-1, 1),
              30 * shake * rng.uniform(-1, 1), 30 * shake * rng.uniform(-1, 1))


TRANS = [
    ("Приближение", "Zoom In с размытием", t_zoom_in),
    ("Отдаление", "Zoom Out", t_zoom_out),
    ("Вращение + зум", "Spin Zoom", t_spin),
    ("Вспышка", "Flash", t_flash),
    ("Глитч", "Glitch / RGB-сдвиг", t_glitch),
    ("Пикселизация", "Pixelize", t_pixel),
    ("Размытие", "Blur Dissolve", t_blur),
    ("Световой засвет", "Light Leak", t_leak),
    ("Удар + тряска", "Punch Zoom / Shake", t_punch),
]


def caption(img, i, name, sub):
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([24, 40, W - 24, 160], 24, fill=(0, 0, 0, 150))
    d.text((W / 2, 82), f"{i}. {name}", font=F_T, fill="white", anchor="mm")
    d.text((W / 2, 128), sub, font=F_S, fill=(255, 220, 120), anchor="mm")
    return img


out = sys.argv[1]
ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                       "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                       "-movflags", "+faststart", out], stdin=subprocess.PIPE)
nh, nt = int(HOLD * FPS), int(TR * FPS)
T = 0.0
for i, (name, sub, fn) in enumerate(TRANS, 1):
    ka, kb = 2 * i, 2 * i + 1
    for f in range(nh):
        T += 1 / FPS; ff.stdin.write(caption(scene(ka, T, "КАДР 1"), i, name, sub).tobytes())
    for f in range(nt):
        T += 1 / FPS; p = (f + 1) / (nt + 1)
        fr = fn(scene(ka, T, "КАДР 1"), scene(kb, T, "КАДР 2"), p)
        ff.stdin.write(caption(fr, i, name, sub).tobytes())
    for f in range(nh):
        T += 1 / FPS; ff.stdin.write(caption(scene(kb, T, "КАДР 2"), i, name, sub).tobytes())
ff.stdin.close(); ff.wait()
print("done", out)
