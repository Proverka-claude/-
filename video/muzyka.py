"""Instrumental track ~20s, 124 BPM, A minor, balalaika-style tremolo pluck lead."""
import sys, subprocess
import numpy as np
import imageio_ffmpeg

SR = 44100
BPM = 124
BEAT = 60 / BPM
BAR = 4 * BEAT
NBARS = 10
DUR = NBARS * BAR + 1.2
N = int(DUR * SR)
rng = np.random.default_rng(7)
L = np.zeros(N); R = np.zeros(N)


def midi(n): return 440 * 2 ** ((n - 69) / 12)


def add(buf, sig, t, gain=1.0):
    i = int(t * SR); j = min(N, i + len(sig))
    if i < N: buf[i:j] += sig[:j - i] * gain


def addst(sig, t, g=1.0, pan=0.0):
    add(L, sig, t, g * (1 - max(0, pan))); add(R, sig, t, g * (1 + min(0, pan)))


def env(n, a, d, s, r, length):
    tot = int(length * SR); e = np.ones(tot) * s
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na); e[na:na + nd] = np.linspace(1, s, nd)[:len(e[na:na + nd])]
    if nr: e[-nr:] *= np.linspace(1, 0, nr)
    return e


def lowpass(x, fc):
    a = np.exp(-2 * np.pi * fc / SR); y = np.zeros_like(x); p = 0.0
    for i in range(len(x)):
        p = (1 - a) * x[i] + a * p; y[i] = p
    return y


def ks(freq, length, bright=0.5, decay=0.996):
    """Karplus-Strong plucked string"""
    n = int(SR / freq); buf = rng.uniform(-1, 1, n)
    out = np.zeros(int(length * SR))
    for i in range(len(out)):
        out[i] = buf[i % n]
        nxt = buf[(i + 1) % n]
        buf[i % n] = decay * (bright * buf[i % n] + (1 - bright) * nxt) if i % n else decay * 0.5 * (buf[i % n] + nxt)
    return out


# --- sounds ---
def kick():
    t = np.arange(int(0.35 * SR)) / SR
    f = 50 + 110 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * 1.0


def clap():
    t = np.arange(int(0.25 * SR)) / SR
    nz = rng.uniform(-1, 1, len(t))
    e = np.exp(-t * 25) + 0.6 * np.exp(-((t - 0.012) * 400) ** 2)
    return np.diff(np.concatenate([[0], nz]))[:len(t)] * e * 0.6


def hat(open_=False):
    t = np.arange(int((0.2 if open_ else 0.06) * SR)) / SR
    nz = np.diff(np.concatenate([[0], rng.uniform(-1, 1, len(t))]))
    return nz * np.exp(-t * (15 if open_ else 70)) * 0.35


def saw_pad(notes, length):
    t = np.arange(int(length * SR)) / SR; s = np.zeros(len(t))
    for n in notes:
        for det in (-0.12, 0, 0.12):
            f = midi(n + det)
            s += 2 * ((t * f + rng.random()) % 1) - 1
    s /= len(notes) * 3
    return lowpass(s, 2200) * env(None, 0.25, 0.3, 0.8, 0.4, length)


def bass(n, length):
    t = np.arange(int(length * SR)) / SR; f = midi(n)
    s = np.sin(2 * np.pi * f * t) + 0.35 * np.sign(np.sin(2 * np.pi * f * t))
    return s * env(None, 0.005, 0.1, 0.7, 0.05, length) * 0.45


def balalaika(n, length):
    """tremolo strum: fast repeated plucks"""
    out = np.zeros(int(length * SR)); step = BEAT / 4
    k = 0
    while k * step < length - 0.02:
        p = ks(midi(n), min(0.35, length - k * step), bright=0.6, decay=0.994)
        p2 = ks(midi(n + 12) * 1.002, len(p) / SR, bright=0.7, decay=0.993) * 0.35
        i = int(k * step * SR); j = min(len(out), i + len(p))
        out[i:j] += (p + p2)[:j - i] * (0.85 if k % 2 else 1.0)
        k += 1
    return out * env(None, 0.002, 0.05, 0.9, 0.08, length)[:len(out)]


def riser(length):
    t = np.arange(int(length * SR)) / SR
    nz = rng.uniform(-1, 1, len(t)); x = t / length
    y = np.zeros_like(nz); p = 0.0
    for i in range(len(nz)):
        a = np.exp(-2 * np.pi * (300 + 7000 * x[i] ** 2) / SR); p = (1 - a) * nz[i] + a * p; y[i] = p
    return y * x ** 2 * 1.2


def impact():
    t = np.arange(int(1.5 * SR)) / SR
    boom = np.sin(2 * np.pi * (40 + 60 * np.exp(-t * 8)) * t) * np.exp(-t * 3)
    nz = lowpass(rng.uniform(-1, 1, len(t)), 3000) * np.exp(-t * 4) * 0.8
    return boom + nz


# --- arrangement: Am F C G | Am F C E ---
CH = [[57, 60, 64], [53, 57, 60], [48, 55, 60, 64], [55, 59, 62],
      [57, 60, 64], [53, 57, 60], [48, 55, 60, 64], [52, 56, 59]]
ROOT = [45, 41, 48, 43, 45, 41, 48, 40]
# lead melody (A harmonic minor), per bar: list of (beat, note, len_beats)
MEL = [
    [(0, 76, 1), (1, 74, .5), (1.5, 72, .5), (2, 71, 1), (3, 72, 1)],
    [(0, 69, 1.5), (1.5, 72, .5), (2, 74, 2)],
    [(0, 76, 1), (1, 79, 1), (2, 76, .5), (2.5, 74, .5), (3, 72, 1)],
    [(0, 74, 2), (2, 71, 1), (3, 74, 1)],
    [(0, 81, 1), (1, 79, .5), (1.5, 77, .5), (2, 76, 1), (3, 74, 1)],
    [(0, 72, 1), (1, 74, 1), (2, 76, 2)],
    [(0, 79, 1), (1, 76, .5), (1.5, 72, .5), (2, 74, 1), (3, 76, 1)],
    [(0, 75, 1), (1, 76, 1), (2, 80, 2)],
]

# bar 0-1: intro (pad + filtered lead), riser into bar 2 drop
for b in range(NBARS):
    t0 = b * BAR; c = b % 8
    last = b == NBARS - 1
    ln = BAR * (1.6 if last else 1)
    addst(saw_pad(CH[c], ln), t0, 0.28 if b >= 2 else 0.2)
    for bt, n, lb in MEL[c]:
        g = 0.18 if b >= 2 else 0.11
        addst(balalaika(n, lb * BEAT), t0 + bt * BEAT, g, pan=0.2)
    if b >= 2 and not last:
        for k in range(4):
            addst(kick(), t0 + k * BEAT, 0.9)
            addst(hat(), t0 + k * BEAT + BEAT / 2, 0.5, pan=-0.3)
        for k in (1, 3):
            addst(clap(), t0 + k * BEAT, 0.8)
        for k in range(8):
            addst(bass(ROOT[c] if k % 2 == 0 else ROOT[c] + 12, BEAT / 2 * 0.9), t0 + k * BEAT / 2, 0.8)
        if b == 5:  # build fill
            for k in range(8): addst(clap(), t0 + 2 * BEAT + k * BEAT / 4, 0.25 + 0.05 * k)
    if last:
        addst(kick(), t0, 0.9); addst(bass(ROOT[c], BAR), t0, 0.8)

addst(riser(2 * BAR), 0, 0.35)
addst(impact(), 2 * BAR, 0.7)
addst(impact(), 6 * BAR, 0.45)

mix = np.stack([L, R], 1)
# gentle fade out last 1.5s
fo = int(1.5 * SR); mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
mix = np.tanh(mix / np.abs(mix).max() * 1.6) * 0.9
out = sys.argv[1]
p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "f32le", "-ar", str(SR),
                      "-ac", "2", "-i", "-", "-af", "loudnorm=I=-14:TP=-1", "-ar", "44100", "-c:a", "aac", "-b:a", "192k", out],
                     stdin=subprocess.PIPE)
p.stdin.write(mix.astype(np.float32).tobytes()); p.stdin.close(); p.wait()
print("done", DUR)
