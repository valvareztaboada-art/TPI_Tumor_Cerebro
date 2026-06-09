"""
Paso 1: Caracterizacion del dataset
Dataset: archive_prep/
  - notumor_prep/   : 19 imagenes axiales sin tumor
  - meningioma_prep/: 20 imagenes axiales con meningioma
  - glioma_prep/    : 16 imagenes axiales con glioma

Analisis:
  - Conteo y distribucion por clase
  - Estadisticas de tamano de imagen
  - Histograma de intensidades por clase
  - Galeria de ejemplos
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


BASE    = Path(__file__).parent / "archive_prep"
OUT_DIR = Path(__file__).parent / "resultados"
OUT_DIR.mkdir(exist_ok=True)

CLASSES = {
    "notumor_prep":    "Sin tumor",
    "meningioma_prep": "Meningioma",
    "glioma_prep":     "Glioma",
}
COLORS = {
    "notumor_prep":    "#2ecc71",
    "meningioma_prep": "#e74c3c",
    "glioma_prep":     "#9b59b6",
}

# umbral de fondo: pixeles por debajo se consideran fondo negro
BG_THRESH = 20

# ── 1. Conteo y estadisticas de tamano ────────────────────────────────────────

print("=" * 55)
print("CARACTERIZACION DEL DATASET")
print("=" * 55)

all_imgs = {}
for folder, label in CLASSES.items():
    paths = sorted((BASE / folder).glob("*"))
    all_imgs[folder] = paths
    if not paths:
        print(f"\n  {label} ({folder}): SIN IMAGENES")
        continue
    widths  = [Image.open(p).size[0] for p in paths]
    heights = [Image.open(p).size[1] for p in paths]
    print(f"\n  {label} ({folder})")
    print(f"    Imagenes : {len(paths)}")
    print(f"    Ancho    : min={min(widths)}  max={max(widths)}  media={np.mean(widths):.0f}")
    print(f"    Alto     : min={min(heights)} max={max(heights)} media={np.mean(heights):.0f}")

# ── 2. Estadisticas de intensidad ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("ESTADISTICAS DE INTENSIDAD (escala de grises 0-255)")
print("=" * 55)
print(f"  {'Clase':20s} {'Media':>7} {'Std':>7} {'Mediana':>8} {'P5':>6} {'P95':>6}")

hist_by_class  = {}
stats_by_class = {}

for folder, label in CLASSES.items():
    if not all_imgs[folder]:
        continue
    pixels = []
    accum  = np.zeros(256)
    for p in all_imgs[folder]:
        arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
        flat = arr.ravel()
        pixels.append(flat)
        h, _ = np.histogram(flat[flat > BG_THRESH], bins=256, range=(0, 255))
        accum += h
    accum /= accum.sum()
    hist_by_class[folder] = accum
    px = np.concatenate(pixels)
    stats_by_class[folder] = {
        "mean":   float(px.mean()),
        "std":    float(px.std()),
        "median": float(np.median(px)),
        "p5":     float(np.percentile(px, 5)),
        "p95":    float(np.percentile(px, 95)),
    }
    s = stats_by_class[folder]
    print(f"  {label:20s} {s['mean']:7.1f} {s['std']:7.1f} {s['median']:8.1f} "
          f"{s['p5']:6.1f} {s['p95']:6.1f}")

# ── Fig 1: distribucion por clase + estadisticas ──────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# barras de conteo
ax = axes[0]
labels_list = [CLASSES[f] for f in CLASSES if all_imgs[f]]
counts      = [len(all_imgs[f]) for f in CLASSES if all_imgs[f]]
colors_list = [COLORS[f] for f in CLASSES if all_imgs[f]]
bars = ax.bar(labels_list, counts, color=colors_list, edgecolor="white", width=0.5)
for bar, n in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(n), ha="center", fontsize=12, fontweight="bold")
ax.set_title("Cantidad de imagenes", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(counts) * 1.2)
ax.spines[["top", "right"]].set_visible(False)

# histogramas superpuestos (sin fondo)
ax = axes[1]
x = np.linspace(0, 255, 256)
for folder in CLASSES:
    if folder not in hist_by_class:
        continue
    ax.fill_between(x, hist_by_class[folder], alpha=0.35, color=COLORS[folder])
    ax.plot(x, hist_by_class[folder], color=COLORS[folder],
            linewidth=2, label=CLASSES[folder])
ax.set_title("Histogramas de intensidad\n(sin fondo)", fontsize=12, fontweight="bold")
ax.set_xlabel("Intensidad (0-255)")
ax.set_ylabel("Frecuencia normalizada")
ax.set_xlim(20, 255)
ax.set_ylim(0, 0.025)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# boxplot de intensidades
ax = axes[2]
pixel_data  = []
tick_labels = []
box_colors  = []
for folder in CLASSES:
    if not all_imgs[folder]:
        continue
    px = np.concatenate([
        np.array(Image.open(p).convert("L")).ravel()
        for p in all_imgs[folder]
    ])
    pixel_data.append(px[px > BG_THRESH])
    tick_labels.append(CLASSES[folder])
    box_colors.append(COLORS[folder] + "80")

bp = ax.boxplot(pixel_data, tick_labels=tick_labels, patch_artist=True, widths=0.4)
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
ax.set_title("Distribucion de intensidades\n(sin fondo)", fontsize=12, fontweight="bold")
ax.set_ylabel("Intensidad")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Caracterizacion del dataset - archive_prep",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_caracterizacion.png", dpi=150)
plt.close()
print("\nOK fig1_caracterizacion.png guardada")

# ── Fig 2: galeria de ejemplos ────────────────────────────────────────────────

# tomar hasta 15 imagenes por clase para que la galeria no sea enorme
N_SHOW = 15
clases_validas = [f for f in CLASSES if all_imgs[f]]
n_rows = len(clases_validas)
n_cols = N_SHOW

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 2.5))

for row, folder in enumerate(clases_validas):
    imgs = all_imgs[folder][:N_SHOW]
    for col in range(n_cols):
        ax = axes[row, col]
        if col < len(imgs):
            arr = np.array(Image.open(imgs[col]).convert("L"))
            ax.imshow(arr, cmap="gray")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(CLASSES[folder], fontsize=11, fontweight="bold",
                          rotation=0, labelpad=75, va="center")

fig.suptitle(f"Galeria del dataset (hasta {N_SHOW} imagenes por clase)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_galeria.png", dpi=150)
plt.close()
print("OK fig2_galeria.png guardada")

print(f"\nResultados en: {OUT_DIR}")