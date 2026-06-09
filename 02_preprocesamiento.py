"""
Paso 2: Preprocesamiento
Pipeline por imagen:
  1. Escala de grises
  2. Suavizado gaussiano leve (reduce ruido de alta frecuencia)
  3. Recortar fondo negro (crop bounding box del cerebro)
  4. Normalizar a [0, 1]
  5. Resize a 224x224
  6. N4 Bias Field Correction (corrige inhomogeneidad de campo magnetico)
  7. Ajuste de tono (exposicion + sombras)
  8. Black point adjustment (oscurece tejido gris medio, resalta tumor)
  9. Guardar como PNG en archive_prep_proc/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage.filters import gaussian
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")

BASE      = Path(__file__).parent / "archive_prep"
OUT_BASE  = Path(__file__).parent / "archive_prep_proc"
OUT_DIR   = Path(__file__).parent / "resultados"
OUT_DIR.mkdir(exist_ok=True)

CLASSES     = ["notumor_prep", "meningioma_prep", "glioma_prep"]
TARGET_SIZE = (224, 224)
BG_THRESH   = 10
CROP_MARGIN = 5
GAUSS_SIGMA = 0.8
BLACK_POINT = 0.2

# umbral para excluir fondo en los histogramas de la fig4
HIST_BG = 20

# ── Ajustes de tono ────────────────────────────────────────────────────────────
# exposure: positivo = mas brillante (ej: 0.8), negativo = mas oscuro
# shadows : positivo = sube los grises oscuros, negativo = los baja mas a negro
EXPOSURE = 0.6
SHADOWS  = -0.8

COLORS = {"notumor_prep": "#2ecc71", "meningioma_prep": "#e74c3c", "glioma_prep": "#9b59b6"}

# ── Funciones del pipeline ─────────────────────────────────────────────────────

def crop_brain(arr: np.ndarray) -> np.ndarray:
    mask = arr > BG_THRESH
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return arr
    r0 = max(0, rows[0]  - CROP_MARGIN)
    r1 = min(arr.shape[0], rows[-1] + CROP_MARGIN + 1)
    c0 = max(0, cols[0]  - CROP_MARGIN)
    c1 = min(arr.shape[1], cols[-1] + CROP_MARGIN + 1)
    return arr[r0:r1, c0:c1]

def n4_correction(img: np.ndarray) -> np.ndarray:
    """Corrige inhomogeneidad de campo magnetico. Input/output float32 [0,1]."""
    arr = (img * 255.0).astype(np.float32)
    sitk_img  = sitk.GetImageFromArray(arr)
    sitk_mask = sitk.Cast(
        sitk.GetImageFromArray((arr > arr.max() * 0.05).astype(np.uint8)),
        sitk.sitkUInt8
    )
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([20, 10])
    corrected = sitk.GetArrayFromImage(corrector.Execute(sitk_img, sitk_mask))
    mn, mx = corrected.min(), corrected.max()
    return ((corrected - mn) / (mx - mn)).astype(np.float32) if mx > mn else corrected

def apply_tono(img: np.ndarray) -> np.ndarray:
    """Aplica exposicion y ajuste de sombras."""
    out = img.copy()
    # exposicion: sube o baja el brillo general
    if EXPOSURE != 0.0:
        out = np.clip(out * (2 ** EXPOSURE), 0, 1)
    # sombras: afecta solo los pixeles oscuros (<= 0.5)
    if SHADOWS != 0.0:
        mask = out <= 0.5
        out[mask] = np.clip(out[mask] + SHADOWS * (0.5 - out[mask]), 0, 1)
    return out.astype(np.float32)

def apply_black_point(img: np.ndarray) -> np.ndarray:
    """Estira el rango [BLACK_POINT, 1] a [0, 1]. Oscurece el tejido gris medio."""
    return np.clip((img - BLACK_POINT) / (1.0 - BLACK_POINT), 0, 1).astype(np.float32)

def preprocess(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    arr = gaussian(arr, sigma=GAUSS_SIGMA, preserve_range=True)
    arr = crop_brain(arr)
    mn, mx = arr.min(), arr.max()
    arr = (arr - mn) / (mx - mn) if mx > mn else arr
    arr = np.array(
        Image.fromarray((arr * 255).astype(np.uint8)).resize(TARGET_SIZE, Image.LANCZOS)
    ) / 255.0
    arr = n4_correction(arr)
    arr = apply_tono(arr)
    arr = apply_black_point(arr)
    return arr

# ── Procesar ───────────────────────────────────────────────────────────────────

print("Preprocesando dataset...")
print(f"  Suavizado gaussiano : sigma={GAUSS_SIGMA}")
print(f"  Resize              : {TARGET_SIZE}")
print(f"  N4 Bias Field       : iteraciones=[20, 10]")
print(f"  Exposicion          : {EXPOSURE}")
print(f"  Sombras             : {SHADOWS}")
print(f"  Black point         : {BLACK_POINT}\n")

for cls in CLASSES:
    src = BASE    / cls
    dst = OUT_BASE / cls
    dst.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob("*"))
    for p in paths:
        result = preprocess(p)
        Image.fromarray((result * 255).astype(np.uint8)).save(dst / (p.stem + ".png"))
    print(f"  {cls}: {len(paths)} imagenes procesadas -> {dst}")

# ── Figura: pipeline paso a paso ──────────────────────────────────────────────
print("\nGenerando figura...")

col_titles = ["Original", "Gris + suavizado", "Crop + norm", "224x224", "N4", "Tono", "Black point"]
fig, axes = plt.subplots(len(CLASSES), len(col_titles), figsize=(21, 4.5 * len(CLASSES)))

for row, cls in enumerate(CLASSES):
    p          = sorted((BASE / cls).glob("*"))[0]
    arr_orig   = np.array(Image.open(p).convert("L"), dtype=np.float32)
    arr_smooth = gaussian(arr_orig, sigma=GAUSS_SIGMA, preserve_range=True)
    arr_crop   = crop_brain(arr_smooth)
    mn, mx     = arr_crop.min(), arr_crop.max()
    arr_norm   = (arr_crop - mn) / (mx - mn) if mx > mn else arr_crop
    arr_resize = np.array(
        Image.fromarray((arr_norm * 255).astype(np.uint8)).resize(TARGET_SIZE, Image.LANCZOS)
    ) / 255.0
    arr_n4     = n4_correction(arr_resize)
    arr_tono   = apply_tono(arr_n4)
    arr_bp     = apply_black_point(arr_tono)

    steps = [arr_orig, arr_smooth, arr_norm, arr_resize, arr_n4, arr_tono, arr_bp]
    for col, (arr, title) in enumerate(zip(steps, col_titles)):
        ax = axes[row, col]
        ax.imshow(arr, cmap="gray")
        if row == 0: ax.set_title(title, fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel(cls.replace("_prep","").capitalize(),
                          fontsize=12, fontweight="bold",
                          rotation=0, labelpad=75, va="center")
        ax.axis("off")

fig.suptitle("Pipeline de preprocesamiento", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_preprocesamiento.png", dpi=150)
plt.close()
print("OK fig3_preprocesamiento.png guardada")

# ── Figura: histogramas antes vs despues (sin fondo) ──────────────────────────
from scipy.ndimage import gaussian_filter1d

fig, axes = plt.subplots(len(CLASSES), 2, figsize=(12, 4 * len(CLASSES)))

for row, cls in enumerate(CLASSES):
    hist_antes = np.zeros(256)
    hist_desp  = np.zeros(256)
    for p in sorted((BASE / cls).glob("*")):
        arr = np.array(Image.open(p).convert("L"))
        h, _ = np.histogram(arr[arr > HIST_BG], bins=256, range=(0, 255))
        hist_antes += h
    for p in sorted((OUT_BASE / cls).glob("*.png")):
        arr = np.array(Image.open(p))
        h, _ = np.histogram(arr[arr > HIST_BG], bins=256, range=(0, 255))
        hist_desp += h
    hist_antes /= hist_antes.sum()
    hist_desp  /= hist_desp.sum()
    # suavizado para eliminar el serrucho de cuantizacion
    hist_antes = gaussian_filter1d(hist_antes, sigma=1.5)
    hist_desp  = gaussian_filter1d(hist_desp, sigma=1.5)
    x = np.linspace(0, 255, 256)
    for col, (hist, lbl) in enumerate([(hist_antes, "Antes"), (hist_desp, "Despues")]):
        ax = axes[row, col]
        ax.fill_between(x, hist, alpha=0.35, color=COLORS[cls])
        ax.plot(x, hist, color=COLORS[cls], linewidth=1.5)
        if row == 0: ax.set_title(lbl, fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel(cls.replace("_prep","").capitalize(),
                          fontsize=11, fontweight="bold",
                          rotation=0, labelpad=75, va="center")
        ax.set_xlabel("Intensidad")
        ax.set_xlim(HIST_BG, 255)
        ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Histogramas antes vs despues del preprocesamiento (sin fondo)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "fig4_histogramas_prep.png", dpi=150)
plt.close()
print("OK fig4_histogramas_prep.png guardada")

print(f"\nImagenes procesadas en: {OUT_BASE}")