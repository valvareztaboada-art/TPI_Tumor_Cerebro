"""
Paso 5: Comparacion de features de textura entre clases de tumor
  - Carga imagenes preprocesadas + mascaras guardadas por 03_segmentacion.py
  - Extrae features de textura (GLCM) y forma sobre la region del tumor
  - Guarda CSV con todas las features
  - Genera figuras comparativas para analizar si la textura discrimina entre clases

Features de textura (GLCM a multiples distancias y angulos):
  - Contraste, homogeneidad, energia, correlacion, disimilaridad

Features de intensidad por zona (nucleo, borde, peritumoral):
  - Media, std, p25, p75

Features de forma:
  - Area relativa, solidez, excentricidad, irregularidad

Input : archive_prep_proc/          (imagenes preprocesadas)
        archive_mascaras/            (mascaras de 03_segmentacion.py)
Output: resultados/features_todas_clases.csv
        resultados/fig10_*.png ... fig13_*.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage.filters import sobel
from skimage.morphology import binary_dilation, disk
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings("ignore")

RAW_BASE   = Path(__file__).parent / "archive_prep"
MASK_BASE  = Path(__file__).parent / "archive_mascaras"
OUT_DIR    = Path(__file__).parent / "resultados"
OUT_DIR.mkdir(exist_ok=True)

TARGET_SIZE = (224, 224)

def load_original(stem, cls):
    """Busca la imagen original, aplica crop del cerebro + resize 224x224.
    Sin N4 ni black point -> conserva intensidades reales del tejido."""
    for ext in [".jpg", ".jpeg", ".png"]:
        p = RAW_BASE / cls / (stem + ext)
        if p.exists():
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
            # crop bounding box del cerebro
            mask_bg = arr > 10
            rows = np.where(mask_bg.any(axis=1))[0]
            cols = np.where(mask_bg.any(axis=0))[0]
            if len(rows) > 0 and len(cols) > 0:
                r0 = max(0, rows[0]  - 5)
                r1 = min(arr.shape[0], rows[-1] + 6)
                c0 = max(0, cols[0]  - 5)
                c1 = min(arr.shape[1], cols[-1] + 6)
                arr = arr[r0:r1, c0:c1]
            # resize a 224x224
            arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize(TARGET_SIZE, Image.LANCZOS)) / 255.0
            return arr.astype(np.float32)
    return None

RING_BORDER = 8
RING_PERI   = 16

TUMOR_CLASSES = ["meningioma_prep", "glioma_prep"]
NOTUMOR_CLASS = "notumor_prep"
COLORS = {"meningioma_prep": "#e74c3c", "glioma_prep": "#9b59b6"}
LABELS = {"meningioma_prep": "Meningioma", "glioma_prep": "Glioma"}

# ── Regiones concentricas ─────────────────────────────────────────────────────

def build_regions(mask):
    dil_inner   = binary_dilation(mask, disk(RING_BORDER)).astype(np.uint8)
    dil_outer   = binary_dilation(mask, disk(RING_BORDER + RING_PERI)).astype(np.uint8)
    borde       = (dil_inner - mask).clip(0, 1)
    peritumoral = (dil_outer - dil_inner).clip(0, 1)
    return mask, borde, peritumoral

# ── Features de textura GLCM ─────────────────────────────────────────────────
#
# La GLCM (Gray-Level Co-occurrence Matrix) cuenta con que frecuencia aparecen
# pares de intensidades a una distancia y angulo dados.
# Se calcula a 4 angulos (0, 45, 90, 135) y 3 distancias (1, 2, 3 px).
# De la matriz se derivan 5 descriptores:
#   - contrast    : variacion de intensidad entre pixeles vecinos (alto = textura rugosa)
#   - homogeneity : que tan similares son los pares vecinos (alto = textura uniforme)
#   - energy      : uniformidad de la distribucion (alto = patron repetitivo)
#   - correlation : dependencia lineal entre pares (alto = patron direccional)
#   - dissimilarity: diferencia absoluta media entre pares

def glcm_features(img, mask, prefix):
    """Extrae features GLCM sobre el bounding box de la region.
    Se recorta el bbox para evitar que los ceros fuera de la mascara dominen la matriz."""
    px = img[mask == 1]
    if len(px) < 16:
        return {f"{prefix}_glcm_{p}_{d}": 0
                for p in ["contrast","homogeneity","energy","correlation","dissimilarity"]
                for d in [1, 2, 3]}

    # recortar bounding box
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    img8 = (img[r0:r1, c0:c1] * 255).astype(np.uint8)
    roi  = img8.copy()
    roi[mask[r0:r1, c0:c1] == 0] = 0

    feats = {}
    distances = [1, 2, 3]
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props     = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]

    glcm = graycomatrix(roi, distances=distances, angles=angles,
                         levels=256, symmetric=True, normed=True)

    for prop in props:
        values = graycoprops(glcm, prop)  # shape: (len(distances), len(angles))
        for di, d in enumerate(distances):
            # promedio sobre los 4 angulos -> invariante a rotacion
            feats[f"{prefix}_glcm_{prop}_{d}"] = float(values[di, :].mean())

    return feats

# ── Features de intensidad ────────────────────────────────────────────────────

def intensity_features(img, mask, prefix):
    px = img[mask == 1]
    if len(px) == 0:
        return {f"{prefix}_mean": 0, f"{prefix}_std": 0,
                f"{prefix}_p25": 0, f"{prefix}_p75": 0}
    return {
        f"{prefix}_mean": float(np.mean(px)),
        f"{prefix}_std":  float(np.std(px)),
        f"{prefix}_p25":  float(np.percentile(px, 25)),
        f"{prefix}_p75":  float(np.percentile(px, 75)),
    }

# ── Features de gradiente ─────────────────────────────────────────────────────

def gradient_features(img, mask, prefix):
    px = sobel(img)[mask == 1]
    return {f"{prefix}_grad_mean": float(np.mean(px)) if len(px) > 0 else 0}

# ── Features de forma ─────────────────────────────────────────────────────────

def shape_features(img, mask):
    brain_px = (img > 0.05).sum()
    regions  = regionprops(label(mask))
    if not regions:
        return {"shape_area_ratio": 0, "shape_perimeter": 0,
                "shape_solidity": 0, "shape_eccentricity": 0,
                "shape_irregularidad": 0}
    reg  = max(regions, key=lambda r: r.area)
    peri = reg.perimeter
    area = reg.area
    return {
        "shape_area_ratio":    float(mask.sum() / brain_px) if brain_px > 0 else 0,
        "shape_perimeter":     float(peri),
        "shape_solidity":      float(reg.solidity),
        "shape_eccentricity":  float(reg.eccentricity),
        "shape_irregularidad": float(peri**2 / (4*np.pi*area)) if area > 0 else 0,
    }

# ── Perfil radial ─────────────────────────────────────────────────────────────

def radial_profile(img, mask, n_bins=10):
    regions = regionprops(label(mask))
    if not regions:
        return {f"radial_{i}": 0 for i in range(n_bins)}
    reg    = max(regions, key=lambda r: r.area)
    cy, cx = reg.centroid
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    dist   = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    max_r  = dist[mask == 1].max() * 2.0 if mask.sum() > 0 else 1.0
    bins   = np.linspace(0, max_r, n_bins + 1)
    result = {}
    for i in range(n_bins):
        ring = (dist >= bins[i]) & (dist < bins[i+1]) & (img > 0.05)
        result[f"radial_{i}"] = float(img[ring].mean()) if ring.sum() > 0 else 0
    return result

# ── Extraer todas las features ────────────────────────────────────────────────

def extract_all_features(img, mask):
    nucleo, borde, peri = build_regions(mask)
    feats = {}
    for zona, region in [("nucleo", nucleo), ("borde", borde), ("peri", peri)]:
        feats.update(glcm_features(img, region, zona))
        feats.update(intensity_features(img, region, zona))
        feats.update(gradient_features(img, region, zona))
    feats.update(shape_features(img, mask))
    feats.update(radial_profile(img, mask, n_bins=10))
    return feats

# ── Procesar clases ───────────────────────────────────────────────────────────

all_rows = []
resumen  = {}

print("=" * 60)
print("COMPARACION DE CLASES - extraccion de features de textura")
print("=" * 60)

for cls in TUMOR_CLASSES:
    mask_dir = MASK_BASE / cls
    paths    = sorted(mask_dir.glob("*_mask.png"))
    n_ok = n_vacio = n_sinimg = 0

    print(f"\n{LABELS[cls]} ({len(paths)} mascaras)...")

    for mask_path in paths:
        stem = mask_path.stem.replace("_mask", "")

        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = (mask > 127).astype(np.uint8)

        if mask.sum() == 0:
            n_vacio += 1
            continue

        # cargar imagen ORIGINAL redimensionada a 224x224
        img = load_original(stem, cls)
        if img is None:
            n_sinimg += 1
            print(f"  {stem}: no encontrada en archive_prep/")
            continue

        feats = extract_all_features(img, mask)
        feats["clase"]  = cls
        feats["imagen"] = stem
        all_rows.append(feats)
        n_ok += 1

    resumen[cls] = {"validas": n_ok, "vacias": n_vacio, "sin_img": n_sinimg, "total": len(paths)}
    print(f"  Procesadas: {n_ok}/{len(paths)}  |  vacias: {n_vacio}  |  sin imagen original: {n_sinimg}")

# ── Verificar falsos positivos en notumor ─────────────────────────────────────
print(f"\nNotumor - contando mascaras no vacias (falsos positivos)...")
paths_nt = sorted((MASK_BASE / NOTUMOR_CLASS).glob("*_mask.png"))
fp = 0
for mp in paths_nt:
    m = np.array(Image.open(mp), dtype=np.uint8)
    if (m > 127).sum() > 0:
        fp += 1
resumen[NOTUMOR_CLASS] = {"falsos_positivos": fp, "total": len(paths_nt)}
print(f"  Falsos positivos: {fp}/{len(paths_nt)}")

# ── Guardar CSV ───────────────────────────────────────────────────────────────
df   = pd.DataFrame(all_rows)
cols = ["clase", "imagen"] + [c for c in df.columns if c not in ["clase","imagen"]]
df   = df[cols]
df.to_csv(OUT_DIR / "features_todas_clases.csv", index=False)
print(f"\n  CSV guardado: features_todas_clases.csv  ({len(df)} filas x {len(df.columns)} columnas)")

# ── Fig 10: GLCM por zona y clase ─────────────────────────────────────────────
print("\nGenerando figuras...")

# features GLCM a distancia 1 (las mas representativas)
glcm_props = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]
zonas      = ["nucleo", "borde", "peri"]
zona_names = ["Nucleo", "Borde", "Peritumoral"]

fig, axes = plt.subplots(len(glcm_props), len(zonas),
                          figsize=(14, 3.5 * len(glcm_props)))

for row, prop in enumerate(glcm_props):
    for col, (zona, zname) in enumerate(zip(zonas, zona_names)):
        ax    = axes[row, col]
        feat  = f"{zona}_glcm_{prop}_1"
        data  = [df[df["clase"]==cls][feat].dropna().tolist() for cls in TUMOR_CLASSES]
        labels = [LABELS[c] for c in TUMOR_CLASSES]
        if any(data):
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.4)
            for patch, cls in zip(bp["boxes"], TUMOR_CLASSES):
                patch.set_facecolor(COLORS[cls] + "80")
        if row == 0:  ax.set_title(zname, fontsize=10, fontweight="bold")
        if col == 0:  ax.set_ylabel(prop.capitalize(), fontsize=9, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

fig.suptitle("Features GLCM por zona y clase (distancia=1, promedio 4 angulos)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "fig10_glcm_por_zona.png", dpi=130)
plt.close()
print("OK fig10_glcm_por_zona.png guardada")

# ── Fig 11: GLCM a distintas distancias (nucleo) ──────────────────────────────
fig, axes = plt.subplots(1, len(glcm_props), figsize=(18, 4))

for col, prop in enumerate(glcm_props):
    ax = axes[col]
    for cls in TUMOR_CLASSES:
        df_cls = df[df["clase"] == cls]
        means  = [df_cls[f"nucleo_glcm_{prop}_{d}"].mean() for d in [1, 2, 3]]
        stds   = [df_cls[f"nucleo_glcm_{prop}_{d}"].std()  for d in [1, 2, 3]]
        ax.errorbar([1, 2, 3], means, yerr=stds,
                    label=LABELS[cls], color=COLORS[cls],
                    marker="o", linewidth=2, capsize=4)
    ax.set_title(prop.capitalize(), fontsize=10, fontweight="bold")
    ax.set_xlabel("Distancia (px)", fontsize=9)
    ax.set_xticks([1, 2, 3])
    if col == 0:
        ax.set_ylabel("Valor medio ± std", fontsize=9)
        ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("GLCM del nucleo tumoral a distintas distancias: Meningioma vs Glioma",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "fig11_glcm_distancias.png", dpi=150)
plt.close()
print("OK fig11_glcm_distancias.png guardada")

# ── Fig 12: features discriminativas (intensidad + forma + gradiente) ─────────
feat_disc = [
    ("nucleo_mean",          "Intensidad media\nnucleo"),
    ("borde_mean",           "Intensidad media\nborde"),
    ("peri_mean",            "Intensidad media\nperitumoral"),
    ("nucleo_grad_mean",     "Gradiente medio\nnucleo"),
    ("borde_grad_mean",      "Gradiente medio\nborde"),
    ("shape_irregularidad",  "Irregularidad"),
    ("shape_solidity",       "Solidez"),
    ("shape_eccentricity",   "Excentricidad"),
    ("shape_area_ratio",     "Area relativa"),
]

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes = axes.flatten()

for i, (feat, titulo) in enumerate(feat_disc):
    ax = axes[i]
    data   = [df[df["clase"]==cls][feat].dropna().tolist() for cls in TUMOR_CLASSES]
    labels = [LABELS[c] for c in TUMOR_CLASSES]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, cls in zip(bp["boxes"], TUMOR_CLASSES):
        patch.set_facecolor(COLORS[cls] + "80")
    ax.set_title(titulo, fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Features discriminativas: Meningioma vs Glioma",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "fig12_features_discriminativas.png", dpi=130)
plt.close()
print("OK fig12_features_discriminativas.png guardada")

# ── Fig 13: perfil radial por clase ───────────────────────────────────────────
radial_cols = [f"radial_{i}" for i in range(10)]
x = np.arange(10)

fig, ax = plt.subplots(figsize=(9, 4))
for cls in TUMOR_CLASSES:
    vals   = df[df["clase"]==cls][radial_cols].values
    mean_r = vals.mean(axis=0)
    std_r  = vals.std(axis=0)
    ax.plot(x, mean_r, color=COLORS[cls], linewidth=2.5, label=LABELS[cls])
    ax.fill_between(x, mean_r - std_r, mean_r + std_r,
                    color=COLORS[cls], alpha=0.15)

ax.set_xlabel("Anillo (centro → periferia)", fontsize=10)
ax.set_ylabel("Intensidad media", fontsize=10)
ax.set_title("Perfil radial de intensidad por clase", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"r{i}" for i in x], fontsize=8)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig13_perfil_radial_clases.png", dpi=150)
plt.close()
print("OK fig13_perfil_radial_clases.png guardada")

# ── Resumen estadistico en consola ────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN DE FEATURES POR CLASE")
print("=" * 60)

feat_resumen = [
    "nucleo_glcm_contrast_1", "nucleo_glcm_homogeneity_1",
    "nucleo_glcm_energy_1",   "nucleo_glcm_correlation_1",
    "nucleo_mean", "nucleo_std",
    "shape_irregularidad", "shape_solidity",
]

print(f"\n  {'Feature':35s}", end="")
for cls in TUMOR_CLASSES:
    print(f"  {LABELS[cls]:>14}", end="")
print()
print("  " + "-" * (35 + 16 * len(TUMOR_CLASSES)))

for feat in feat_resumen:
    if feat not in df.columns:
        continue
    print(f"  {feat:35s}", end="")
    for cls in TUMOR_CLASSES:
        vals = df[df["clase"]==cls][feat].dropna()
        print(f"  {vals.mean():6.4f}±{vals.std():5.4f}", end="")
    print()

print("\n" + "=" * 60)
print("FALSOS POSITIVOS")
print("=" * 60)
fp_info = resumen.get(NOTUMOR_CLASS, {})
print(f"  Notumor: {fp_info.get('falsos_positivos', '?')}/{fp_info.get('total', '?')} imagenes detectadas como tumor")

print(f"\nResultados en: {OUT_DIR}")
