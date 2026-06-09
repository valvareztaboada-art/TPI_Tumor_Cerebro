"""
Paso 4: Regiones concentricas (galerias visuales)
Define las 3 zonas sobre la mascara del tumor y genera una galeria por clase.
El analisis comparativo de features entre clases lo hace el Paso 5.

Zonas (a partir de la mascara guardada por paso3_segmentacion.py):
  - Nucleo      : la mascara del tumor
  - Borde       : anillo exterior (dilatacion - nucleo)
  - Peritumoral : corona mas exterior (dilatacion mayor - dilatacion menor)

Las features y la comparacion entre clases se calculan en el Paso 5.

Input : archive_prep/<clase>/      (imagenes originales, para visualizar)
        archive_mascaras/<clase>/   (mascaras de paso3)
Output: fig7_regiones_<clase>.png en resultados/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage.morphology import binary_dilation, disk
import warnings
warnings.filterwarnings("ignore")

RAW_BASE   = Path(r"C:\Users\pauli\Downloads\TP imagenes tumores\archive_prep")
MASK_BASE  = Path(r"C:\Users\pauli\Downloads\TP imagenes tumores\archive_mascaras")
OUT_DIR    = Path(r"C:\Users\pauli\Downloads\TP imagenes tumores\resultados")
OUT_DIR.mkdir(exist_ok=True)

TARGET_SIZE = (224, 224)
RING_BORDER = 8    # pixeles de ancho del anillo de borde
RING_PERI   = 16   # pixeles de ancho de la corona peritumoral

# clases a procesar: carpeta -> etiqueta
CLASES = {
    "meningioma_prep": "Meningioma",
    "glioma_prep":     "Glioma",
}

def load_original(stem, cls):
    """Busca la imagen original, aplica crop del cerebro + resize 224x224.
    Sin N4 ni black point -> conserva intensidades reales del tejido."""
    for ext in [".jpg", ".jpeg", ".png"]:
        p = RAW_BASE / cls / (stem + ext)
        if p.exists():
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
            mask_bg = arr > 10
            rows = np.where(mask_bg.any(axis=1))[0]
            cols = np.where(mask_bg.any(axis=0))[0]
            if len(rows) > 0 and len(cols) > 0:
                r0 = max(0, rows[0]  - 5)
                r1 = min(arr.shape[0], rows[-1] + 6)
                c0 = max(0, cols[0]  - 5)
                c1 = min(arr.shape[1], cols[-1] + 6)
                arr = arr[r0:r1, c0:c1]
            arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize(TARGET_SIZE, Image.LANCZOS)) / 255.0
            return arr.astype(np.float32)
    return None

# ── Regiones concentricas ─────────────────────────────────────────────────────

def build_regions(mask):
    dil_inner   = binary_dilation(mask, disk(RING_BORDER)).astype(np.uint8)
    dil_outer   = binary_dilation(mask, disk(RING_BORDER + RING_PERI)).astype(np.uint8)
    nucleo      = mask
    borde       = (dil_inner - mask).clip(0, 1)
    peritumoral = (dil_outer - dil_inner).clip(0, 1)
    return nucleo, borde, peritumoral

# ── Galeria de regiones por clase ─────────────────────────────────────────────

def fig_galeria_regiones(galeria, label_cls, fname):
    n = len(galeria)
    if n == 0:
        print(f"  (sin mascaras validas para {label_cls}, no se genera figura)")
        return
    fig, axes = plt.subplots(n, 5, figsize=(17, 3.5 * n))
    if n == 1: axes = axes[np.newaxis, :]

    col_titles = ["Original", "Nucleo (rojo)", "Borde (amarillo)",
                  "Peritumoral (cian)", "Todas las zonas"]

    for row, r in enumerate(galeria):
        img, nuc, bor, per = r["img"], r["nucleo"], r["borde"], r["peri"]

        def ov(img, mask, rgb):
            out = np.stack([img, img, img], axis=-1)
            if mask is not None and mask.sum() > 0:
                for c, v in enumerate(rgb):
                    out[:,:,c] = np.where(mask==1, 0.4*img + 0.6*v, img)
            return np.clip(out, 0, 1)

        ov_nuc = ov(img, nuc, [1.0, 0.1, 0.1])
        ov_bor = ov(img, bor, [1.0, 1.0, 0.0])
        ov_per = ov(img, per, [0.0, 1.0, 1.0])

        comp = np.stack([img, img, img], axis=-1)
        comp[per==1, 0] = 0.0
        comp[per==1, 1] = np.clip(0.4*img[per==1] + 0.6, 0, 1)
        comp[per==1, 2] = np.clip(0.4*img[per==1] + 0.6, 0, 1)
        comp[bor==1, 0] = np.clip(0.4*img[bor==1] + 0.6, 0, 1)
        comp[bor==1, 1] = np.clip(0.4*img[bor==1] + 0.6, 0, 1)
        comp[bor==1, 2] = 0.0
        comp[nuc==1, 0] = np.clip(0.4*img[nuc==1] + 0.6, 0, 1)
        comp[nuc==1, 1] = 0.0
        comp[nuc==1, 2] = 0.0
        comp = np.clip(comp, 0, 1)

        for col, arr in enumerate([img, ov_nuc, ov_bor, ov_per, comp]):
            ax = axes[row, col]
            ax.imshow(arr, cmap="gray" if arr.ndim==2 else None)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel(r["stem"], fontsize=7, rotation=0,
                                 labelpad=80, va="center")

    fig.suptitle(f"Regiones concentricas - {label_cls}",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_DIR / fname, dpi=120)
    plt.close()
    print(f"OK {fname} guardada")

# ── Procesar una clase: armar las regiones de cada mascara valida ─────────────

def procesar_clase(cls_folder, label_cls):
    mask_dir = MASK_BASE / cls_folder
    paths = sorted(mask_dir.glob("*_mask.png"))
    print("=" * 60)
    print(f"REGIONES CONCENTRICAS - {cls_folder} ({len(paths)} mascaras)")
    print("=" * 60)

    galeria = []
    n_validas = n_vacias = 0

    for mask_path in paths:
        stem = mask_path.stem.replace("_mask", "")
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = (mask > 127).astype(np.uint8)
        if mask.sum() == 0:
            n_vacias += 1
            continue
        img = load_original(stem, cls_folder)
        if img is None:
            print(f"  {stem}: no se encontro original")
            continue
        nuc, bor, per = build_regions(mask)
        galeria.append({"img": img, "nucleo": nuc, "borde": bor, "peri": per, "stem": stem})
        n_validas += 1

    print(f"  Con mascara valida : {n_validas}/{len(paths)}")
    print(f"  Mascaras vacias    : {n_vacias}/{len(paths)}")
    return galeria

# ══════════════════════════════════════════════════════════════════════════════
# EJECUCION: una galeria por clase
# ══════════════════════════════════════════════════════════════════════════════

for cls_folder, label_cls in CLASES.items():
    galeria = procesar_clase(cls_folder, label_cls)
    fname   = f"fig7_regiones_{cls_folder.replace('_prep','')}.png"
    fig_galeria_regiones(galeria, label_cls, fname)
    print()

print(f"Resultados en: {OUT_DIR}")
