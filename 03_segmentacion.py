"""
Paso 3: Segmentacion del tumor - K-means vs Multi-Otsu
Clases procesadas: meningioma_prep y glioma_prep.

Pipeline por imagen:
  1. Cargar imagen preprocesada (ya tiene N4 + black point aplicados)
  2. Restar craneo (Hysteresis Thresholding) antes de segmentar
  3. Segmentacion con K-means (k=5) y Multi-Otsu (N=4)
  4. Refinamiento con contornos activos (Morphological Geodesic Active Contour)
  5. Filtros de validacion:
       a. area_ratio entre 0.005 y 0.70
       b. contraste >= 0.10
       c. no toca el borde de la imagen
       d. solapamiento con craneo < 40%
       e. irregularidad <= 4.0
       f. solidez >= 0.70
  6. Mascara final: Multi-Otsu (principal), K-means (respaldo si MO falla)

Input : archive_prep_proc/meningioma_prep/  y  archive_prep_proc/glioma_prep/
Output: mascaras en archive_mascaras/, figuras en resultados/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage.filters import apply_hysteresis_threshold, threshold_multiotsu
from skimage.morphology import (remove_small_objects, remove_small_holes,
                                 binary_closing, binary_opening, disk)
from skimage.measure import label, regionprops
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

PROC_BASE  = Path(r"C:\Users\pauli\Downloads\TP imagenes tumores\archive_prep_proc")
OUT_DIR    = Path(r"C:\Users\pauli\Downloads\TP imagenes tumores\resultados")
MASK_BASE  = Path(r"C:\Users\pauli\Downloads\TP imagenes tumores\archive_mascaras")
OUT_DIR.mkdir(exist_ok=True)

AR_MIN, AR_MAX = 0.005, 0.60
CON_MIN        = 0.10
SKULL_OVERLAP  = 0.40
IRREG_MAX      = 4.0
SOLID_MIN      = 0.70
SNAKE_ITERS    = 60      # reducido de 200 para mayor velocidad
MAX_GALERIA    = 8       # max imagenes en galeria para no trabar matplotlib

# ── Deteccion de craneo ────────────────────────────────────────────────────────

def detect_skull(img: np.ndarray) -> np.ndarray:
    skull_raw = apply_hysteresis_threshold(img, low=0.50, high=0.75)
    h, w   = img.shape
    border = np.ones_like(img, dtype=bool)
    border[int(h*0.20):-int(h*0.20), int(w*0.20):-int(w*0.20)] = False
    return (skull_raw & border).astype(np.uint8)

# ── Utilidades ─────────────────────────────────────────────────────────────────

def keep_largest(mask):
    labeled = label(mask)
    if labeled.max() == 0: return mask
    regions = regionprops(labeled)
    return (labeled == max(regions, key=lambda r: r.area).label).astype(np.uint8)

def clean_mask(mask):
    mask = binary_opening(mask, disk(3))
    mask = binary_closing(mask, disk(5))
    mask = remove_small_objects(mask, min_size=100)
    mask = remove_small_holes(mask, area_threshold=500)
    return keep_largest(mask)

def is_valid(img, mask, skull):
    if mask.sum() == 0: return False, "vacia"
    brain = img > 0.05
    if brain.sum() == 0: return False, "sin_cerebro"
    ar = mask.sum() / brain.sum()
    if not (AR_MIN <= ar <= AR_MAX): return False, f"area={ar:.3f}"
    mean_in  = img[mask==1].mean()
    mean_out = img[(mask==0)&brain].mean() if ((mask==0)&brain).any() else 0
    if (mean_in - mean_out) < CON_MIN: return False, f"contraste={mean_in-mean_out:.3f}"
    if mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any():
        return False, "toca_borde"
    if skull.sum() > 0 and (mask & skull).sum() / mask.sum() > SKULL_OVERLAP:
        return False, "solapamiento_craneo"
    regs = regionprops(label(mask))
    if regs:
        reg   = max(regs, key=lambda r: r.area)
        irreg = reg.perimeter**2 / (4*np.pi*reg.area) if reg.area > 0 else 999
        if irreg > IRREG_MAX:        return False, f"irregularidad={irreg:.2f}"
        if reg.solidity < SOLID_MIN: return False, f"solidez={reg.solidity:.2f}"
    return True, "ok"

# ── Refinamiento con contornos activos ────────────────────────────────────────

def refine_with_snake(img, mask_init):
    if mask_init is None or mask_init.sum() == 0:
        return mask_init
    gimage  = inverse_gaussian_gradient(img, alpha=100, sigma=2.0)
    refined = morphological_geodesic_active_contour(
        gimage,
        num_iter=SNAKE_ITERS,
        init_level_set=mask_init.astype(np.float64),
        smoothing=2,
        balloon=0.6
    )
    return refined.astype(np.uint8)

# ── K-means k=5 ───────────────────────────────────────────────────────────────

def segment_kmeans(img, skull):
    img_clean = img.copy()
    img_clean[skull == 1] = 0.0
    brain = img_clean > 0.05
    if brain.sum() < 10:
        return None, "vacia"
    px    = img_clean[brain].reshape(-1, 1)
    km    = KMeans(n_clusters=5, n_init=5, random_state=42)
    lbs   = km.fit_predict(px)
    full  = np.zeros(img_clean.shape, dtype=np.int32)
    full[brain] = lbs
    means = [img_clean[brain][lbs==i].mean() for i in range(5)]
    mask  = clean_mask((full == int(np.argmax(means))).astype(np.uint8))
    mask  = refine_with_snake(img_clean, mask)
    mask  = keep_largest(mask)
    valid, reason = is_valid(img, mask, skull)
    return (mask if valid else None), reason

# ── Multi-Otsu N=4 ────────────────────────────────────────────────────────────

def segment_multiotsu(img, skull):
    img_clean = img.copy()
    img_clean[skull == 1] = 0.0
    try:
        thresh = threshold_multiotsu(img_clean, classes=4)
    except Exception:
        return None, "error_multiotsu"
    mask  = clean_mask((img_clean > thresh[-1]).astype(np.uint8))
    mask  = refine_with_snake(img_clean, mask)
    mask  = keep_largest(mask)
    valid, reason = is_valid(img, mask, skull)
    return (mask if valid else None), reason

# ── Procesar una clase completa (ambos metodos) ───────────────────────────────

def procesar_clase(cls_name):
    """Corre K-means y Multi-Otsu sobre toda una clase.
    Devuelve los conteos, metricas y la lista de resultados por imagen."""
    paths = sorted((PROC_BASE / cls_name).glob("*.png"))
    (MASK_BASE / cls_name).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SEGMENTACION - {cls_name}")
    print(f"  Total de imagenes encontradas: {len(paths)}")
    print("=" * 60)

    resultados = []
    n_km, n_mo = 0, 0
    rej_km, rej_mo = {}, {}
    m_km = {"area_ratio": [], "contrast": [], "mean_intensity": [], "solidity": []}
    m_mo = {"area_ratio": [], "contrast": [], "mean_intensity": [], "solidity": []}

    for i, p in enumerate(paths, 1):
        print(f"  [{i:3d}/{len(paths)}] {p.stem}...", end=" ", flush=True)
        img   = np.array(Image.open(p), dtype=np.float32) / 255.0
        skull = detect_skull(img)
        mk, rk = segment_kmeans(img, skull)
        mm, rm = segment_multiotsu(img, skull)

        def collect(mask, mdict, reason, rej_dict):
            if mask is not None:
                brain    = img > 0.05
                mean_in  = float(img[mask==1].mean())
                mean_out = float(img[(mask==0)&brain].mean()) if ((mask==0)&brain).any() else 0
                reg      = regionprops(label(mask))
                sol      = max(reg, key=lambda r: r.area).solidity if reg else 0
                mdict["area_ratio"].append(mask.sum() / brain.sum())
                mdict["contrast"].append(mean_in - mean_out)
                mdict["mean_intensity"].append(mean_in)
                mdict["solidity"].append(sol)
                return 1
            else:
                rej_dict[reason] = rej_dict.get(reason, 0) + 1
                return 0

        n_km += collect(mk, m_km, rk, rej_km)
        n_mo += collect(mm, m_mo, rm, rej_mo)
        resultados.append({"path": p, "img": img, "skull": skull,
                            "mk": mk, "mm": mm, "rk": rk, "rm": rm})

        # Guardar mascara: preferir Multi-Otsu, fallback a K-means
        best_mask = mm if mm is not None else mk
        mask_out  = MASK_BASE / cls_name / (p.stem + "_mask.png")
        if best_mask is not None:
            Image.fromarray((best_mask * 255).astype(np.uint8)).save(mask_out)
            print(f"OK ({('MO' if mm is not None else 'KM')})")
        else:
            Image.fromarray(np.zeros((224, 224), dtype=np.uint8)).save(mask_out)
            print(f"rechazada (km={rk}, mo={rm})")

    print(f"\n  Mascaras guardadas en: {MASK_BASE / cls_name}")
    print(f"\n  Validas -> K-means: {n_km}/{len(paths)}  |  Multi-Otsu: {n_mo}/{len(paths)}")
    print(f"  Rechazos K-means   : {rej_km}")
    print(f"  Rechazos Multi-Otsu: {rej_mo}")
    print(f"\n  {'Metrica':18s}  {'K-means':>10}  {'Multi-Otsu':>10}")
    for key in m_km:
        vk = np.mean(m_km[key]) if m_km[key] else 0
        vm = np.mean(m_mo[key]) if m_mo[key] else 0
        print(f"  {key:18s}  {vk:10.4f}  {vm:10.4f}")
    print()

    return {"paths": paths, "resultados": resultados, "n_km": n_km, "n_mo": n_mo,
            "m_km": m_km, "m_mo": m_mo}
    
# ── NOTUMOR: evaluacion de falsos positivos ───────────────────────────────────

def procesar_notumor(cls_name="notumor_prep"):
    """Corre la MISMA segmentacion que en las clases tumorales sobre imagenes
    sin tumor. Si el segmentador devuelve una mascara valida, es un FALSO
    POSITIVO (detecto tumor donde no hay)."""
    paths = sorted((PROC_BASE / cls_name).glob("*.png"))
    (MASK_BASE / cls_name).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FALSOS POSITIVOS - {cls_name}")
    print(f"  Total de imagenes encontradas: {len(paths)}")
    print("=" * 60)

    n_fp = 0
    detectadas = []
    for i, p in enumerate(paths, 1):
        print(f"  [{i:3d}/{len(paths)}] {p.stem}...", end=" ", flush=True)
        img   = np.array(Image.open(p), dtype=np.float32) / 255.0
        skull = detect_skull(img)
        mm, rm = segment_multiotsu(img, skull)
        mk, rk = segment_kmeans(img, skull)

        # Misma logica que las clases tumorales: MO principal, KM respaldo
        best_mask = mm if mm is not None else mk
        mask_out  = MASK_BASE / cls_name / (p.stem + "_mask.png")
        if best_mask is not None:
            Image.fromarray((best_mask * 255).astype(np.uint8)).save(mask_out)
            n_fp += 1
            detectadas.append(p.stem)
            print(f"FALSO POSITIVO ({('MO' if mm is not None else 'KM')})")
        else:
            Image.fromarray(np.zeros((224, 224), dtype=np.uint8)).save(mask_out)
            print("OK (sin deteccion)")

    print(f"\n  Mascaras guardadas en: {MASK_BASE / cls_name}")
    print(f"  Falsos positivos: {n_fp}/{len(paths)}")
    if detectadas:
        print(f"  Imagenes con deteccion erronea: {detectadas}")
    return {"n_fp": n_fp, "total": len(paths), "detectadas": detectadas}

# ── Figura: galeria comparativa ───────────────────────────────────────────────

def fig_galeria(resultados, cls_name, fname):
    muestra = resultados[:MAX_GALERIA]
    n       = len(muestra)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1: axes = axes[np.newaxis, :]

    col_titles = ["Preprocesada", "Craneo detectado", "K-means (azul)", "Multi-Otsu (naranja)"]

    for row, r in enumerate(muestra):
        img, skull, mk, mm = r["img"], r["skull"], r["mk"], r["mm"]

        skull_ov = np.stack([img, img, img], axis=-1)
        skull_ov[skull==1] = 0.5*skull_ov[skull==1] + 0.5*np.array([0.1, 1.0, 0.1])

        def ov(img, mask, rgb):
            out = np.stack([img, img, img], axis=-1)
            if mask is not None:
                for c, v in enumerate(rgb):
                    out[:,:,c] = np.where(mask==1, 0.4*img + 0.6*v, img)
            return np.clip(out, 0, 1)

        ov_k = ov(img, mk, [0.1, 0.4, 1.0])
        ov_m = ov(img, mm, [1.0, 0.5, 0.0])

        for col, arr in enumerate([img, skull_ov, ov_k, ov_m]):
            ax = axes[row, col]
            ax.imshow(arr, cmap="gray" if arr.ndim==2 else None)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel(r["path"].stem, fontsize=8, rotation=0,
                                 labelpad=80, va="center")
        lbl_k = "OK" if mk is not None else r["rk"]
        lbl_m = "OK" if mm is not None else r["rm"]
        axes[row, 2].set_xlabel(f"K-means: {lbl_k}", fontsize=8,
                                 color="blue" if mk is not None else "red")
        axes[row, 3].set_xlabel(f"Multi-Otsu: {lbl_m}", fontsize=8,
                                 color="darkorange" if mm is not None else "red")

    fig.suptitle(f"Segmentacion: K-means vs Multi-Otsu - {cls_name} (primeras {n})",
                 fontsize=13, fontweight="bold", y=1.002)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT_DIR / fname, dpi=120)
    plt.close()
    print(f"OK {fname} guardada")

# ── Figura: boxplots comparativos ─────────────────────────────────────────────

def fig_metricas(m_km, m_mo, cls_name, fname):
    met_labels = ["Area relativa", "Intensidad media", "Contraste", "Solidez"]
    met_keys   = ["area_ratio", "mean_intensity", "contrast", "solidity"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for col, (key, lbl) in enumerate(zip(met_keys, met_labels)):
        ax          = axes[col]
        data        = [m_km[key], m_mo[key]]
        valid_data  = [d for d in data if d]
        valid_lbls  = ["K-means", "Multi-Otsu"][:len(valid_data)]
        if valid_data:
            bp = ax.boxplot(valid_data, tick_labels=valid_lbls,
                            patch_artist=True, widths=0.4)
            for patch, c in zip(bp["boxes"], ["#3498db80", "#f39c1280"]):
                patch.set_facecolor(c)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle(f"Metricas de segmentacion - {cls_name} (mascaras validas)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close()
    print(f"OK {fname} guardada")

# ══════════════════════════════════════════════════════════════════════════════
# EJECUCION
# ══════════════════════════════════════════════════════════════════════════════

# ── MENINGIOMA ──
res_men = procesar_clase("meningioma_prep")
print("Generando figuras de meningioma...")
fig_galeria(res_men["resultados"], "meningioma_prep", "fig5_segmentacion_galeria.png")
fig_metricas(res_men["m_km"], res_men["m_mo"], "meningioma", "fig6_metricas_segmentacion.png")

# ── GLIOMA ──
print()
res_gli = procesar_clase("glioma_prep")
print("Generando figuras de glioma...")
fig_galeria(res_gli["resultados"], "glioma_prep", "fig5b_segmentacion_galeria_glioma.png")
fig_metricas(res_gli["m_km"], res_gli["m_mo"], "glioma", "fig6b_metricas_segmentacion_glioma.png")

# ── NOTUMOR (falsos positivos) ──
print()
res_nt = procesar_notumor("notumor_prep")

# ── RESUMEN FINAL ──
print("\n" + "=" * 60)
print("RESUMEN DE SEGMENTACION (mascaras validas por metodo)")
print("=" * 60)
print(f"  {'Clase':14s} {'Total':>6} {'K-means':>10} {'Multi-Otsu':>12}")
for nombre, res in [("Meningioma", res_men), ("Glioma", res_gli)]:
    tot = len(res["paths"])
    print(f"  {nombre:14s} {tot:6d} {res['n_km']:10d} {res['n_mo']:12d}")

print(f"\n  Falsos positivos (notumor): {res_nt['n_fp']}/{res_nt['total']}")
print(f"\nResultados en: {OUT_DIR}")

