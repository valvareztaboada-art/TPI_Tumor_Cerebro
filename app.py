"""
Sistema de Analisis de Tumores Cerebrales por RM
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import json, uuid, shutil
from datetime import datetime
from pathlib import Path
from PIL import Image
from skimage.filters import apply_hysteresis_threshold, threshold_multiotsu, sobel, gaussian
from skimage.morphology import (binary_dilation, binary_opening, binary_closing,
                                 remove_small_objects, remove_small_holes, disk)
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")

# -- Rutas --------------------------------------------------------------------
BASE          = Path(__file__).parent
PROC_BASE     = BASE / "archive_prep_proc"
RAW_BASE      = BASE / "archive_prep"
RESULTADOS    = BASE / "resultados"
HISTORIAL_F   = RESULTADOS / "historial.json"
PACIENTES_F   = RESULTADOS / "pacientes.json"
FEATURES_CSV  = RESULTADOS / "features_app_acumulados.csv"
RESULTADOS.mkdir(exist_ok=True)

# -- Constantes ---------------------------------------------------------------
AR_MIN, AR_MAX = 0.005, 0.70
CON_MIN        = 0.10
SKULL_OVERLAP  = 0.40
IRREG_MAX      = 4.0
SOLID_MIN      = 0.70
RING_BORDER    = 8
RING_PERI      = 16
BLACK_POINT    = 0.2
EXPOSURE       = 0.6
SHADOWS        = -0.8
GAUSS_SIGMA    = 0.8
TARGET_SIZE    = (224, 224)
BG_THRESH      = 10
CROP_MARGIN    = 5
SNAKE_ITERS    = 60   # igual que en 03_segmentacion.py

CLASSES = {
    "meningioma_prep": "Meningioma",
    "glioma_prep":     "Glioma",
    "notumor_prep":    "Sin tumor",
}
CLASS_COLORS = {
    "meningioma_prep": "#c0392b",
    "glioma_prep":     "#7d3c98",
    "notumor_prep":    "#1e8449",
}

# -- Configuracion pagina -----------------------------------------------------
st.set_page_config(page_title="NeuroScan", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp, .main, .block-container {
    background-color: #0d1b2a !important; color: #ecf0f1 !important;
}
.stApp p, .stApp span, .stApp div, .stApp label,
.stApp li, .stApp h1, .stApp h2, .stApp h3,
.stMarkdown, .stText { color: #ecf0f1 !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a2636 100%) !important;
}
[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
.header-box {
    background: linear-gradient(135deg, #0d1b2a, #1e3a5f);
    padding: 26px 32px; border-radius: 12px;
    border-left: 6px solid #2980b9; margin-bottom: 24px;
}
.header-title { color: #ffffff !important; font-size: 2rem; font-weight: 700; margin: 0; }
.header-sub   { color: #a8c0d6 !important; font-size: 0.9rem; margin-top: 5px; }
.section-title {
    color: #ecf0f1 !important; font-size: 1rem; font-weight: 700;
    border-bottom: 2px solid #2980b9; padding-bottom: 5px; margin: 18px 0 12px 0;
}
.card {
    background: #1a2636 !important; border-radius: 10px; padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.30); margin-bottom: 10px;
    border: 1px solid #2c3e50;
}
.metric-label {
    color: #a8c0d6 !important; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 1px;
}
.metric-value { color: #ecf0f1 !important; font-size: 1.35rem; font-weight: 700; }
.badge-tumor  { background: #c0392b; color: #ffffff !important; padding: 5px 14px; border-radius: 16px; font-size: 0.85rem; font-weight: 600; }
.badge-normal { background: #1e8449; color: #ffffff !important; padding: 5px 14px; border-radius: 16px; font-size: 0.85rem; font-weight: 600; }
.img-label {
    text-align: center; font-size: 0.72rem; color: #a8c0d6 !important;
    margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div {
    background-color: #1a2636 !important; color: #ecf0f1 !important;
    border: none !important; box-shadow: none !important; outline: none !important;
}
[data-testid="stSelectbox"] > div, div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #1a2636 !important; border: none !important;
    box-shadow: none !important; outline: none !important;
}
div[data-baseweb="select"] * { color: #ecf0f1 !important; }
div[data-baseweb="select"] > div:first-child,
div[data-baseweb="base-input"], div[data-baseweb="textarea"],
[data-testid="stWidgetLabel"], .stSelectbox > label,
.stTextInput > label, .stNumberInput > label, .stTextArea > label {
    border: none !important; box-shadow: none !important;
    outline: none !important; color: #a8c0d6 !important;
}
[data-testid="stExpander"] {
    background: #1a2636 !important; border: 1px solid #2c3e50 !important; border-radius: 8px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] p,
[data-testid="stExpander"] span { color: #ecf0f1 !important; }
[data-testid="stDataFrame"] { background: #1a2636 !important; }
[data-testid="stDataFrame"] * { color: #ecf0f1 !important; }
</style>
""", unsafe_allow_html=True)

# -- Persistencia -------------------------------------------------------------
def load_json(path):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_historial():
    if HISTORIAL_F.exists():
        with open(HISTORIAL_F, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_historial(records):
    with open(HISTORIAL_F, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def append_features_csv(clase, imagen, paciente, feats_flat):
    """Agrega una fila al CSV acumulado de features por clase."""
    row = {"fecha": datetime.now().strftime("%d/%m/%Y %H:%M"),
           "clase": clase, "imagen": imagen, "paciente": paciente}
    row.update(feats_flat)
    df_new = pd.DataFrame([row])
    if FEATURES_CSV.exists():
        df_old = pd.read_csv(FEATURES_CSV)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(FEATURES_CSV, index=False)

# -- Preprocesamiento ---------------------------------------------------------
def crop_and_resize(img_raw):
    """Aplica solo crop del cerebro + resize 224x224 a la imagen original.
    Sin N4, sin tono, sin black point -> conserva intensidades reales."""
    arr = (img_raw * 255).astype(np.float32)
    mask_bg = arr > BG_THRESH
    rows = np.where(mask_bg.any(axis=1))[0]
    cols = np.where(mask_bg.any(axis=0))[0]
    if len(rows) > 0 and len(cols) > 0:
        r0 = max(0, rows[0]  - CROP_MARGIN)
        r1 = min(arr.shape[0], rows[-1] + CROP_MARGIN + 1)
        c0 = max(0, cols[0]  - CROP_MARGIN)
        c1 = min(arr.shape[1], cols[-1] + CROP_MARGIN + 1)
        arr = arr[r0:r1, c0:c1]
    arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize(TARGET_SIZE, Image.LANCZOS)) / 255.0
    return arr.astype(np.float32)

def crop_brain(arr):
    mask = arr > BG_THRESH
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows)==0 or len(cols)==0: return arr
    r0,r1 = max(0,rows[0]-CROP_MARGIN), min(arr.shape[0],rows[-1]+CROP_MARGIN+1)
    c0,c1 = max(0,cols[0]-CROP_MARGIN), min(arr.shape[1],cols[-1]+CROP_MARGIN+1)
    return arr[r0:r1, c0:c1]

def n4_correction(img):
    arr = (img*255.0).astype(np.float32)
    si  = sitk.GetImageFromArray(arr)
    sm  = sitk.Cast(sitk.GetImageFromArray((arr>arr.max()*0.05).astype(np.uint8)), sitk.sitkUInt8)
    cor = sitk.N4BiasFieldCorrectionImageFilter()
    cor.SetMaximumNumberOfIterations([20,10])
    out = sitk.GetArrayFromImage(cor.Execute(si,sm))
    mn,mx = out.min(), out.max()
    return ((out-mn)/(mx-mn)).astype(np.float32) if mx>mn else out

def apply_tono(img):
    """Exposicion + ajuste de sombras (igual que 02_preprocesamiento.py)."""
    out = img.copy()
    if EXPOSURE != 0.0:
        out = np.clip(out * (2 ** EXPOSURE), 0, 1)
    if SHADOWS != 0.0:
        mask = out <= 0.5
        out[mask] = np.clip(out[mask] + SHADOWS * (0.5 - out[mask]), 0, 1)
    return out.astype(np.float32)

def apply_bp(img):
    return np.clip((img-BLACK_POINT)/(1.0-BLACK_POINT),0,1).astype(np.float32)

@st.cache_data
def preprocess_bytes(img_bytes):
    arr = np.array(Image.open(img_bytes).convert("L"), dtype=np.float32)
    raw = arr.copy()
    arr = gaussian(arr, sigma=GAUSS_SIGMA, preserve_range=True)
    arr = crop_brain(arr)
    mn,mx = arr.min(), arr.max()
    arr = (arr-mn)/(mx-mn) if mx>mn else arr
    arr = np.array(Image.fromarray((arr*255).astype(np.uint8)).resize(TARGET_SIZE,Image.LANCZOS))/255.0
    arr = n4_correction(arr)
    arr = apply_tono(arr)
    arr = apply_bp(arr)
    return raw, arr

# -- Segmentacion -------------------------------------------------------------
def detect_skull(img):
    sr = apply_hysteresis_threshold(img, low=0.50, high=0.75)
    h,w = img.shape
    b = np.ones_like(img, dtype=bool)
    b[int(h*.20):-int(h*.20), int(w*.20):-int(w*.20)] = False
    return (sr & b).astype(np.uint8)

def keep_largest(mask):
    lb = label(mask)
    if lb.max()==0: return mask
    regs = regionprops(lb)
    return (lb == max(regs,key=lambda r:r.area).label).astype(np.uint8)

def clean_mask(mask):
    mask = binary_opening(mask, disk(3))
    mask = binary_closing(mask, disk(5))
    mask = remove_small_objects(mask, min_size=100)
    mask = remove_small_holes(mask, area_threshold=500)
    return keep_largest(mask)

def is_valid(img, mask, skull):
    if mask.sum()==0: return False
    brain = img>0.05
    if brain.sum()==0: return False
    ar = mask.sum()/brain.sum()
    if not (AR_MIN<=ar<=AR_MAX): return False
    mi = img[mask==1].mean()
    mo = img[(mask==0)&brain].mean() if ((mask==0)&brain).any() else 0
    if (mi-mo)<CON_MIN: return False
    if mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any(): return False
    if skull.sum()>0 and (mask&skull).sum()/mask.sum()>SKULL_OVERLAP: return False
    regs = regionprops(label(mask))
    if regs:
        reg   = max(regs, key=lambda r: r.area)
        irreg = reg.perimeter**2 / (4*np.pi*reg.area) if reg.area>0 else 999
        if irreg > IRREG_MAX:        return False
        if reg.solidity < SOLID_MIN: return False
    return True

def refine_snake(img, mask):
    if mask is None or mask.sum()==0: return mask
    gi  = inverse_gaussian_gradient(img, alpha=100, sigma=2.0)
    ref = morphological_geodesic_active_contour(gi, num_iter=SNAKE_ITERS,
          init_level_set=mask.astype(np.float64), smoothing=2, balloon=0.6)
    return ref.astype(np.uint8)

def segment(img, skull):
    ic = img.copy(); ic[skull==1]=0.0
    try:    thresh = threshold_multiotsu(ic, classes=4)
    except: return None
    mask = clean_mask((ic>thresh[-1]).astype(np.uint8))
    mask = refine_snake(ic, mask)
    mask = keep_largest(mask)
    return mask if is_valid(img,mask,skull) else None

# -- Regiones -----------------------------------------------------------------
def build_regions(mask):
    di = binary_dilation(mask, disk(RING_BORDER)).astype(np.uint8)
    do = binary_dilation(mask, disk(RING_BORDER+RING_PERI)).astype(np.uint8)
    return mask, (di-mask).clip(0,1), (do-di).clip(0,1)

# -- Features (igual que 05_comparacion_clases.py) ----------------------------
def glcm_features_zona(img, mask, prefix):
    """GLCM a 3 distancias y 4 angulos sobre el bounding box de la region."""
    props_list = ["contrast","homogeneity","energy","correlation","dissimilarity"]
    px = img[mask==1]
    if px.size < 16:
        return {f"{prefix}_glcm_{p}_d{d}": 0 for p in props_list for d in [1,2,3]}

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    img_crop  = (img[r0:r1, c0:c1] * 255).astype(np.uint8)
    mask_crop = mask[r0:r1, c0:c1]

    roi = img_crop.copy()
    roi[mask_crop == 0] = 0

    glcm = graycomatrix(roi, distances=[1,2,3],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=256, symmetric=True, normed=True)
    feats = {}
    for prop in props_list:
        values = graycoprops(glcm, prop)
        for di, d in enumerate([1,2,3]):
            feats[f"{prefix}_glcm_{prop}_d{d}"] = round(float(values[di,:].mean()), 4)
    return feats

def intensity_zona(img, mask, prefix):
    px = img[mask==1]
    if len(px)==0:
        return {f"{prefix}_mean":0, f"{prefix}_std":0, f"{prefix}_p25":0, f"{prefix}_p75":0}
    return {
        f"{prefix}_mean": round(float(np.mean(px)),4),
        f"{prefix}_std":  round(float(np.std(px)),4),
        f"{prefix}_p25":  round(float(np.percentile(px,25)),4),
        f"{prefix}_p75":  round(float(np.percentile(px,75)),4),
    }

def gradient_zona(img, mask, prefix):
    px = sobel(img)[mask==1]
    return {f"{prefix}_grad_mean": round(float(np.mean(px)),4) if len(px)>0 else 0}

def shape_feats(img, mask):
    bp   = (img>0.05).sum()
    regs = regionprops(label(mask))
    if not regs:
        return {"shape_area_ratio":0,"shape_perimeter":0,
                "shape_solidity":0,"shape_eccentricity":0,"shape_irregularidad":0}
    reg = max(regs,key=lambda r:r.area)
    p,a = reg.perimeter, reg.area
    return {
        "shape_area_ratio":    round(float(mask.sum()/bp),4) if bp>0 else 0,
        "shape_perimeter":     round(float(p),2),
        "shape_solidity":      round(float(reg.solidity),4),
        "shape_eccentricity":  round(float(reg.eccentricity),4),
        "shape_irregularidad": round(float(p**2/(4*np.pi*a)),4) if a>0 else 0,
    }

def extract_features(img, mask):
    """Retorna (feats_display, feats_flat, nucleo, borde, peri)."""
    nucleo, borde, peri = build_regions(mask)
    feats_flat    = {}
    feats_display = {}

    for zona_key, zona_name, region in [
        ("nucleo","Nucleo",nucleo), ("borde","Borde",borde), ("peri","Peritumoral",peri)
    ]:
        ifeat  = intensity_zona(img, region, zona_key)
        gfeat  = gradient_zona(img, region, zona_key)
        glfeat = glcm_features_zona(img, region, zona_key)
        feats_flat.update(ifeat); feats_flat.update(gfeat); feats_flat.update(glfeat)

        feats_display[f"{zona_name} - Intensidad media"]    = ifeat[f"{zona_key}_mean"]
        feats_display[f"{zona_name} - Desviacion estandar"] = ifeat[f"{zona_key}_std"]
        feats_display[f"{zona_name} - Percentil 25"]        = ifeat[f"{zona_key}_p25"]
        feats_display[f"{zona_name} - Percentil 75"]        = ifeat[f"{zona_key}_p75"]
        feats_display[f"{zona_name} - Gradiente medio"]     = gfeat[f"{zona_key}_grad_mean"]
        feats_display[f"{zona_name} - Contraste GLCM"]      = glfeat.get(f"{zona_key}_glcm_contrast_d1",0)
        feats_display[f"{zona_name} - Homogeneidad GLCM"]   = glfeat.get(f"{zona_key}_glcm_homogeneity_d1",0)
        feats_display[f"{zona_name} - Energia GLCM"]        = glfeat.get(f"{zona_key}_glcm_energy_d1",0)
        feats_display[f"{zona_name} - Correlacion GLCM"]    = glfeat.get(f"{zona_key}_glcm_correlation_d1",0)

    sfeat = shape_feats(img, mask)
    feats_flat.update(sfeat)
    feats_display["Forma - Area relativa"]   = sfeat["shape_area_ratio"]
    feats_display["Forma - Perimetro"]       = sfeat["shape_perimeter"]
    feats_display["Forma - Solidez"]         = sfeat["shape_solidity"]
    feats_display["Forma - Excentricidad"]   = sfeat["shape_eccentricity"]
    feats_display["Forma - Irregularidad"]   = sfeat["shape_irregularidad"]

    return feats_display, feats_flat, nucleo, borde, peri

def radial_profile(img, mask, n=10):
    regs = regionprops(label(mask))
    if not regs: return np.zeros(n)
    reg   = max(regs,key=lambda r:r.area)
    cy,cx = reg.centroid
    ys,xs = np.mgrid[0:img.shape[0],0:img.shape[1]]
    dist  = np.sqrt((ys-cy)**2+(xs-cx)**2)
    maxr  = dist[mask==1].max()*2.0 if mask.sum()>0 else 1.0
    bins  = np.linspace(0,maxr,n+1)
    vals  = []
    for i in range(n):
        ring = (dist>=bins[i])&(dist<bins[i+1])&(img>0.05)
        vals.append(float(img[ring].mean()) if ring.sum()>0 else 0)
    return np.array(vals)

def overlay(img, mask, rgb):
    out = np.stack([img,img,img],axis=-1)
    if mask is not None and mask.sum()>0:
        for c,v in enumerate(rgb):
            out[:,:,c] = np.where(mask==1, 0.4*img+0.6*v, img)
    return np.clip(out,0,1)

# -- Bloque de analisis completo ----------------------------------------------
def bloque_analisis(img_proc, img_raw, paciente, nombre_img, clase_img):
    st.markdown('<div class="section-title">Preprocesamiento</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1,2,figsize=(8,3.2),facecolor="#f0f2f5")
    axes[0].imshow(img_raw, cmap="gray"); axes[0].set_title("Original",fontsize=9,color="#2c3e50"); axes[0].axis("off")
    axes[1].imshow(img_proc,cmap="gray"); axes[1].set_title("Preprocesada",fontsize=9,color="#2c3e50"); axes[1].axis("off")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Segmentacion</div>', unsafe_allow_html=True)
    with st.spinner("Segmentando..."):
        skull = detect_skull(img_proc)
        mask  = segment(img_proc, skull)

    if mask is None:
        st.markdown('<span class="badge-normal">SIN TUMOR DETECTADO</span>', unsafe_allow_html=True)
        st.info("El segmentador no encontro ninguna region tumoral en esta imagen.")
        if st.button("Guardar en historial", key=f"guardar_notumor_{nombre_img}"):
            _guardar_historial(nombre_img, paciente, clase_img, "Sin tumor", {}, {})
            st.success("Guardado en el historial.")
        return

    st.markdown('<span class="badge-tumor">REGION TUMORAL DETECTADA</span>', unsafe_allow_html=True)
    st.write("")

    img_orig_224 = crop_and_resize(img_raw)

    feats_display, feats_flat, nucleo, borde, peri = extract_features(img_orig_224, mask)
    radial = radial_profile(img_orig_224, mask)

    skull_ov = overlay(img_proc, skull, [0.1,1.0,0.1])
    ov_mask  = overlay(img_proc, mask,  [1.0,0.2,0.2])
    comp = np.stack([img_orig_224, img_orig_224, img_orig_224], axis=-1)
    comp[peri==1,0]   = 0.0
    comp[peri==1,1]   = np.clip(0.4*img_orig_224[peri==1]+0.6,0,1)
    comp[peri==1,2]   = np.clip(0.4*img_orig_224[peri==1]+0.6,0,1)
    comp[borde==1,0]  = np.clip(0.4*img_orig_224[borde==1]+0.6,0,1)
    comp[borde==1,1]  = np.clip(0.4*img_orig_224[borde==1]+0.6,0,1)
    comp[borde==1,2]  = 0.0
    comp[nucleo==1,0] = np.clip(0.4*img_orig_224[nucleo==1]+0.6,0,1)
    comp[nucleo==1,1] = 0.0
    comp[nucleo==1,2] = 0.0
    comp = np.clip(comp,0,1)

    c1,c2,c3 = st.columns(3)
    c1.image(skull_ov, caption="Craneo detectado (preprocesada)",    use_container_width=True)
    c2.image(ov_mask,  caption="Region tumoral (preprocesada)",      use_container_width=True)
    c3.image(comp,     caption="Regiones: Rojo=nucleo  Amarillo=borde  Cian=peritumoral (original)", use_container_width=True)

    st.markdown('<div class="section-title">Caracteristicas por zona</div>', unsafe_allow_html=True)
    zonas   = ["Nucleo","Borde","Peritumoral","Forma"]
    colores = ["#c0392b","#e67e22","#2980b9","#27ae60"]
    zcols   = st.columns(4)
    for i, zona in enumerate(zonas):
        with zcols[i]:
            st.markdown(f"<div style='background:{colores[i]};color:white;padding:7px 10px;"
                        f"border-radius:6px;font-weight:700;text-align:center;"
                        f"margin-bottom:8px'>{zona}</div>", unsafe_allow_html=True)
            for k,v in {k2.split(" - ")[1]:v2 for k2,v2 in feats_display.items()
                        if k2.startswith(zona)}.items():
                st.markdown(f"""<div class="card">
                    <div class="metric-label">{k}</div>
                    <div class="metric-value">{v}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Textura GLCM por distancia (nucleo)</div>', unsafe_allow_html=True)
    glcm_props = ["contrast","homogeneity","energy","correlation"]
    glcm_names = ["Contraste","Homogeneidad","Energia","Correlacion"]
    fig, axes  = plt.subplots(1,4,figsize=(14,3),facecolor="white")
    for ax, prop, name in zip(axes, glcm_props, glcm_names):
        vals = [feats_flat.get(f"nucleo_glcm_{prop}_d{d}",0) for d in [1,2,3]]
        ax.bar([1,2,3], vals, color="#c0392b", alpha=0.75, width=0.5)
        ax.set_title(name, fontsize=9, fontweight="bold", color="#2c3e50")
        ax.set_xlabel("Distancia (px)", fontsize=8)
        ax.set_xticks([1,2,3])
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#fafafa")
    fig.suptitle("Nucleo tumoral - GLCM a distintas distancias",
                 fontsize=10, fontweight="bold", color="#2c3e50")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Perfil radial de intensidad</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9,3),facecolor="white")
    x = np.arange(10)
    ax.plot(x, radial, color="#c0392b", linewidth=2.5, marker="o", markersize=5)
    ax.fill_between(x, 0, radial, color="#c0392b", alpha=0.12)
    ax.set_xlabel("Anillo (centro hacia periferia)",fontsize=9)
    ax.set_ylabel("Intensidad media",fontsize=9)
    ax.set_title("Perfil radial desde el centro del tumor",fontsize=10,fontweight="bold",color="#2c3e50")
    ax.set_xticks(x); ax.set_xticklabels([f"r{i}" for i in x],fontsize=8)
    ax.spines[["top","right"]].set_visible(False); ax.set_facecolor("#fafafa")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    col_dl, col_save = st.columns([1,1])
    with col_dl:
        st.download_button("Descargar features (CSV)",
            data=pd.DataFrame([feats_flat]).to_csv(index=False),
            file_name=f"features_{nombre_img}.csv", mime="text/csv")
    with col_save:
        if st.button("Guardar en historial", key=f"guardar_{nombre_img}", type="primary"):
            _guardar_historial(nombre_img, paciente, clase_img, "Tumor detectado",
                               feats_display, feats_flat)
            st.success("Guardado en historial y en CSV acumulado.")

def _guardar_historial(nombre_img, paciente, clase_img, resultado, feats_display, feats_flat):
    rec = {
        "id":        str(uuid.uuid4())[:8],
        "fecha":     datetime.now().strftime("%d/%m/%Y %H:%M"),
        "imagen":    nombre_img,
        "clase":     clase_img,
        "paciente":  paciente.get("nombre","Desconocido"),
        "edad":      paciente.get("edad","-"),
        "sexo":      paciente.get("sexo","-"),
        "id_pac":    paciente.get("id","-"),
        "resultado": resultado,
        "features":  {k: float(v) for k,v in feats_display.items()},
    }
    h = load_historial(); h.insert(0,rec); save_historial(h)
    if feats_flat:
        append_features_csv(clase_img, nombre_img,
                            paciente.get("nombre","Desconocido"), feats_flat)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 8px 0'>
        <div style='font-size:2.2rem'>&#9762;</div>
        <div style='font-size:1.05rem;font-weight:700;color:#ecf0f1;letter-spacing:1px'>NEUROSCAN</div>
        <div style='font-size:0.72rem;color:#7f8c8d;margin-top:3px'>Sistema de Analisis Tumoral</div>
    </div>
    <hr style='border-color:#2c3e50;margin:8px 0'>
    """, unsafe_allow_html=True)

    pagina = st.radio("", [
        "Inicio", "Galeria", "Agregar paciente", "Historial", "Comparacion de clases"
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#2c3e50;margin:16px 0 8px 0'>
    <div style='font-size:0.7rem;color:#555;text-align:center'>
        Grupo 6<br>Alvarez Taboada / Losinno / Gowland<br>
        <span style='color:#3d5a80'>Procesamiento de Imagenes 2026</span>
    </div>""", unsafe_allow_html=True)

# -- Header -------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="header-title">Sistema de Analisis de Tumores Cerebrales por RM</div>
    <div class="header-sub">Segmentacion automatica / Extraccion de caracteristicas / Regiones concentricas</div>
</div>""", unsafe_allow_html=True)

# =============================================================================
# INICIO
# =============================================================================
if pagina == "Inicio":
    total = sum(len(list((PROC_BASE/c).glob("*.png"))) for c in CLASSES if (PROC_BASE/c).exists())
    hist  = load_historial()
    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,val,clr) in zip([c1,c2,c3,c4],[
        ("Imagenes en base de datos", total,  "#2980b9"),
        ("Analisis realizados",       len(hist), "#27ae60"),
        ("Clases disponibles",        len(CLASSES), "#8e44ad"),
        ("Con tumor detectado", sum(1 for r in hist if r.get("resultado")=="Tumor detectado"), "#c0392b"),
    ]):
        col.markdown(f"""<div class="card" style="border-top:4px solid {clr}">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="color:{clr}">{val}</div></div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="section-title">Descripcion del sistema</div>', unsafe_allow_html=True)
    st.markdown("""
- **Preprocesamiento**: suavizado gaussiano, crop cerebral, normalizacion, correccion N4, ajuste de tono y ajuste de punto negro
- **Segmentacion**: Multi-Otsu (4 clases) con resta de craneo y refinamiento por contornos activos geodesicos (60 iter.)
- **Regiones concentricas**: nucleo tumoral, banda de borde (+8 px) y corona peritumoral (+16 px)
- **Textura GLCM**: contraste, homogeneidad, energia, correlacion y disimilaridad a 3 distancias (1, 2, 3 px), 4 angulos promediados
- **Perfil radial**: evolucion de intensidad desde el centro del tumor hacia la periferia (10 anillos)
- **Acumulacion de features**: cada analisis guardado se suma al CSV por clase para comparacion posterior
    """)

    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
    cols = st.columns(len(CLASSES))
    for i,(cls,lbl) in enumerate(CLASSES.items()):
        n = len(list((PROC_BASE/cls).glob("*.png"))) if (PROC_BASE/cls).exists() else 0
        cols[i].markdown(f"""<div class="card" style="border-top:4px solid {CLASS_COLORS[cls]};text-align:center">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="color:{CLASS_COLORS[cls]}">{n} imagenes</div></div>""",
            unsafe_allow_html=True)

# =============================================================================
# GALERIA
# =============================================================================
elif pagina == "Galeria":
    pacientes_db = load_json(PACIENTES_F)
    cls_sel = st.selectbox("Clase", list(CLASSES.keys()), format_func=lambda x: CLASSES[x])

    raw_dir  = RAW_BASE  / cls_sel
    proc_dir = PROC_BASE / cls_sel

    raw_paths = []
    if raw_dir.exists():
        for ext in ["*.jpg","*.jpeg","*.png"]:
            raw_paths += list(raw_dir.glob(ext))
    raw_paths = sorted(raw_paths)

    if not raw_paths:
        st.warning("No hay imagenes originales para esta clase.")
    else:
        st.markdown(f"**{len(raw_paths)} imagenes disponibles**")

        if "sel_galeria_cls" not in st.session_state:
            st.session_state["sel_galeria_cls"] = None
            st.session_state["sel_galeria_img"] = None

        CPR = 5
        for i in range(0, len(raw_paths), CPR):
            row_cols = st.columns(CPR)
            for j, p in enumerate(raw_paths[i:i+CPR]):
                with row_cols[j]:
                    img_g = np.array(Image.open(p).convert("L"), dtype=np.float32)/255.0
                    st.image(img_g, clamp=True, use_container_width=True)
                    st.markdown(f'<div class="img-label">{p.stem}</div>', unsafe_allow_html=True)
                    if st.button("Analizar", key=f"anal_{p.stem}", use_container_width=True):
                        st.session_state["sel_galeria_cls"] = cls_sel
                        st.session_state["sel_galeria_img"] = str(p)

        if st.session_state.get("sel_galeria_img"):
            p_raw    = Path(st.session_state["sel_galeria_img"])
            cls_sel2 = st.session_state.get("sel_galeria_cls", cls_sel)
            st.divider()
            st.markdown(f'<div class="section-title">Analisis: {p_raw.stem}</div>', unsafe_allow_html=True)

            img_raw = np.array(Image.open(p_raw).convert("L"), dtype=np.float32)/255.0

            p_proc = proc_dir / (p_raw.stem + ".png")
            if p_proc.exists():
                img_proc = np.array(Image.open(p_proc), dtype=np.float32)/255.0
            else:
                st.warning(f"No se encontro imagen preprocesada para {p_raw.stem}. Ejecuta 02_preprocesamiento.py primero.")
                img_proc = img_raw

            pac_data = pacientes_db.get(p_raw.stem, {"nombre":"Desconocido","edad":"-","sexo":"-","id":"-"})
            bloque_analisis(img_proc, img_raw, pac_data, p_raw.stem, cls_sel2)
            if st.button("Cerrar analisis"):
                st.session_state["sel_galeria_img"] = None
                st.rerun()

# =============================================================================
# AGREGAR PACIENTE
# =============================================================================
elif pagina == "Agregar paciente":
    st.markdown('<div class="section-title">Registrar nuevo paciente e imagen</div>', unsafe_allow_html=True)
    pacientes_db = load_json(PACIENTES_F)
    col_form, col_prev = st.columns([1,1])

    with col_form:
        nombre   = st.text_input("Nombre completo *")
        c1,c2    = st.columns(2)
        edad     = c1.number_input("Edad", min_value=1, max_value=120, value=45)
        sexo     = c2.selectbox("Sexo", ["M","F"])
        id_pac   = st.text_input("ID Paciente", value=f"P-{str(uuid.uuid4())[:4].upper()}")
        cls_new  = st.selectbox("Tipo de tumor / clase", list(CLASSES.keys()),
                                format_func=lambda x: CLASSES[x])
        notas    = st.text_area("Notas clinicas", height=80)
        uploaded = st.file_uploader("Imagen de resonancia magnetica (JPG/PNG)", type=["jpg","jpeg","png"])

    with col_prev:
        if uploaded:
            st.markdown('<div class="section-title">Vista previa</div>', unsafe_allow_html=True)
            st.image(Image.open(uploaded).convert("L"), caption="Imagen cargada", use_container_width=True)

    if uploaded and nombre:
        if st.button("Guardar paciente e imagen", type="primary"):
            with st.spinner("Preprocesando y guardando..."):
                uploaded.seek(0)
                raw_arr, proc_arr = preprocess_bytes(uploaded)
                stem     = f"{nombre.replace(' ','_')}_{id_pac}"
                dst_raw  = RAW_BASE  / cls_new / f"{stem}.jpg"
                dst_proc = PROC_BASE / cls_new / f"{stem}.png"
                (RAW_BASE/cls_new).mkdir(parents=True, exist_ok=True)
                (PROC_BASE/cls_new).mkdir(parents=True, exist_ok=True)
                uploaded.seek(0)
                Image.open(uploaded).convert("L").save(dst_raw)
                Image.fromarray((proc_arr*255).astype(np.uint8)).save(dst_proc)
                pacientes_db[stem] = {
                    "nombre": nombre, "edad": int(edad), "sexo": sexo, "id": id_pac,
                    "clase": cls_new, "notas": notas,
                    "fecha_registro": datetime.now().strftime("%d/%m/%Y"),
                }
                save_json(PACIENTES_F, pacientes_db)
            st.success(f"Paciente '{nombre}' registrado en {CLASSES[cls_new]}.")
            st.info("La imagen ya aparece en la Galeria.")
    elif not nombre and uploaded:
        st.warning("Ingresa el nombre del paciente para continuar.")

# =============================================================================
# HISTORIAL
# =============================================================================
elif pagina == "Historial":
    hist = load_historial()
    if not hist:
        st.info("No hay analisis registrados aun.")
    else:
        st.markdown(f"**{len(hist)} analisis registrados**")
        c1,c2 = st.columns([2,1])
        f_pac = c1.text_input("Buscar por paciente")
        f_res = c2.selectbox("Resultado", ["Todos","Tumor detectado","Sin tumor"])

        registros = hist
        if f_pac: registros = [r for r in registros if f_pac.lower() in r.get("paciente","").lower()]
        if f_res != "Todos": registros = [r for r in registros if r.get("resultado")==f_res]

        st.markdown("---")
        for rec in registros:
            color  = "#e74c3c" if rec.get("resultado")=="Tumor detectado" else "#27ae60"
            badge  = "TUMOR"  if rec.get("resultado")=="Tumor detectado" else "NORMAL"
            rec_id = rec.get("id","")
            with st.expander(f"{rec.get('fecha','--')}  |  {rec.get('paciente','--')}  |  {rec.get('imagen','--')}"):
                cc1,cc2,cc3,cc4,cc5 = st.columns([2,2,2,1,1])
                cc1.markdown(f"**Paciente:** {rec.get('paciente','-')}")
                cc2.markdown(f"**Edad:** {rec.get('edad','-')}  |  **Sexo:** {rec.get('sexo','-')}")
                cc3.markdown(f"**Clase:** {CLASSES.get(rec.get('clase',''), rec.get('clase','-'))}")
                cc4.markdown(f"<span style='background:{color};color:white;padding:4px 12px;"
                             f"border-radius:12px;font-size:0.8rem;font-weight:700'>{badge}</span>",
                             unsafe_allow_html=True)
                if cc5.button("Eliminar", key=f"del_{rec_id}", type="secondary"):
                    save_historial([r for r in hist if r.get("id")!=rec_id])
                    st.rerun()
                if rec.get("features"):
                    st.dataframe(pd.DataFrame([rec["features"]]), use_container_width=True)

        if st.button("Exportar historial completo (CSV)"):
            rows = []
            for r in hist:
                row = {k:r.get(k) for k in ["fecha","clase","paciente","edad","sexo","id_pac","imagen","resultado"]}
                row.update(r.get("features",{}))
                rows.append(row)
            st.download_button("Descargar CSV", data=pd.DataFrame(rows).to_csv(index=False),
                               file_name="historial_analisis.csv", mime="text/csv")

# =============================================================================
# COMPARACION DE CLASES
# =============================================================================
elif pagina == "Comparacion de clases":
    st.markdown('<div class="section-title">Comparacion de features entre clases</div>', unsafe_allow_html=True)

    if not FEATURES_CSV.exists():
        st.info("Aun no hay features acumulados. Analiza imagenes desde la Galeria y guardalas en el historial.")
    else:
        df = pd.read_csv(FEATURES_CSV)
        clases_disp = df["clase"].unique().tolist()

        st.markdown(f"**{len(df)} analisis acumulados** | Clases: "
                    f"{', '.join(CLASSES.get(c,c) for c in clases_disp)}")

        if len(clases_disp) < 2:
            st.warning("Necesitas analisis de al menos 2 clases distintas para comparar.")
        else:
            feat_cols = [c for c in df.columns if c not in ["fecha","clase","imagen","paciente"]]

            st.markdown("#### Comparar una feature")
            feat_sel = st.selectbox("Feature", feat_cols)
            fig, ax  = plt.subplots(figsize=(8,4), facecolor="white")
            data, labels, colors = [], [], []
            for cls in clases_disp:
                vals = df[df["clase"]==cls][feat_sel].dropna().tolist()
                if vals:
                    data.append(vals); labels.append(CLASSES.get(cls,cls))
                    colors.append(CLASS_COLORS.get(cls,"#888888"))
            if data:
                bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.4)
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c + "80")
            ax.set_title(feat_sel.replace("_"," "), fontsize=11, fontweight="bold", color="#2c3e50")
            ax.spines[["top","right"]].set_visible(False); ax.set_facecolor("#fafafa")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("#### GLCM del nucleo por distancia")
            glcm_props = ["contrast","homogeneity","energy","correlation"]
            glcm_names = ["Contraste","Homogeneidad","Energia","Correlacion"]
            fig, axes  = plt.subplots(1,4,figsize=(16,4),facecolor="white")
            for ax, prop, name in zip(axes, glcm_props, glcm_names):
                for cls in clases_disp:
                    df_cls = df[df["clase"]==cls]
                    means  = []
                    for d in [1,2,3]:
                        col = f"nucleo_glcm_{prop}_d{d}"
                        means.append(df_cls[col].mean() if col in df_cls.columns else 0)
                    ax.plot([1,2,3], means, label=CLASSES.get(cls,cls),
                            color=CLASS_COLORS.get(cls,"#888"), marker="o", linewidth=2)
                ax.set_title(name, fontsize=9, fontweight="bold", color="#2c3e50")
                ax.set_xlabel("Distancia (px)", fontsize=8)
                ax.set_xticks([1,2,3])
                ax.spines[["top","right"]].set_visible(False); ax.set_facecolor("#fafafa")
                if ax is axes[0]: ax.legend(fontsize=8)
            fig.suptitle("GLCM nucleo tumoral - Meningioma vs Glioma",
                         fontsize=11, fontweight="bold", color="#2c3e50")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("#### Tabla resumen (media por clase)")
            resumen = df.groupby("clase")[feat_cols].mean().round(4)
            resumen.index = [CLASSES.get(i,i) for i in resumen.index]
            st.dataframe(resumen, use_container_width=True)

            st.download_button("Descargar features acumulados (CSV)",
                data=df.to_csv(index=False),
                file_name="features_app_acumulados.csv", mime="text/csv")