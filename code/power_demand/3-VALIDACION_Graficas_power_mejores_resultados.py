# -*- coding: utf-8 -*-
"""
3-VALIDACION_Graficas_power_mejores_resultados.py
------------------------------------------------------------
Genera la figura de una configuración concreta buscándola
por parámetros, no por índice de fila.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import re

# =========================================================
# CONFIGURACIÓN
# =========================================================
VALIDACION_DIR = Path(__file__).resolve().parent
ROOT_DIR = VALIDACION_DIR.parent

CANDIDATES_FILE = VALIDACION_DIR / "xlsx" / "candidatos_pass_filter_train_selection.xlsx"
MODELS_DIR = ROOT_DIR / "modelos_dinamico" / "otros_modelos"
DATASET_FILE = ROOT_DIR / "power_dataset.txt"
OUT_IMG_DIR = VALIDACION_DIR / "imagenes"

TRAIN_FRAC = 0.70

# Rangos anómalos
A1_RANGE = (7872, 9216)
A2_RANGE = (11232, 12576)
A3_RANGE = (33780, 35040)

ANOM_RANGES = [A1_RANGE, A2_RANGE, A3_RANGE]

SHOW_FIGURES = True
PAUSE_SECONDS = 5

# =========================================================
# CONFIGURACIÓN A BUSCAR
# =========================================================
TARGET_MODEL = "base_power_dataset_WS128_H50_BS16.h5"
TARGET_WINDOW_THR = 550
TARGET_FACTOR = 1.2
TARGET_K = 1.85

# =========================================================
# HELPERS
# =========================================================
def load_series(path: Path) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for p in line.replace(",", " ").split():
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
    if not vals:
        raise ValueError(f"No se pudo leer ninguna muestra válida de: {path}")
    return np.asarray(vals, dtype=float)

def parse_model_params(name: str):
    ws_match = re.search(r"_WS(\d+)", name)
    h_match = re.search(r"_H(\d+)", name)

    if ws_match is None or h_match is None:
        raise ValueError(f"No se han podido extraer WS y H del nombre del modelo: {name}")

    ws = int(ws_match.group(1))
    h = int(h_match.group(1))
    return ws, h

def rolling_mean_std_nan(x: np.ndarray, w: int):
    n = len(x)
    mu = np.full(n, np.nan, dtype=float)
    sd = np.full(n, np.nan, dtype=float)
    for i in range(w - 1, n):
        win = x[i - w + 1:i + 1]
        win = win[np.isfinite(win)]
        if win.size >= 2:
            mu[i] = win.mean()
            sd[i] = win.std()
    return mu, sd

def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero de candidatos: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Formato no soportado: {path.suffix}")

def find_target_row(df: pd.DataFrame) -> tuple[int, pd.Series]:
    # Normalizamos tipos
    df2 = df.copy()

    df2["model"] = df2["model"].astype(str)
    df2["WINDOW_THR"] = pd.to_numeric(df2["WINDOW_THR"], errors="coerce")
    df2["FACTOR"] = pd.to_numeric(df2["FACTOR"], errors="coerce")
    df2["k"] = pd.to_numeric(df2["k"], errors="coerce")

    mask = (
        (df2["model"] == TARGET_MODEL) &
        (df2["WINDOW_THR"] == TARGET_WINDOW_THR) &
        (np.isclose(df2["FACTOR"], TARGET_FACTOR, atol=1e-9)) &
        (np.isclose(df2["k"], TARGET_K, atol=1e-9))
    )

    matches = df2[mask]

    if len(matches) == 0:
        raise ValueError(
            "No se ha encontrado ninguna fila con:\n"
            f"  model={TARGET_MODEL}\n"
            f"  WINDOW_THR={TARGET_WINDOW_THR}\n"
            f"  FACTOR={TARGET_FACTOR}\n"
            f"  k={TARGET_K}"
        )

    if len(matches) > 1:
        print("[AVISO] Se han encontrado varias filas. Se usará la primera coincidencia.")

    row_idx = matches.index[0]
    row = matches.iloc[0]
    return row_idx, row

# =========================================================
# MAIN
# =========================================================
def main():
    print(f"[VALIDACION] {VALIDACION_DIR}")
    print(f"[ROOT]       {ROOT_DIR}")
    print(f"[CANDIDATOS] {CANDIDATES_FILE}")
    print(f"[MODELOS]    {MODELS_DIR}")
    print(f"[DATASET]    {DATASET_FILE}")
    print(f"[IMAGENES]   {OUT_IMG_DIR}")
    print(f"[A3 USADA]   {A3_RANGE}")

    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_candidates(CANDIDATES_FILE)
    y = load_series(DATASET_FILE)

    n = len(y)
    split_raw = int(n * TRAIN_FRAC)

    ymin, ymax = y[:split_raw].min(), y[:split_raw].max()
    y_s = (y - ymin) / (ymax - ymin + 1e-8)

    row_idx, row = find_target_row(df)

    print("\n[FILA ENCONTRADA]")
    print(f"Índice pandas : {row_idx}")
    print(row.to_string())

    model_name = row["model"]
    W = int(row["WINDOW_THR"])
    F = float(row["FACTOR"])
    K = float(row["k"])

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    print(f"\n[INFO] Generando figura para la fila encontrada")

    ws, h = parse_model_params(model_name)
    model = tf.keras.models.load_model(model_path)

    # ---- forecast ----
    s = tf.expand_dims(tf.convert_to_tensor(y_s, tf.float32), -1)
    ds = tf.data.Dataset.from_tensor_slices(s)
    ds = ds.window(ws, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x.batch(ws))
    pred_s = model.predict(ds.batch(1024), verbose=0).squeeze()

    start_idx = ws + h - 1
    L = min(len(pred_s), n - start_idx)
    if L <= 0:
        raise ValueError("Longitud inválida al alinear predicción")

    index = np.arange(start_idx, start_idx + L)
    y_al = y[start_idx:start_idx + L]

    pred = pred_s[:L] * (ymax - ymin + 1e-8) + ymin
    pred_shift = np.full(L, np.nan, dtype=float)
    if L > h:
        pred_shift[:-h] = pred[h:]

    err = np.abs(y_al - pred_shift)

    mu_roll, sd_roll = rolling_mean_std_nan(err, W)
    valid = np.isfinite(err) & np.isfinite(mu_roll) & np.isfinite(sd_roll)

    thr = mu_roll + K * sd_roll
    thr_scaled = F * thr

    alarm = (err > thr_scaled) & valid
    alarm[:W] = False

    # ---- resumen A3 ----
    a3_mask = (index >= A3_RANGE[0]) & (index < A3_RANGE[1])
    alarms_a3 = int(np.sum(alarm[a3_mask]))
    print(f"\n[RESUMEN A3]")
    print(f"Alarmas en A3 recalculadas: {alarms_a3}")

    # ---- PLOT ----
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(index, y_al, label="Real")
    ax1.plot(index, pred_shift, label="Predicción")

    if np.any(alarm):
        ax1.scatter(index[alarm], y_al[alarm], color="red", s=15, label="Alarmas")

    for a, b in ANOM_RANGES:
        ax1.axvspan(a, b, color="red", alpha=0.12)

    ax1.axvline(split_raw, linestyle="--", color="tab:blue", label="Train/Test")
    ax1.legend()
    ax1.set_title("dataset completo con alertas generadas")

    ax2.plot(index, err, label="|error|")
    ax2.plot(index, thr_scaled, linestyle="--", label="Umbral")

    if np.any(alarm):
        ax2.scatter(index[alarm], err[alarm], color="red", s=12)

    for a, b in ANOM_RANGES:
        ax2.axvspan(a, b, color="red", alpha=0.12)

    ax2.axvline(split_raw, linestyle="--", color="tab:blue")
    ax2.legend()
    ax2.set_xlabel("Índice")

    fig.tight_layout()

    out_name = (
        f"target_{model_name.replace('.h5','')}"
        f"_W{W}_F{str(F).replace('.','p')}_K{str(K).replace('.','p')}.png"
    )
    out_path = OUT_IMG_DIR / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show(block=False)
        plt.pause(PAUSE_SECONDS)

    plt.close(fig)

    print("[OK] Guardada:", out_path)

if __name__ == "__main__":
    main()