# -*- coding: utf-8 -*-
"""
ESTE CÓDIGO CALCULA SÓLO LAS MÉTRICAS DEL MODELO MÁS ADECUADO GENERADO
POR LSTM_SHUTTLE_ESTATICO_BARRIDO.py
"""

"""
LSTM_SHUTTLE_ESTATICO_BARRIDO_METRICAS.py
------------------------------------------------------------
Evalúa MÉTRICAS del mejor modelo obtenido en:
  LSTM_SHUTTLE_ESTATICO_BARRIDO.py

⚠️ Este script:
  - NO entrena modelos
  - NO guarda modelos
  - SOLO carga un modelo ya existente
------------------------------------------------------------
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================================================
# CONFIG (FIJA – MODELO ÓPTIMO)
# =========================================================

SRC_TXT = "TEK16.txt"
DATASET_NAME = Path(SRC_TXT).stem

TRAIN_FRAC = 0.70

ANOM_I0 = 4200
ANOM_I1 = 4380
W_EARLY = 40

WS = 128
H  = 45
BS = 16

MAXv = 0
k = 8.10

EVAL_ON_TEST_ONLY = True

CACHE_DIR = Path("LSTM_SHUTTLE_ESTATICO")
FIG_DIR = CACHE_DIR / "figs_metricas"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAPER_PREC = 0.93
PAPER_REC  = 0.10
PAPER_F01  = 0.84

TAG = f"{DATASET_NAME}_k{k:.2f}_MAX{MAXv}_WS{WS}_H{H}_BS{BS}"

# =========================================================
# HELPERS
# =========================================================

def load_series(path):
    values = []
    with open(path) as f:
        for line in f:
            for p in line.replace("\t", " ").split():
                try:
                    values.append(float(p))
                except:
                    pass
    if len(values) == 0:
        raise ValueError("Dataset vacío o mal leído")
    return np.asarray(values)

def split_index_by_frac(n, frac):
    return int(np.floor(n * frac))

def model_forecast(model, series, w):
    s = tf.expand_dims(tf.convert_to_tensor(series, tf.float32), -1)
    ds = tf.data.Dataset.from_tensor_slices(s)
    ds = ds.window(w, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x.batch(w))
    return model.predict(ds.batch(32), verbose=0).squeeze()

def fbeta(p, r, beta=0.1):
    if (p + r) == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)

# =========================================================
# PLOTS
# =========================================================

def plot_global(index, y, pred, err, alarm,
                eval_mask, split_idx, mu, sigma, k,
                lead_idx, n_alarms_test, tag):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    in_anom = (index >= ANOM_I0) & (index <= ANOM_I1)
    y_norm = y.copy()
    y_anom = np.full_like(y, np.nan)
    y_anom[in_anom] = y[in_anom]
    y_norm[in_anom] = np.nan

    ax1.plot(index, y_norm, label="Real (normal)")
    ax1.plot(index, y_anom, label="Real (anómalo)", color="red", linewidth=2)
    ax1.plot(index, pred, label="Pred LSTM", alpha=0.8)

    ax1.scatter(
        index[alarm & eval_mask],
        y[alarm & eval_mask],
        color="red",
        s=25,
        label=f"Alarmas TEST: {n_alarms_test}"  # <<< NUEVO >>>
    )

    ax1.axvspan(ANOM_I0 - W_EARLY, ANOM_I0, color="orange", alpha=0.25, label="W_EARLY")
    ax1.axvline(index[split_idx], color="black", linestyle="--", label="TRAIN/TEST")

    if lead_idx is not None:
        ax1.axvline(
            lead_idx,
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"Anticipación: {ANOM_I0 - lead_idx}"
        )

    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(index, err, label="|error|")
    ax2.axhline(mu, linestyle="--", label="μ_train")
    ax2.axhline(mu + k * sigma, linestyle=":", label="μ + kσ")

    ax2.scatter(
        index[alarm & eval_mask],
        err[alarm & eval_mask],
        color="red",
        s=25,
        label=f"Alarmas TEST: {n_alarms_test}"  # <<< NUEVO >>>
    )

    ax2.axvspan(ANOM_I0 - W_EARLY, ANOM_I0, color="orange", alpha=0.25)
    ax2.axvline(index[split_idx], color="black", linestyle="--")

    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"global_{tag}.png", dpi=160)
    plt.show()
    plt.close(fig)

def plot_zoom(index, y, alarm, eval_mask, lead_idx, n_alarms_test, tag):

    m = (index >= 4000) & (index <= 4500)

    fig, ax = plt.subplots(figsize=(16, 5))

    in_anom = (index >= ANOM_I0) & (index <= ANOM_I1)
    y_norm = y.copy()
    y_anom = np.full_like(y, np.nan)
    y_anom[in_anom] = y[in_anom]
    y_norm[in_anom] = np.nan

    ax.plot(index[m], y_norm[m], label="Real (normal)")
    ax.plot(index[m], y_anom[m], label="Real (anómalo)", color="red", linewidth=2)

    ax.scatter(
        index[m][alarm[m] & eval_mask[m]],
        y[m][alarm[m] & eval_mask[m]],
        color="red",
        s=30,
        label=f"Alarmas TEST: {n_alarms_test}"  # <<< NUEVO >>>
    )

    ax.axvspan(ANOM_I0 - W_EARLY, ANOM_I0, color="orange", alpha=0.25)
    ax.axvspan(ANOM_I0, ANOM_I1, color="red", alpha=0.2)

    if lead_idx is not None and 4000 <= lead_idx <= 4500:
        ax.axvline(lead_idx, color="green", linestyle=":", linewidth=2)

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"zoom_{tag}_4000_4500.png", dpi=160)
    plt.show()
    plt.close(fig)

# =========================================================
# MAIN
# =========================================================

def main():

    model_name = (
        f"model_{DATASET_NAME}"
        f"_k{k:.2f}"
        f"_MAX{MAXv}"
        f"_WS{WS}"
        f"_H{H}"
        f"_BS{BS}.h5"
    )

    model_path = CACHE_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"[MODEL] Cargado: {model_path.name}")

    y = load_series(SRC_TXT)
    n = len(y)
    split_raw = split_index_by_frac(n, TRAIN_FRAC)

    ymin, ymax = y[:split_raw].min(), y[:split_raw].max()
    y_s = (y - ymin) / (ymax - ymin + 1e-8)

    pred_s = model_forecast(model, y_s, WS)

    start_idx = WS + H - 1
    L = min(len(pred_s), len(y) - start_idx)

    index = np.arange(start_idx, start_idx + L)
    y_al = y[start_idx:start_idx + L]

    pred = pred_s[:L] * (ymax - ymin + 1e-8) + ymin
    pred[:-H] = pred[H:]
    pred[-H:] = np.nan

    err = np.abs(y_al - pred)

    split_idx = max(0, split_raw - start_idx)
    mu, sigma = np.nanmean(err[:split_idx]), np.nanstd(err[:split_idx])

    alarm = np.abs(err - mu) > k * sigma

    valid = ~np.isnan(err)
    eval_mask = valid & (np.arange(L) >= split_idx)

    gt = (index >= (ANOM_I0 - W_EARLY)) & (index <= ANOM_I1)

    TP = np.sum(alarm & gt & eval_mask)
    FP = np.sum(alarm & ~gt & eval_mask)
    FN = np.sum(~alarm & gt & eval_mask)
    TN = np.sum(~alarm & ~gt & eval_mask)

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F01 = fbeta(P, R)

    n_alarms_test = int(np.sum(alarm & eval_mask))  # <<< NUEVO >>>

    print("\nDataset | Precision | Recall | F0.1-score")
    print("------------------------------------------")
    print(f"Paper   | {PAPER_PREC:.2f}      | {PAPER_REC:.2f}  | {PAPER_F01:.2f}")
    print(f"{DATASET_NAME} | {P:.3f}     | {R:.3f} | {F01:.3f}")

    lead_idx = index[np.where(alarm & gt)[0][0]] if np.any(alarm & gt) else None

    plot_global(
        index, y_al, pred, err, alarm,
        eval_mask, split_idx, mu, sigma, k,
        lead_idx, n_alarms_test, TAG
    )

    plot_zoom(
        index, y_al, alarm,
        eval_mask, lead_idx, n_alarms_test, TAG
    )

    print(f"\n[OK] Alarmas en TEST: {n_alarms_test}")

if __name__ == "__main__":
    main()
