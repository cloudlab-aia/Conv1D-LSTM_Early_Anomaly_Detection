# -*- coding: utf-8 -*-
"""
LSTM_SHUTTLE_ESTATICO_BARRIDO.py
ESTE CÓDIGO REALIZA UN BARRIDO PARA GENERAR MÚLTIPLES MODELOS
EN BASE A WS, HORIZON y BS, Y LUEGO PARA CADA MODELO
GENERA GRÁFICAS EN BASE A MAX < X ALARMAS EN TRAIN
------------------------------------------------------------
Space Shuttle – Umbral FIJO global (tipo Jorge)
Barrido WS / H / BS / MAX con detección anticipada y zoom
------------------------------------------------------------
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# CONFIG GENERAL
# =========================================================

SRC_TXT = "tek16.txt"
DATASET_NAME = Path(SRC_TXT).stem

TRAIN_FRAC = 0.70

EPOCHS = 40
LR = 1e-4
SHUFFLE_BUFFER = 2000
EARLY_STOP_PATIENCE = 6
EARLY_STOP_MIN_DELTA = 1e-5

# Umbral fijo global
K_MIN = 7
K_MAX = 12.0
K_STEP = 0.1

# >>> DIMENSIÓN EXPERIMENTAL: MAX<=X alarmas en TRAIN
MAX_LIST = [0, 2, 4, 6]

# Anomalía (ground truth)
ANOM_I0 = 4200
ANOM_I1 = 4380
ANTICIP_WIN = 100

# Zoom
ZOOM_I0 = 4000
ZOOM_I1 = 4500

# GRID EXPERIMENTAL
WS_LIST = [128, 256]
H_LIST  = [40, 45, 50, 55, 60]
BS_LIST = [16, 32]

# Salidas
CACHE_DIR = Path("LSTM_SHUTTLE_ESTATICO")
FIG_DIR = CACHE_DIR / "figs_estatico"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAVE_FIGS = True
SHOW_FIGS = False

# =========================================================
# HELPERS
# =========================================================

def load_series(path):
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for p in line.replace("\t", " ").split():
                try:
                    values.append(float(p))
                except ValueError:
                    pass
    if len(values) == 0:
        raise ValueError("Dataset vacío o mal leído")
    return np.asarray(values, dtype=float)

def split_index_by_frac(n, frac):
    return int(np.floor(n * frac))

def windowed_dataset(series, w, h, bs, shuffle):
    s = tf.expand_dims(tf.convert_to_tensor(series, tf.float32), -1)
    ds = tf.data.Dataset.from_tensor_slices(s)
    ds = ds.window(w + h, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x.batch(w + h))
    ds = ds.shuffle(shuffle)
    ds = ds.map(lambda x: (x[:w], x[w + h - 1]))
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

def build_model(lr):
    m = Sequential([
        Conv1D(12, 5, padding="causal", activation="relu"),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    m.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr))
    return m

def model_forecast(model, series, w):
    s = tf.expand_dims(tf.convert_to_tensor(series, tf.float32), -1)
    ds = tf.data.Dataset.from_tensor_slices(s)
    ds = ds.window(w, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x.batch(w))
    return model.predict(ds.batch(32), verbose=0).squeeze()

def sweep_k_global(err, split_idx, max_alarms_train):
    err_train = err[:split_idx]
    mu = np.nanmean(err_train)
    sigma = np.nanstd(err_train)

    bestk = None
    for k in np.arange(K_MIN, K_MAX + 1e-9, K_STEP):
        alarm = np.abs(err - mu) > k * sigma
        if np.nansum(alarm[:split_idx]) <= max_alarms_train:
            bestk = float(k)
            break
    return bestk, mu, sigma

def first_tp_alarm(index_al, alarm):
    m = (index_al >= ANOM_I0 - ANTICIP_WIN) & (index_al < ANOM_I0)
    cand = np.where(alarm & m)[0]
    if len(cand) == 0:
        return None, None
    idx = index_al[cand[0]]
    return idx, ANOM_I0 - idx

def save_fig(fig, path):
    if SAVE_FIGS:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)

# =========================================================
# PLOTS
# =========================================================

def plot_global(index_al, y_al, pred, err, alarm,
                split_idx, mu, sigma, bestk,
                idx_tp, anticip, tag):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    in_anom = (index_al >= ANOM_I0) & (index_al <= ANOM_I1)
    y_norm = y_al.copy()
    y_anom = np.full_like(y_al, np.nan)
    y_anom[in_anom] = y_al[in_anom]
    y_norm[in_anom] = np.nan

    ax1.plot(index_al, y_norm, label="Real (normal)", linewidth=1.2)
    ax1.plot(index_al, y_anom, label="Real (anómalo)", color="red", linewidth=2.2)
    ax1.plot(index_al, pred, label="Pred LSTM", linewidth=1.1)

    ax1.scatter(index_al[alarm], y_al[alarm], s=22,
                color="red", edgecolors="black", label="Alarma", zorder=5)

    if idx_tp is not None:
        ax1.axvline(idx_tp, color="green", linestyle=":",
                    linewidth=1.5,
                    label=f"Anticipación: {anticip} muestras")

    ax1.axvline(index_al[split_idx], color="black", linestyle="--",
                alpha=0.6, label="Split TRAIN/TEST")

    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(tag)

    ax2.plot(index_al, err, label="|error|")
    ax2.axhline(mu, linestyle="--", label="μ_train")
    ax2.axhline(mu + bestk * sigma, linestyle=":", label="μ + k·σ")

    ax2.scatter(index_al[alarm], err[alarm], s=22,
                color="red", edgecolors="black", label="Alarma")

    if idx_tp is not None:
        ax2.axvline(idx_tp, color="green", linestyle=":", linewidth=1.5)

    ax2.axvline(index_al[split_idx], color="black", linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(True, alpha=0.3)

    ax2.set_xlabel("Índice")
    ax2.set_ylabel("Error")

    plt.tight_layout()
    save_fig(fig, FIG_DIR / f"global_{tag}.png")

def plot_zoom(index_al, y_al, alarm, idx_tp, anticip, tag):

    m = (index_al >= ZOOM_I0) & (index_al <= ZOOM_I1)
    iz = index_al[m]
    yz = y_al[m]
    az = alarm[m]

    in_anom = (iz >= ANOM_I0) & (iz <= ANOM_I1)
    y_norm = yz.copy()
    y_anom = np.full_like(yz, np.nan)
    y_anom[in_anom] = yz[in_anom]
    y_norm[in_anom] = np.nan

    n_alarms = int(np.sum(az))

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(iz, y_norm, label="Real (normal)", linewidth=1.4)
    ax.plot(iz, y_anom, label="Real (anómalo)", color="red", linewidth=2.4)

    if n_alarms > 0:
        ax.scatter(iz[az], yz[az], s=35,
                   color="red", edgecolors="black",
                   label=f"Alarmas: {n_alarms}")

    if idx_tp is not None and ZOOM_I0 <= idx_tp <= ZOOM_I1:
        ax.axvline(idx_tp, color="green", linestyle=":",
                   linewidth=2,
                   label=f"Anticipación: {anticip} muestras")

    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Zoom anomalía [{ZOOM_I0}, {ZOOM_I1}]")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Valor")

    plt.tight_layout()
    save_fig(fig, FIG_DIR / f"zoom_{tag}_{ZOOM_I0}_{ZOOM_I1}.png")

# =========================================================
# EXPERIMENTO
# =========================================================

def run_experiment(WS, H, BS, MAX):

    print(f"\n=== DATASET={DATASET_NAME} | WS={WS} | H={H} | BS={BS} | MAX={MAX} ===")

    y = load_series(SRC_TXT)
    n = len(y)
    split_raw = split_index_by_frac(n, TRAIN_FRAC)

    x = np.arange(n)

    y_train = y[:split_raw]
    ymin, ymax = y_train.min(), y_train.max()
    y_s = (y - ymin) / (ymax - ymin + 1e-8)

    train_ds = windowed_dataset(y_s[:split_raw], WS, H, BS, SHUFFLE_BUFFER)

    tag_exp = f"WS{WS}_H{H}_BS{BS}"

    # Entrenamiento / carga base (independiente de MAX)
    base_model_path = CACHE_DIR / f"model_{DATASET_NAME}_{tag_exp}.h5"
    if base_model_path.exists():
        model = tf.keras.models.load_model(base_model_path)
        print(f"[MODEL] Base cargado: {base_model_path.name}")
    else:
        model = build_model(LR)
        es = EarlyStopping("loss", patience=EARLY_STOP_PATIENCE,
                           min_delta=EARLY_STOP_MIN_DELTA,
                           restore_best_weights=True)
        model.fit(train_ds, epochs=EPOCHS, callbacks=[es], verbose=0)
        model.save(base_model_path)
        print(f"[MODEL] Base entrenado y guardado: {base_model_path.name}")

    pred_s = model_forecast(model, y_s, WS)

    start_idx = WS + H - 1
    L = min(len(pred_s), len(y) - start_idx)
    if L <= 10:
        print("[WARN] Serie demasiado corta")
        return

    index_al = x[start_idx:start_idx + L]
    y_al = y[start_idx:start_idx + L]

    pred = pred_s[:L] * (ymax - ymin + 1e-8) + ymin
    pred[:-H] = pred[H:]
    pred[-H:] = np.nan

    split_idx = max(0, split_raw - start_idx)
    err = np.abs(y_al - pred)

    bestk, mu, sigma = sweep_k_global(err, split_idx, MAX)
    if bestk is None:
        bestk = K_MAX

    alarm = np.abs(err - mu) > bestk * sigma
    idx_tp, anticip = first_tp_alarm(index_al, alarm)

    tag = f"{DATASET_NAME}_k{bestk:.2f}_MAX{MAX}_{tag_exp}"

    # Guardar modelo final etiquetado con MAX y k
    final_model_path = CACHE_DIR / f"model_{tag}.h5"
    if not final_model_path.exists():
        model.save(final_model_path)

    plot_global(index_al, y_al, pred, err, alarm,
                split_idx, mu, sigma, bestk,
                idx_tp, anticip, tag)

    plot_zoom(index_al, y_al, alarm, idx_tp, anticip, tag)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    for WS in WS_LIST:
        for H in H_LIST:
            for BS in BS_LIST:
                for MAX in MAX_LIST:
                    run_experiment(WS, H, BS, MAX)
