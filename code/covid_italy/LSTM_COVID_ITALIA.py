# -*- coding: utf-8 -*-
"""
LSTM_COVID_ITALIA.py
------------------------------------------------

Barrido sistemático de modelos Conv1D-LSTM
para COVID Italia (dataset PCM-DPC por regiones)

Barrido:
- WINDOW_SIZE (ws)
- HORIZON (h)
- BATCH_SIZE (bs)

Cada combinación genera:
- Un modelo persistente en ./modelos
- Si el modelo ya existe → se carga, no se reentrena
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dropout, Dense

# ---------------------------------------------------------------------------
# 0) CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------------
SRC = "dpc-covid19-ita-regioni.txt"
TARGET_COL = "nuovi_positivi"

DATE_START = "2020-02-24"
DATE_END   = "2020-05-15"

EPOCHS = 100

# 🔁 BARRIDO DE HIPERPARÁMETROS
WS_LIST = [5, 7, 10]
H_LIST  = [1, 3]
BS_LIST = [8, 16, 32]

# Regiones (paper)
TRAIN_REGIONS_PAPER = [
    "P.A. Bolzano", "Emilia-Romagna", "Liguria", "Lombardia",
    "Piemonte", "P.A. Trento", "Valle d'Aosta",
    "Veneto", "Friuli Venezia Giulia"
]
VAL_REGIONS_PAPER  = ["Marche"]
TEST_REGIONS_PAPER = ["Lazio", "Campania", "Sicilia"]

# Carpeta de modelos
MODEL_DIR = "modelos"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1) HELPERS
# ---------------------------------------------------------------------------
def robust_read_csv(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df

def resolve_region_name(region_paper, available_regions):
    rp = region_paper.lower()
    for r in available_regions:
        if rp in r.lower():
            return r
    return None

def resolve_regions(region_list, available_regions):
    resolved, missing = [], []
    for r in region_list:
        rr = resolve_region_name(r, available_regions)
        if rr is not None:
            resolved.append(rr)
        else:
            missing.append(r)
    return resolved, missing

def windowed_dataset(series, window, horizon):
    s = tf.convert_to_tensor(series, tf.float32)
    s = tf.expand_dims(s, -1)
    ds = tf.data.Dataset.from_tensor_slices(s)
    ds = ds.window(window + horizon, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window + horizon))
    ds = ds.map(lambda w: (w[:window], w[window + horizon - 1]))
    return ds

def build_model():
    model = Sequential([
        Conv1D(64, 5, padding="causal", activation="relu",
               input_shape=[None, 1]),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )
    return model

# ---------------------------------------------------------------------------
# 2) CARGA Y PREPROCESADO DEL DATASET (UNA SOLA VEZ)
# ---------------------------------------------------------------------------
df = robust_read_csv(SRC)
df["data"] = pd.to_datetime(df["data"]).dt.normalize()

df = df[
    (df["data"] >= pd.Timestamp(DATE_START)) &
    (df["data"] <= pd.Timestamp(DATE_END))
]

pivot = df.pivot_table(
    index="data",
    columns="denominazione_regione",
    values=TARGET_COL,
    aggfunc="sum"
).sort_index()

pivot = pivot.reindex(
    pd.date_range(pivot.index.min(), pivot.index.max(), freq="D")
).fillna(0.0)

available_regions = list(pivot.columns.astype(str))

TRAIN_REGIONS, MISS_TRAIN = resolve_regions(TRAIN_REGIONS_PAPER, available_regions)
VAL_REGIONS,   MISS_VAL   = resolve_regions(VAL_REGIONS_PAPER, available_regions)
TEST_REGIONS,  MISS_TEST  = resolve_regions(TEST_REGIONS_PAPER, available_regions)

if not TRAIN_REGIONS:
    raise ValueError("❌ No hay regiones TRAIN válidas")

# Escalado SOLO con TRAIN
train_vals = pivot[TRAIN_REGIONS].to_numpy().reshape(-1)
vmin, vmax = train_vals.min(), train_vals.max()
eps = 1e-8

def scale(x):
    return (x - vmin) / (vmax - vmin + eps)

pivot_scaled = pivot.apply(scale)

# ---------------------------------------------------------------------------
# 3) BARRIDO DE MODELOS
# ---------------------------------------------------------------------------
print("\n🚀 INICIANDO BARRIDO DE MODELOS\n")

for WS in WS_LIST:
    for H in H_LIST:
        for BS in BS_LIST:

            model_name = f"covid19_WS{WS}_H{H}_BS{BS}.h5"
            model_path = os.path.join(MODEL_DIR, model_name)

            print("\n----------------------------------------")
            print(f"Modelo: WS={WS} | H={H} | BS={BS}")

            if os.path.exists(model_path):
                print(f"✅ Ya existe → cargando {model_name}")
                _ = load_model(model_path)
                continue

            print("🚧 Entrenando modelo...")

            # Dataset multi-región
            ds_all = None
            for r in TRAIN_REGIONS:
                ds_r = windowed_dataset(
                    pivot_scaled[r].values,
                    window=WS,
                    horizon=H
                )
                ds_all = ds_r if ds_all is None else ds_all.concatenate(ds_r)

            ds_all = (
                ds_all
                .shuffle(1000)
                .batch(BS)
                .prefetch(tf.data.AUTOTUNE)
            )

            model = build_model()
            model.fit(ds_all, epochs=EPOCHS, verbose=1)

            model.save(model_path)
            print(f"💾 Modelo guardado en {model_path}")

print("\n✔ BARRIDO COMPLETADO")
