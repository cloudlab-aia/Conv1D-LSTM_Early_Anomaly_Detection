# ============================================================
# VALIDACION-graficas_mejores_resultados.py
# ------------------------------------------------------------
# Grafica la configuración seleccionada:
#   model      = covid19_WS7_H1_BS32.h5
#   window_thr = 10
#   factor     = 1.2
#   k          = 1.5
#
# Dibuja en las 3 regiones de test:
#   - Lazio
#   - Campania
#   - Sicilia
#
# mostrando:
#   - serie real
#   - predicción LSTM
#   - alarmas detectadas
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# RUTAS
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../VALIDACION
ROOT_DIR = os.path.dirname(THIS_DIR)                    # raíz

DATASET_FILE = os.path.join(ROOT_DIR, "dpc-covid19-ita-regioni.txt")
MODELS_DIR   = os.path.join(ROOT_DIR, "modelos")

OUT_DIR = os.path.join(THIS_DIR, "Imagenes_mejores_resultados")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

TARGET_COL = "nuovi_positivi"
DATE_START = "2020-02-24"
DATE_END   = "2020-05-15"
PRED_BATCH = 1024

# ============================================================
# CONFIGURACIÓN SELECCIONADA
# ============================================================

CONFIG = {
    "model": "covid19_WS7_H1_BS32.h5",
    "window_thr": 10,
    "factor": 1.2,
    "k": 1.5
}

REGIONS = ["Lazio", "Campania", "Sicilia"]

COVID_START_BY_REGION = {
    "Lazio":    pd.to_datetime("2020-03-05"),
    "Campania": pd.to_datetime("2020-03-08"),
    "Sicilia":  pd.to_datetime("2020-03-10"),
}

# ============================================================
# UTILIDADES
# ============================================================

def clean_and_get_date_col(df):
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    if "data" in df.columns:
        return df, "data"
    raise ValueError("No date column found")

def parse_ws(name):
    return int(re.search(r"_WS(\d+)", name).group(1))

def build_forecast(model, y, ws):
    X = []
    for i in range(len(y) - ws):
        X.append(y[i:i+ws])
    X = np.array(X, dtype=np.float32).reshape(-1, ws, 1)

    pred = model.predict(X, batch_size=PRED_BATCH, verbose=0).flatten()
    y_true = y[ws:]
    err = np.abs(y_true - pred)
    return y_true, pred, err

def compute_threshold_full(err_full, window_thr, factor, k):
    s = pd.Series(err_full)
    mu = s.rolling(window=window_thr, min_periods=1).mean().to_numpy()
    sd = s.rolling(window=window_thr, min_periods=1).std().fillna(0.0).to_numpy()
    thr = factor * (mu + k * sd)
    alarm = err_full > thr
    return mu, sd, thr, alarm

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("GRAFICANDO CONFIGURACIÓN SELECCIONADA")
    print("=" * 90)
    print(f"Modelo   : {CONFIG['model']}")
    print(f"N        : {CONFIG['window_thr']}")
    print(f"factor   : {CONFIG['factor']}")
    print(f"k        : {CONFIG['k']}")
    print("=" * 90)

    # ---------- DATASET ----------
    df = pd.read_csv(DATASET_FILE, sep=None, engine="python")
    df, date_col = clean_and_get_date_col(df)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    df = df[(df[date_col] >= DATE_START) & (df[date_col] <= DATE_END)]

    pivot = (
        df.pivot_table(
            index=date_col,
            columns="denominazione_regione",
            values=TARGET_COL,
            aggfunc="sum"
        )
        .sort_index()
        .fillna(0.0)
    )

    # ---------- MODELO ----------
    model_path = os.path.join(MODELS_DIR, CONFIG["model"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encuentra el modelo: {model_path}")

    model = load_model(model_path)
    ws = parse_ws(CONFIG["model"])

    # ---------- FIGURA ÚNICA ----------
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    for ax, region in zip(axes, REGIONS):

        y_raw = pivot[region].to_numpy(dtype=float)
        dates = pivot.index.to_numpy()

        scaler = MinMaxScaler()
        y = scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()

        y_true, y_pred, err = build_forecast(model, y, ws)
        dates_f = dates[ws:]

        _, _, _, alarm = compute_threshold_full(
            err_full=err,
            window_thr=CONFIG["window_thr"],
            factor=CONFIG["factor"],
            k=CONFIG["k"]
        )

        covid_start = COVID_START_BY_REGION[region]
        eval_mask = dates_f >= covid_start

        dates_eval = dates_f[eval_mask]
        y_true_eval = y_true[eval_mask]
        alarm_eval = alarm[eval_mask]

        if np.any(alarm_eval):
            first_alarm_date = pd.to_datetime(dates_eval[np.where(alarm_eval)[0][0]])
            first_alarm_str = first_alarm_date.strftime("%Y-%m-%d")
        else:
            first_alarm_str = "No detectada"

        # --- GRÁFICA ---
        ax.plot(dates_f, y_true, label="Serie real", color="blue", linewidth=1.5)
        ax.plot(dates_f, y_pred, label="Predicción LSTM", color="orange", linewidth=1.5)

        ax.scatter(
            dates_eval[alarm_eval],
            y_true_eval[alarm_eval],
            color="red",
            s=35,
            label="Alarma",
            zorder=5
        )

        ax.set_title(f"{region}. Nº de nuevos casos positivos")
        ax.grid(alpha=0.3)

        ax.text(
            0.01, -0.26,
            f"Primera alarma detectada: {first_alarm_str}",
            transform=ax.transAxes,
            fontsize=10
        )

        ax.legend(loc="upper left")

        print(
            f"[{region}] primera alarma = {first_alarm_str} | "
            f"num_alarmas = {int(np.sum(alarm_eval))}"
        )

    plt.tight_layout()
    out_file = os.path.join(
        OUT_DIR,
        "forecast_config_equilibrada_WS7_H1_BS32_N10_factor1p2_k1p5.png"
    )
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Imagen generada: {out_file}")

if __name__ == "__main__":
    main()