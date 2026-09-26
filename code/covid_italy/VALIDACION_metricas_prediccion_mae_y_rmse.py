"""
AQUI SE CALCULAN LAS MÉTRICAS MAE Y RMSE PARA LA COMBINACIÓN GANADORA DE COVID ITALIA
"""
# ============================================================
# VALIDACION_6_Metricas_regresion_mejor_config_marche.py
# ------------------------------------------------------------
# Calcula métricas de regresión (MAE, RMSE, MAPE) para la
# configuración ganadora obtenida con validación en Marche.
#
# Configuración fija:
#   model      = covid19_WS10_H1_BS8.h5
#   ws         = 10
#   h          = 1
#   bs         = 8
#   window_thr = 5
#   factor     = 0.8
#   k          = 2.0
#
# NOTA:
# - Las métricas MAE/RMSE/MAPE se calculan entre serie real
#   y predicción del modelo.
# - El detector no interviene en MAE/RMSE/MAPE, pero se deja
#   documentada la configuración completa usada.
#
# Salidas:
#   ./metricas_csv/metricas_regresion_mejor_config_marche.csv
#   ./metricas_xlsx/metricas_regresion_mejor_config_marche.xlsx
# ============================================================

import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# RUTAS
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../VALIDACION
ROOT_DIR = os.path.dirname(THIS_DIR)                    # raíz

DATASET_FILE = os.path.join(ROOT_DIR, "dpc-covid19-ita-regioni.txt")
MODELS_DIR   = os.path.join(ROOT_DIR, "modelos")

OUT_CSV  = os.path.join(THIS_DIR, "metricas_csv")
OUT_XLSX = os.path.join(THIS_DIR, "metricas_xlsx")

os.makedirs(OUT_CSV, exist_ok=True)
os.makedirs(OUT_XLSX, exist_ok=True)

# ============================================================
# CONFIGURACIÓN FIJA (GANADORA EN MARCHE)
# ============================================================

MODEL_NAME = "covid19_WS7_H1_BS32.h5"
WINNER_WS = 19
WINNER_H  = 1
WINNER_BS = 32

WINDOW_THR = 10
FACTOR     = 1.2
K_VALUE    = 1.5

TARGET_COL = "nuovi_positivi"

DATE_START = "2020-02-24"
DATE_END   = "2020-05-15"

PRED_BATCH = 1024

REGIONS = ["Marche", "Lazio", "Campania", "Sicilia"]

COVID_START_BY_REGION = {
    "Marche":   pd.to_datetime("2020-03-05"),
    "Lazio":    pd.to_datetime("2020-03-05"),
    "Campania": pd.to_datetime("2020-03-08"),
    "Sicilia":  pd.to_datetime("2020-03-10"),
}

# ============================================================
# UTILIDADES
# ============================================================

def clean_and_get_date_col(df: pd.DataFrame):
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    if "data" in df.columns:
        return df, "data"
    elif "date" in df.columns:
        return df, "date"
    else:
        raise ValueError("No date column found")

def parse_model_name(name: str):
    ws = int(re.search(r"_WS(\d+)", name).group(1))
    h  = int(re.search(r"_H(\d+)", name).group(1))
    bs = int(re.search(r"_BS(\d+)", name).group(1))
    return ws, h, bs

def build_forecast_full(model, y: np.ndarray, ws: int):
    """
    Forecast clásico alineado:
      X[i] = y[i:i+ws]
      y_true = y[ws:]
      y_pred = model.predict(X)
    """
    X = []
    for i in range(len(y) - ws):
        X.append(y[i:i+ws])

    X = np.asarray(X, dtype=np.float32).reshape(-1, ws, 1)

    y_pred = model.predict(X, batch_size=PRED_BATCH, verbose=0).flatten()
    y_true = y[ws:]
    return y_true, y_pred

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-12
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("MÉTRICAS DE REGRESIÓN · MEJOR CONFIGURACIÓN MARCHE")
    print("=" * 100)
    print(f"Modelo      : {MODEL_NAME}")
    print(f"WS          : {WINNER_WS}")
    print(f"H           : {WINNER_H}")
    print(f"BS          : {WINNER_BS}")
    print(f"window_thr  : {WINDOW_THR}")
    print(f"factor      : {FACTOR}")
    print(f"k           : {K_VALUE}")
    print("=" * 100)

    # --------------------------------------------------------
    # 1) DATASET
    # --------------------------------------------------------
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

    missing = [r for r in REGIONS if r not in pivot.columns]
    if missing:
        raise ValueError(f"Faltan regiones en el dataset: {missing}")

    # --------------------------------------------------------
    # 2) MODELO
    # --------------------------------------------------------
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encuentra el modelo: {model_path}")

    model = load_model(model_path)
    ws, h, bs = parse_model_name(MODEL_NAME)

    print(f"[OK] Modelo cargado: {model_path}")
    print(f"[OK] Dataset cargado: {pivot.index.min().date()} -> {pivot.index.max().date()}")

    rows = []

    # --------------------------------------------------------
    # 3) MÉTRICAS POR REGIÓN
    # --------------------------------------------------------
    for region in REGIONS:

        y_raw = pivot[region].to_numpy(dtype=float)
        dates = pivot.index.to_numpy()

        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()

        y_true_all, y_pred_all = build_forecast_full(model, y_scaled, ws)
        dates_f = dates[ws:]

        # --- métricas en escala normalizada, serie completa alineada ---
        met_all_norm = regression_metrics(y_true_all, y_pred_all)

        # --- desnormalizar para escala original ---
        y_true_all_orig = scaler.inverse_transform(y_true_all.reshape(-1, 1)).flatten()
        y_pred_all_orig = scaler.inverse_transform(y_pred_all.reshape(-1, 1)).flatten()

        met_all_orig = regression_metrics(y_true_all_orig, y_pred_all_orig)

        # --- métricas desde inicio COVID de la región ---
        covid_start = COVID_START_BY_REGION[region]
        mask_covid = dates_f >= covid_start

        y_true_covid = y_true_all[mask_covid]
        y_pred_covid = y_pred_all[mask_covid]

        y_true_covid_orig = y_true_all_orig[mask_covid]
        y_pred_covid_orig = y_pred_all_orig[mask_covid]

        met_covid_norm = regression_metrics(y_true_covid, y_pred_covid)
        met_covid_orig = regression_metrics(y_true_covid_orig, y_pred_covid_orig)

        rows.append({
            "region": region,
            "covid_start": covid_start,
            "model": MODEL_NAME,
            "ws": ws,
            "h": h,
            "bs": bs,
            "window_thr": WINDOW_THR,
            "factor": FACTOR,
            "k": K_VALUE,

            # Serie completa
            "MAE_all_norm": met_all_norm["MAE"],
            "RMSE_all_norm": met_all_norm["RMSE"],
            "MAPE_all_norm": met_all_norm["MAPE"],

            "MAE_all_orig": met_all_orig["MAE"],
            "RMSE_all_orig": met_all_orig["RMSE"],
            "MAPE_all_orig": met_all_orig["MAPE"],

            # Solo desde inicio COVID
            "MAE_covid_norm": met_covid_norm["MAE"],
            "RMSE_covid_norm": met_covid_norm["RMSE"],
            "MAPE_covid_norm": met_covid_norm["MAPE"],

            "MAE_covid_orig": met_covid_orig["MAE"],
            "RMSE_covid_orig": met_covid_orig["RMSE"],
            "MAPE_covid_orig": met_covid_orig["MAPE"],
        })

        print(
            f"[{region}] "
            f"RMSE_all_orig={met_all_orig['RMSE']:.4f} | "
            f"RMSE_covid_orig={met_covid_orig['RMSE']:.4f}"
        )

    # --------------------------------------------------------
    # 4) GUARDAR
    # --------------------------------------------------------
    df_metrics = pd.DataFrame(rows)

    csv_path  = os.path.join(OUT_CSV,  "metricas_regresion_mejor_config_marche.csv")
    xlsx_path = os.path.join(OUT_XLSX, "metricas_regresion_mejor_config_marche.xlsx")

    df_metrics.to_csv(csv_path, index=False)
    df_metrics.to_excel(xlsx_path, index=False)

    print("\n" + "=" * 100)
    print("[OK] Archivos generados:")
    print(" -", csv_path)
    print(" -", xlsx_path)
    print("=" * 100)

    print("\n[RESUMEN]")
    print(df_metrics.to_string(index=False))

if __name__ == "__main__":
    main()