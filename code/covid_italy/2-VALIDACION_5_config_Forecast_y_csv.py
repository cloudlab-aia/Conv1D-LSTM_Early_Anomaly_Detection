# ============================================================
# VALIDACION_5_Todas_las_regiones_norte_como_validacion.py
# ------------------------------------------------------------
# Repite el experimento usando como validación, una a una,
# todas las regiones del norte.
#
# Mantiene la lógica del script bueno:
# - contexto previo para el umbral dinámico
# - selección por validación
# - test en Lazio, Campania y Sicilia
#
# Salidas:
#   ./5_config_csv/resultados_todas_validaciones.csv
#   ./5_config_xlsx/resultados_todas_validaciones.xlsx
#   ./5_config_csv/resumen_todas_validaciones.csv
#   ./5_config_xlsx/resumen_todas_validaciones.xlsx
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

OUT_CSV  = os.path.join(THIS_DIR, "5_config_csv")
OUT_XLSX = os.path.join(THIS_DIR, "5_config_xlsx")

os.makedirs(OUT_CSV, exist_ok=True)
os.makedirs(OUT_XLSX, exist_ok=True)

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

TARGET_COL = "nuovi_positivi"

VALID_REGIONS = [
    "P.A. Bolzano",
    "Emilia-Romagna",
    "Liguria",
    "Lombardia",
    "Piemonte",
    "P.A. Trento",
    "Valle d'Aosta",
    "Veneto",
    "Friuli Venezia Giulia",
    "Marche",
]

TEST_REGIONS = ["Lazio", "Campania", "Sicilia"]

DATE_START = "2020-02-24"
DATE_END   = "2020-05-15"

WINDOW_THR_LIST = [5, 7, 10]
FACTOR_LIST     = [0.8, 1.0, 1.2]
K_LIST          = [1.0, 1.5, 2.0]

PRED_BATCH = 1024

COVID_START_BY_REGION = {
    "Marche":   pd.to_datetime("2020-03-05"),
    "Lazio":    pd.to_datetime("2020-03-05"),
    "Campania": pd.to_datetime("2020-03-08"),
    "Sicilia":  pd.to_datetime("2020-03-10"),
}

DEFAULT_VALID_START = pd.to_datetime("2020-03-05")

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
    elif "date" in df.columns:
        return df, "date"
    else:
        raise ValueError("No date column found")

def parse_model_name(name):
    ws = int(re.search(r"_WS(\d+)", name).group(1))
    h  = int(re.search(r"_H(\d+)", name).group(1))
    bs = int(re.search(r"_BS(\d+)", name).group(1))
    return ws, h, bs

def build_forecast_full(model, y, ws):
    X = []
    for i in range(len(y) - ws):
        X.append(y[i:i+ws])

    X = np.asarray(X, dtype=np.float32).reshape(-1, ws, 1)

    y_pred = model.predict(X, batch_size=PRED_BATCH, verbose=0).flatten()
    y_true = y[ws:]
    err = np.abs(y_true - y_pred)

    return y_true, y_pred, err

def compute_threshold_full(err_full, window_thr, factor, k):
    s = pd.Series(err_full)
    mu = s.rolling(window=window_thr, min_periods=1).mean().to_numpy()
    sd = s.rolling(window=window_thr, min_periods=1).std().fillna(0.0).to_numpy()
    thr = factor * (mu + k * sd)
    alarm = err_full > thr
    return mu, sd, thr, alarm

def get_region_start(region_name):
    return COVID_START_BY_REGION.get(region_name, DEFAULT_VALID_START)

def evaluate_from_start(err_dates, err_full, thr_full, alarm_full, region_name):
    region_start = get_region_start(region_name)

    zone_mask = err_dates >= region_start

    dates_zone = err_dates[zone_mask]
    err_zone   = err_full[zone_mask]
    thr_zone   = thr_full[zone_mask]
    alarm_zone = alarm_full[zone_mask]

    num_alarms = int(np.sum(alarm_zone))

    if np.any(alarm_zone):
        first_idx = np.where(alarm_zone)[0][0]
        first_alarm_date = pd.to_datetime(dates_zone[first_idx])
        detection_delay_days = int((first_alarm_date - region_start).days)
    else:
        first_alarm_date = pd.NaT
        detection_delay_days = np.nan

    return {
        "region_start": region_start,
        "dates_zone": dates_zone,
        "err_zone": err_zone,
        "thr_zone": thr_zone,
        "alarm_zone": alarm_zone,
        "num_alarms": num_alarms,
        "first_alarm_date": first_alarm_date,
        "detection_delay_days": detection_delay_days,
    }

def rank_tuple(delay_days, num_alarms, ws, h, bs, window_thr, factor, k):
    delay_rank = delay_days if pd.notna(delay_days) else 10**9
    return (
        delay_rank,
        num_alarms,
        ws,
        h,
        bs,
        window_thr,
        factor,
        k
    )

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("COVID ITALIA · TODAS LAS REGIONES DEL NORTE COMO VALIDACIÓN")
    print("=" * 100)

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

    needed_regions = VALID_REGIONS + TEST_REGIONS
    missing_regions = [r for r in needed_regions if r not in pivot.columns]
    if missing_regions:
        raise ValueError(f"Faltan regiones en el dataset: {missing_regions}")

    model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".h5")])
    if not model_files:
        raise ValueError(f"No se han encontrado modelos .h5 en {MODELS_DIR}")

    print(f"[OK] Dataset cargado")
    print(f"[OK] Modelos encontrados: {len(model_files)}")

    all_rows = []
    summary_rows = []

    for valid_region in VALID_REGIONS:

        print("\n" + "#" * 100)
        print(f"[VALIDACIÓN] Región actual: {valid_region}")
        print("#" * 100)

        region_results = []

        for model_idx, model_file in enumerate(model_files, start=1):

            ws, h, bs = parse_model_name(model_file)
            model_path = os.path.join(MODELS_DIR, model_file)

            print(f"\n[MODELO {model_idx}/{len(model_files)}] {model_file} | WS={ws} | H={h} | BS={bs}")

            model = load_model(model_path)

            y_valid_raw = pivot[valid_region].to_numpy(dtype=float)
            dates_valid = pivot.index.to_numpy()

            scaler_valid = MinMaxScaler()
            y_valid = scaler_valid.fit_transform(y_valid_raw.reshape(-1, 1)).flatten()

            _, _, err_valid_full = build_forecast_full(model, y_valid, ws)
            err_valid_dates = dates_valid[ws:]

            for window_thr in WINDOW_THR_LIST:
                for factor in FACTOR_LIST:
                    for k in K_LIST:

                        _, _, thr_valid_full, alarm_valid_full = compute_threshold_full(
                            err_full=err_valid_full,
                            window_thr=window_thr,
                            factor=factor,
                            k=k
                        )

                        valid_eval = evaluate_from_start(
                            err_dates=err_valid_dates,
                            err_full=err_valid_full,
                            thr_full=thr_valid_full,
                            alarm_full=alarm_valid_full,
                            region_name=valid_region
                        )

                        row = {
                            "valid_region": valid_region,
                            "model": model_file,
                            "ws": ws,
                            "h": h,
                            "bs": bs,
                            "window_thr": window_thr,
                            "factor": factor,
                            "k": k,

                            "valid_start": valid_eval["region_start"],
                            "valid_num_alarms": valid_eval["num_alarms"],
                            "valid_first_alarm_date": valid_eval["first_alarm_date"],
                            "valid_detection_delay_days": valid_eval["detection_delay_days"],
                        }

                        for region in TEST_REGIONS:
                            y_test_raw = pivot[region].to_numpy(dtype=float)
                            dates_test = pivot.index.to_numpy()

                            scaler_test = MinMaxScaler()
                            y_test = scaler_test.fit_transform(y_test_raw.reshape(-1, 1)).flatten()

                            _, _, err_test_full = build_forecast_full(model, y_test, ws)
                            err_test_dates = dates_test[ws:]

                            _, _, thr_test_full, alarm_test_full = compute_threshold_full(
                                err_full=err_test_full,
                                window_thr=window_thr,
                                factor=factor,
                                k=k
                            )

                            test_eval = evaluate_from_start(
                                err_dates=err_test_dates,
                                err_full=err_test_full,
                                thr_full=thr_test_full,
                                alarm_full=alarm_test_full,
                                region_name=region
                            )

                            prefix = region.lower()
                            row[f"{prefix}_start"] = test_eval["region_start"]
                            row[f"{prefix}_num_alarms"] = test_eval["num_alarms"]
                            row[f"{prefix}_first_alarm_date"] = test_eval["first_alarm_date"]
                            row[f"{prefix}_detection_delay_days"] = test_eval["detection_delay_days"]

                        all_rows.append(row)
                        region_results.append(row)

                        print(
                            f"   N={window_thr} | factor={factor:.1f} | k={k:.1f} "
                            f"-> VALID first_alarm={valid_eval['first_alarm_date']} "
                            f"| VALID delay={valid_eval['detection_delay_days']}"
                        )

        df_region = pd.DataFrame(region_results)

        df_region["rank_key"] = df_region.apply(
            lambda r: rank_tuple(
                r["valid_detection_delay_days"],
                r["valid_num_alarms"],
                r["ws"], r["h"], r["bs"],
                r["window_thr"], r["factor"], r["k"]
            ),
            axis=1
        )

        df_region = (
            df_region
            .sort_values("rank_key")
            .drop(columns=["rank_key"])
            .reset_index(drop=True)
        )

        best_row = df_region.iloc[0]

        summary_rows.append({
            "valid_region": valid_region,
            "model": best_row["model"],
            "ws": best_row["ws"],
            "h": best_row["h"],
            "bs": best_row["bs"],
            "window_thr": best_row["window_thr"],
            "factor": best_row["factor"],
            "k": best_row["k"],
            "lazio_first_alarm_date": best_row["lazio_first_alarm_date"],
            "campania_first_alarm_date": best_row["campania_first_alarm_date"],
            "sicilia_first_alarm_date": best_row["sicilia_first_alarm_date"],
        })

        print("\n[MEJOR CONFIGURACIÓN PARA ESTA VALIDACIÓN]")
        print(f"Validación: {valid_region}")
        print(f"Modelo    : {best_row['model']}")
        print(f"N         : {best_row['window_thr']}")
        print(f"factor    : {best_row['factor']}")
        print(f"k         : {best_row['k']}")
        print(f"Lazio     : {best_row['lazio_first_alarm_date']}")
        print(f"Campania  : {best_row['campania_first_alarm_date']}")
        print(f"Sicilia   : {best_row['sicilia_first_alarm_date']}")

    df_all = pd.DataFrame(all_rows)
    df_summary = pd.DataFrame(summary_rows)

    csv_all  = os.path.join(OUT_CSV,  "resultados_todas_validaciones.csv")
    xlsx_all = os.path.join(OUT_XLSX, "resultados_todas_validaciones.xlsx")

    csv_summary  = os.path.join(OUT_CSV,  "resumen_todas_validaciones.csv")
    xlsx_summary = os.path.join(OUT_XLSX, "resumen_todas_validaciones.xlsx")

    df_all.to_csv(csv_all, index=False)
    df_all.to_excel(xlsx_all, index=False)

    df_summary.to_csv(csv_summary, index=False)
    df_summary.to_excel(xlsx_summary, index=False)

    print("\n" + "=" * 100)
    print("[OK] Archivos generados:")
    print(" -", csv_all)
    print(" -", xlsx_all)
    print(" -", csv_summary)
    print(" -", xlsx_summary)
    print("=" * 100)

    print("\n[RESUMEN FINAL]")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()