# -*- coding: utf-8 -*-
"""
2-Calculo_de_metricas.py
============================================================
CASO POWER DATASET — ENFOQUE A ADAPTADO
============================================================

Objetivo metodológico:
----------------------
1) Los modelos base (WS, H, BS) ya vienen entrenados SOLO con TRAIN.
2) Este script NO reentrena modelos.
3) La calibración del detector (WINDOW_THR, FACTOR, MAX -> k) se realiza
   SOLO con información de TRAIN.
4) La selección final de configuraciones se realiza SOLO con métricas de TRAIN.
5) TEST se usa únicamente para evaluación final y para resaltar resultados,
   sin intervenir en la selección.

Estructura del dataset POWER:
-----------------------------
- A1: anomalía en TRAIN
- A2: anomalía en TRAIN
- A3: anomalía en TEST (ligeramente ampliada para evaluación)

Qué genera:
-----------
Todo cuelga de la carpeta VALIDACION (donde estará este script):

VALIDACION/
    csv/
        - candidatos_all_train_selection.csv
        - candidatos_pass_filter_train_selection.csv
    xlsx/
        - candidatos_all_train_selection.xlsx
        - candidatos_pass_filter_train_selection.xlsx
    imagenes/
        - (reservada para el script 3)

Rutas esperadas:
----------------
- Este script estará en:           .../VALIDACION/2-Calculo_de_metricas.py
- El dataset estará en el raíz:    .../power_dataset.txt
- Los modelos estarán en:          .../modelos_dinamico/otros_modelos/

Notas:
------
- El criterio de Jorge se mantiene:
      NORMAL = clase positiva
      TP = normal sin alarma
      FN = normal con alarma
      FP = anómalo sin alarma
      TN = anómalo con alarma
- La selección/filtro se hace SOLO con columnas de TRAIN.
- Las columnas de TEST se calculan solo como evaluación final descriptiva.
"""

from __future__ import annotations

from pathlib import Path
import re
import csv
import time
import numpy as np
import tensorflow as tf
import pandas as pd


# =========================================================
# RUTAS
# =========================================================
VALIDACION_DIR = Path(__file__).resolve().parent           # .../VALIDACION
ROOT_DIR = VALIDACION_DIR.parent                           # .../ (raíz del proyecto)

MODELS_DIR = ROOT_DIR / "modelos_dinamico" / "otros_modelos"
DATASET_FILE = ROOT_DIR / "power_dataset.txt"

OUT_CSV_DIR = VALIDACION_DIR / "csv"
OUT_XLSX_DIR = VALIDACION_DIR / "xlsx"
OUT_IMG_DIR = VALIDACION_DIR / "imagenes"   # reservado para el script 3

OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
OUT_XLSX_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# CONFIG GENERAL
# =========================================================
TRAIN_FRAC = 0.70

WINDOW_THR_LIST = list(range(300, 751, 50))
FACTOR_LIST = [i / 10 for i in range(10, 21)]      # 1.0 .. 2.0
MAX_LIST = [0, 2, 4, 6, 12]
K_VALUES = np.arange(1.00, 12.01, 0.01)

# =========================================================
# FRANJAS POWER
# =========================================================
# A1 y A2 se mantienen igual
A1_RANGE = (7872, 9216)        # TRAIN
A2_RANGE = (11232, 12576)      # TRAIN

# A3 se amplía un poco SOLO para la evaluación en TEST
A3_MARGIN = 300
A3_RANGE_ORIG = (34080, 35040)
A3_RANGE = (A3_RANGE_ORIG[0] - A3_MARGIN, A3_RANGE_ORIG[1] + A3_MARGIN)   # TEST ampliada

ANOM_RANGES = [
    A1_RANGE,
    A2_RANGE,
    A3_RANGE,
]

# Franjas normales ajustadas para no solapar con A3 ampliada
NORM_RANGES = [
    (0, A1_RANGE[0]),
    (A1_RANGE[1], A2_RANGE[0]),
    (A2_RANGE[1], A3_RANGE[0]),
]

# Filtros de selección (SOLO TRAIN)
FILTER_PREC_MIN = 0.80
FILTER_REC_MIN  = 0.70
FILTER_F01_MIN  = 0.60


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
    ws = int(re.search(r"_WS(\d+)", name).group(1))
    h  = int(re.search(r"_H(\d+)", name).group(1))
    bs = int(re.search(r"_BS(\d+)", name).group(1))
    return ws, h, bs


def rolling_mean_std_nan(x: np.ndarray, w: int):
    """
    Media y desviación típica causales sobre ventana de tamaño w.
    Devuelve NaN mientras no hay suficientes puntos.
    """
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


def mask_from_ranges(index: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros_like(index, dtype=bool)
    for a, b in ranges:
        m |= (index >= a) & (index < b)
    return m


def mask_single_range(index: np.ndarray, a: int, b: int) -> np.ndarray:
    return (index >= a) & (index < b)


def fbeta(p: float, r: float, beta: float = 0.1) -> float:
    if (p + r) == 0:
        return 0.0
    b2 = beta * beta
    den = b2 * p + r
    if den == 0:
        return 0.0
    return (1 + b2) * p * r / den


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def write_excel(path: Path, rows: list[dict]):
    if not rows:
        return
    pd.DataFrame(rows).to_excel(path, index=False)


def compute_jorge_metrics(alarm: np.ndarray,
                          m_eval: np.ndarray,
                          m_norm: np.ndarray,
                          m_anom: np.ndarray) -> dict:
    """
    Métricas pointwise según criterio Jorge:
      TP = normal sin alarma
      FN = normal con alarma
      FP = anómalo sin alarma
      TN = anómalo con alarma
    """
    TP = int(np.sum(m_eval & m_norm & (~alarm)))
    FN = int(np.sum(m_eval & m_norm & alarm))
    FP = int(np.sum(m_eval & m_anom & (~alarm)))
    TN = int(np.sum(m_eval & m_anom & alarm))

    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F01       = fbeta(Precision, Recall, beta=0.1)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Precision": Precision,
        "Recall": Recall,
        "f0.1-score": F01,
    }


# =========================================================
# MAIN
# =========================================================
def main():
    t0 = time.time()
    tf.get_logger().setLevel("ERROR")

    print("=" * 90)
    print("POWER DATASET — VALIDACION SIN LEAKAGE (ENFOQUE A)")
    print("=" * 90)
    print(f"[VALIDACION] Carpeta script/salidas: {VALIDACION_DIR}")
    print(f"[ROOT]       Carpeta raíz proyecto  : {ROOT_DIR}")
    print(f"[DATASET]    {DATASET_FILE}")
    print(f"[MODELOS]    {MODELS_DIR}")
    print()
    print(f"[A3 ORIG]    {A3_RANGE_ORIG}")
    print(f"[A3 AMPL.]   {A3_RANGE}  (margin={A3_MARGIN})")
    print()

    # -----------------------------------------------------
    # Carga dataset
    # -----------------------------------------------------
    y = load_series(DATASET_FILE)
    n_raw = len(y)
    split_raw = int(n_raw * TRAIN_FRAC)

    print(f"[DATA] Número total de puntos: {n_raw}")
    print(f"[SPLIT] TRAIN={split_raw} | TEST={n_raw - split_raw}")
    print()

    # Escalado con TRAIN únicamente
    ymin = float(np.min(y[:split_raw]))
    ymax = float(np.max(y[:split_raw]))
    y_s = (y - ymin) / (ymax - ymin + 1e-8)

    # Modelos
    model_files = sorted(MODELS_DIR.glob("*.h5"))
    if not model_files:
        raise FileNotFoundError(f"No hay modelos .h5 en: {MODELS_DIR}")

    print(f"[MODELOS] Encontrados: {len(model_files)}")
    print()

    results_all = []
    results_pass = []

    # -----------------------------------------------------
    # Bucle principal
    # -----------------------------------------------------
    for i_model, mp in enumerate(model_files, start=1):
        print("-" * 90)
        print(f"[{i_model}/{len(model_files)}] Procesando modelo: {mp.name}")

        ws, h, bs = parse_model_params(mp.name)
        model = tf.keras.models.load_model(mp)

        # -------------------------------
        # Forecast sobre toda la serie
        # -------------------------------
        s = tf.expand_dims(tf.convert_to_tensor(y_s, tf.float32), -1)
        ds = tf.data.Dataset.from_tensor_slices(s)
        ds = ds.window(ws, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda x: x.batch(ws))

        pred_s = model.predict(ds.batch(1024), verbose=0).squeeze()

        start_idx = ws + h - 1
        L = min(len(pred_s), n_raw - start_idx)

        if L <= 0:
            print(f"[SKIP] L <= 0 para {mp.name}")
            continue

        index = np.arange(start_idx, start_idx + L)
        y_al = y[start_idx:start_idx + L]

        # Desnormalización
        pred = pred_s[:L] * (ymax - ymin + 1e-8) + ymin

        # Shift para alinear horizonte h
        pred_shift = np.full_like(pred, np.nan, dtype=float)
        if L > h:
            pred_shift[:-h] = pred[h:]

        err = np.abs(y_al - pred_shift)

        # Máscaras globales
        m_norm = mask_from_ranges(index, NORM_RANGES)
        m_anom = mask_from_ranges(index, ANOM_RANGES)
        m_train = index < split_raw
        m_test = index >= split_raw

        # Zonas anomalías individuales
        m_a1 = mask_single_range(index, *A1_RANGE)
        m_a2 = mask_single_range(index, *A2_RANGE)
        m_a3 = mask_single_range(index, *A3_RANGE)

        for W in WINDOW_THR_LIST:
            mu_roll, sd_roll = rolling_mean_std_nan(err, W)

            valid = np.isfinite(err) & np.isfinite(mu_roll) & np.isfinite(sd_roll)

            warm = np.zeros_like(valid, dtype=bool)
            warm[:W] = True

            # -------------------------------------------------
            # CALIBRACIÓN SOLO EN TRAIN NORMAL
            # -------------------------------------------------
            m_train_norm_cal = m_train & m_norm & valid & (~warm)

            # Evaluación TRAIN: normales + anómalas de train
            m_train_eval = m_train & (m_norm | m_anom) & valid & (~warm)

            # Evaluación TEST: normales + anómalas de test
            m_test_eval = m_test & (m_norm | m_anom) & valid & (~warm)

            if not np.any(m_train_norm_cal):
                continue
            if not np.any(m_train_eval):
                continue

            err_tr = err[m_train_norm_cal]
            mu_tr = mu_roll[m_train_norm_cal]
            sd_tr = sd_roll[m_train_norm_cal]

            for F in FACTOR_LIST:
                # -------------------------------------------------
                # Buscar k mínimo que cumpla MAX SOLO EN TRAIN NORMAL
                # -------------------------------------------------
                counts = []
                for k in K_VALUES:
                    thr_tr = mu_tr + k * sd_tr
                    counts.append(int(np.sum(err_tr > F * thr_tr)))
                counts = np.asarray(counts)

                for MAXV in MAX_LIST:
                    ok = np.where(counts <= MAXV)[0]
                    if ok.size == 0:
                        continue

                    k_sel = float(K_VALUES[ok[0]])

                    # Umbral final congelado
                    thr = mu_roll + k_sel * sd_roll
                    alarm = (err > F * thr) & valid
                    alarm[:W] = False

                    # =============================================
                    # MÉTRICAS TRAIN (SELECCIÓN)
                    # =============================================
                    train_metrics = compute_jorge_metrics(
                        alarm=alarm,
                        m_eval=m_train_eval,
                        m_norm=m_norm,
                        m_anom=m_anom
                    )

                    alarms_A1_train = int(np.sum(alarm & m_a1 & m_train))
                    alarms_A2_train = int(np.sum(alarm & m_a2 & m_train))
                    alarms_train_total = int(np.sum(alarm & m_train_eval))

                    # =============================================
                    # MÉTRICAS TEST (SOLO EVALUACIÓN FINAL)
                    # =============================================
                    if np.any(m_test_eval):
                        test_metrics = compute_jorge_metrics(
                            alarm=alarm,
                            m_eval=m_test_eval,
                            m_norm=m_norm,
                            m_anom=m_anom
                        )
                        alarms_A3_test = int(np.sum(alarm & m_a3 & m_test))
                        alarms_test_total = int(np.sum(alarm & m_test_eval))
                    else:
                        test_metrics = {
                            "TP": 0, "FP": 0, "FN": 0, "TN": 0,
                            "Precision": 0.0, "Recall": 0.0, "f0.1-score": 0.0
                        }
                        alarms_A3_test = 0
                        alarms_test_total = 0

                    row = {
                        "model": mp.name,
                        "WS": ws,
                        "H": h,
                        "BS": bs,
                        "WINDOW_THR": W,
                        "FACTOR": F,
                        "MAX": MAXV,
                        "k": round(k_sel, 2),

                        # -------------------------
                        # TRAIN = SELECCIÓN
                        # -------------------------
                        "Precision_train": round(train_metrics["Precision"], 6),
                        "Recall_train": round(train_metrics["Recall"], 6),
                        "f0.1-score_train": round(train_metrics["f0.1-score"], 6),
                        "TP_train": train_metrics["TP"],
                        "FP_train": train_metrics["FP"],
                        "FN_train": train_metrics["FN"],
                        "TN_train": train_metrics["TN"],
                        "alarms_A1_train": alarms_A1_train,
                        "alarms_A2_train": alarms_A2_train,
                        "alarms_train_total": alarms_train_total,

                        # -------------------------
                        # TEST = SOLO INFORME
                        # -------------------------
                        "Precision_test": round(test_metrics["Precision"], 6),
                        "Recall_test": round(test_metrics["Recall"], 6),
                        "f0.1-score_test": round(test_metrics["f0.1-score"], 6),
                        "TP_test": test_metrics["TP"],
                        "FP_test": test_metrics["FP"],
                        "FN_test": test_metrics["FN"],
                        "TN_test": test_metrics["TN"],
                        "alarms_A3_test": alarms_A3_test,
                        "alarms_test_total": alarms_test_total,
                    }

                    results_all.append(row)

                    # =============================================
                    # FILTRO/S ELECCIÓN SOLO CON TRAIN
                    # =============================================
                    if (
                        train_metrics["Precision"] >= FILTER_PREC_MIN and
                        train_metrics["Recall"]    >= FILTER_REC_MIN and
                        train_metrics["f0.1-score"] >= FILTER_F01_MIN and
                        alarms_A1_train > 0 and
                        alarms_A2_train > 0
                    ):
                        results_pass.append(row)

    # -----------------------------------------------------
    # Guardado
    # -----------------------------------------------------
    csv_all_path = OUT_CSV_DIR / "candidatos_all_train_selection.csv"
    csv_pass_path = OUT_CSV_DIR / "candidatos_pass_filter_train_selection.csv"

    xlsx_all_path = OUT_XLSX_DIR / "candidatos_all_train_selection.xlsx"
    xlsx_pass_path = OUT_XLSX_DIR / "candidatos_pass_filter_train_selection.xlsx"

    write_csv(csv_all_path, results_all)
    write_csv(csv_pass_path, results_pass)
    write_excel(xlsx_all_path, results_all)
    write_excel(xlsx_pass_path, results_pass)

    elapsed = time.time() - t0

    print()
    print("=" * 90)
    print("[OK] Resultados guardados")
    print(f"[CSV ALL ] {csv_all_path}")
    print(f"[CSV PASS] {csv_pass_path}")
    print(f"[XLSX ALL ] {xlsx_all_path}")
    print(f"[XLSX PASS] {xlsx_pass_path}")
    print(f"[TOTAL FILAS ALL ] {len(results_all)}")
    print(f"[TOTAL FILAS PASS] {len(results_pass)}")
    print(f"[TIME] Duración total: {elapsed:.1f} s ({elapsed/60:.2f} min)")
    print("=" * 90)


if __name__ == "__main__":
    main()