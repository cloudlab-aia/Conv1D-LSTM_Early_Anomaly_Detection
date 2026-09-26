# -*- coding: utf-8 -*-
"""
1. 1-LSTM_power_dataset_DINAMICO.py
2. 2-Calculo_de_metricas.py
3. 3-Graficas_power_mejores_resultados.py
4. 4-Graficas_power_mejores_resultados_zoom.py
------------------------------------------------------------
Genera, para cada fila seleccionada del fichero de candidatos
filtrado con selección en TRAIN:

1) UNA figura global por cada fila seleccionada
   - Serie completa
   - Zona de entrenamiento sombreada en gris claro
   - Regiones anómalas sombreadas
   - Alarmas
   - Error + umbral dinámico

2) UNA figura adicional por cada región anómala
   - Zoom local en la región
   - Contexto antes y después
   - Zona de entrenamiento sombreada si aparece en el zoom

Todas las imágenes se guardan en VALIDACION/imagenes.
------------------------------------------------------------
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

# Filas a graficar
ROW_INDEXES = [6550]   # ejemplo: [11, 393, 972]

CANDIDATES_FILE = VALIDACION_DIR / "xlsx" / "candidatos_pass_filter_train_selection.xlsx"
MODELS_DIR = ROOT_DIR / "modelos_dinamico" / "otros_modelos"
DATASET_FILE = ROOT_DIR / "power_dataset.txt"
OUT_IMG_DIR = VALIDACION_DIR / "imagenes"

TRAIN_FRAC = 0.70

# Rangos anómalos actualizados para coincidir con el script de métricas
A1_RANGE = (7872, 9216)
A2_RANGE = (11232, 12576)
A3_RANGE = (33780, 35040)

ANOM_RANGES = [
    A1_RANGE,
    A2_RANGE,
    A3_RANGE,
]

# Margen para las gráficas individuales
ZOOM_MARGIN = 600

# Sombreado de entrenamiento
TRAIN_SHADE_COLOR = "lightgray"
TRAIN_SHADE_ALPHA = 0.35

# Mostrar o no las figuras en pantalla
SHOW_FIGURES = True
PAUSE_SECONDS = 3

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
    mu = np.full(n, np.nan)
    sd = np.full(n, np.nan)

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


def print_table1_comparison(df, row_indexes):
    print("\n" + "=" * 70)
    print("TABLE 1 – POWER DATASET (Paper vs Ours)")
    print("=" * 70)
    print(f"{'Method':<30} {'Precision':<10} {'Recall':<10} {'F0.1':<10}")
    print("-" * 70)

    # Resultado del paper
    print(f"{'LSTM (30-20) [Paper]':<30} {0.94:<10.2f} {0.17:<10.2f} {0.90:<10.2f}")

    # Resultados propios en TEST
    for idx in row_indexes:
        if idx < 0 or idx >= len(df):
            continue
        r = df.iloc[idx]
        print(
            f"{'LSTM-Conv1D':<30} "
            f"{r['Precision_test']:<10.3f} "
            f"{r['Recall_test']:<10.3f} "
            f"{r['f0.1-score_test']:<10.3f}"
        )

    print("=" * 70 + "\n")


def save_and_optionally_show(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("[OK] Guardada:", out_path.name)

    if SHOW_FIGURES:
        plt.show(block=False)
        plt.pause(PAUSE_SECONDS)

    plt.close(fig)


# =========================================================
# GRÁFICA GLOBAL
# =========================================================
def plot_global_figure(
    row_idx, model_name, index, y_al, pred_shift, err, thr, alarm,
    split_raw, out_dir: Path
):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # ---------------------------
    # Panel superior
    # ---------------------------
    ax1.plot(index, y_al, label="Real")
    ax1.plot(index, pred_shift, label="Predicción")

    # Zona de entrenamiento sombreada
    train_left = index[0]
    train_right = min(split_raw, index[-1])
    if train_left < train_right:
        ax1.axvspan(train_left, train_right,
                    color=TRAIN_SHADE_COLOR, alpha=TRAIN_SHADE_ALPHA,
                    label="Train" if split_raw >= index[0] else None)

    if np.any(alarm):
        ax1.scatter(index[alarm], y_al[alarm], color="red", s=15, label="Alarmas")

    for a, b in ANOM_RANGES:
        ax1.axvspan(a, b, color="red", alpha=0.12)

    ax1.axvline(split_raw, linestyle="--", color="tab:blue", label="Train/Test")
    ax1.legend()
    ax1.set_title(f"Fila {row_idx} | {model_name}")

    # ---------------------------
    # Panel inferior
    # ---------------------------
    ax2.plot(index, err, label="|error|")
    ax2.plot(index, thr, linestyle="--", label="Umbral")

    if train_left < train_right:
        ax2.axvspan(train_left, train_right,
                    color=TRAIN_SHADE_COLOR, alpha=TRAIN_SHADE_ALPHA)

    if np.any(alarm):
        ax2.scatter(index[alarm], err[alarm], color="red", s=12)

    for a, b in ANOM_RANGES:
        ax2.axvspan(a, b, color="red", alpha=0.12)

    ax2.axvline(split_raw, linestyle="--", color="tab:blue")
    ax2.legend()
    ax2.set_xlabel("Índice")

    out_name = f"row{row_idx}_{model_name.replace('.h5','')}_GLOBAL.png"
    out_path = out_dir / out_name
    save_and_optionally_show(fig, out_path)


# =========================================================
# GRÁFICAS INDIVIDUALES POR ANOMALÍA
# =========================================================
def plot_anomaly_zoom_figures(
    row_idx, model_name, index, y_al, pred_shift, err, thr, alarm,
    split_raw, out_dir: Path
):
    for region_id, (a, b) in enumerate(ANOM_RANGES, start=1):
        z0 = max(index[0], a - ZOOM_MARGIN)
        z1 = min(index[-1], b + ZOOM_MARGIN)

        mask = (index >= z0) & (index <= z1)
        if not np.any(mask):
            print(f"[WARN] Región {region_id} fuera del rango representable")
            continue

        idx_z = index[mask]
        y_z = y_al[mask]
        pred_z = pred_shift[mask]
        err_z = err[mask]
        thr_z = thr[mask]
        alarm_z = alarm[mask]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 7), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        # ---------------------------
        # Panel superior
        # ---------------------------
        ax1.plot(idx_z, y_z, label="Real")
        ax1.plot(idx_z, pred_z, label="Predicción")

        train_left = idx_z[0]
        train_right = min(split_raw, idx_z[-1])
        if train_left < train_right:
            ax1.axvspan(train_left, train_right,
                        color=TRAIN_SHADE_COLOR, alpha=TRAIN_SHADE_ALPHA)

        if np.any(alarm_z):
            ax1.scatter(idx_z[alarm_z], y_z[alarm_z], color="red", s=20, label="Alarmas")

        ax1.axvspan(a, b, color="red", alpha=0.15, label="Región anómala")

        if z0 <= split_raw <= z1:
            ax1.axvline(split_raw, linestyle="--", color="tab:blue", label="Train/Test")

        ax1.set_title(
            f"Fila {row_idx} | {model_name} | Región anómala {region_id} [{a}, {b}]"
        )
        ax1.legend()

        # ---------------------------
        # Panel inferior
        # ---------------------------
        ax2.plot(idx_z, err_z, label="|error|")
        ax2.plot(idx_z, thr_z, linestyle="--", label="Umbral")

        if train_left < train_right:
            ax2.axvspan(train_left, train_right,
                        color=TRAIN_SHADE_COLOR, alpha=TRAIN_SHADE_ALPHA)

        if np.any(alarm_z):
            ax2.scatter(idx_z[alarm_z], err_z[alarm_z], color="red", s=18)

        ax2.axvspan(a, b, color="red", alpha=0.15)

        if z0 <= split_raw <= z1:
            ax2.axvline(split_raw, linestyle="--", color="tab:blue")

        ax2.legend()
        ax2.set_xlabel("Índice")

        out_name = (
            f"row{row_idx}_{model_name.replace('.h5','')}"
            f"_ANOM_{region_id}_{a}_{b}.png"
        )
        out_path = out_dir / out_name
        save_and_optionally_show(fig, out_path)


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

    print_table1_comparison(df, ROW_INDEXES)

    y = load_series(DATASET_FILE)
    n = len(y)
    split_raw = int(n * TRAIN_FRAC)

    # Normalización con train
    ymin = y[:split_raw].min()
    ymax = y[:split_raw].max()
    y_s = (y - ymin) / (ymax - ymin + 1e-8)

    for row_idx in ROW_INDEXES:
        if row_idx < 0 or row_idx >= len(df):
            print(f"[SKIP] Fila {row_idx} fuera de rango")
            continue

        row = df.iloc[row_idx]

        model_name = row["model"]
        W = int(row["WINDOW_THR"])
        F = float(row["FACTOR"])
        K = float(row["k"])

        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            print(f"[SKIP] Modelo no encontrado: {model_name}")
            continue

        print(f"[INFO] Generando figuras para fila {row_idx}")

        ws, h = parse_model_params(model_name)
        model = tf.keras.models.load_model(model_path)

        # -------------------------------------------------
        # Forecast
        # -------------------------------------------------
        s = tf.expand_dims(tf.convert_to_tensor(y_s, tf.float32), -1)
        ds = tf.data.Dataset.from_tensor_slices(s)
        ds = ds.window(ws, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda x: x.batch(ws))
        ds = ds.batch(1024)

        pred_s = model.predict(ds, verbose=0).squeeze()

        start_idx = ws + h - 1
        L = min(len(pred_s), n - start_idx)
        if L <= 0:
            print(f"[SKIP] Longitud inválida para fila {row_idx}")
            continue

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

        # -------------------------------------------------
        # Figura global
        # -------------------------------------------------
        plot_global_figure(
            row_idx=row_idx,
            model_name=model_name,
            index=index,
            y_al=y_al,
            pred_shift=pred_shift,
            err=err,
            thr=thr_scaled,
            alarm=alarm,
            split_raw=split_raw,
            out_dir=OUT_IMG_DIR
        )

        # -------------------------------------------------
        # Figuras por anomalía
        # -------------------------------------------------
        plot_anomaly_zoom_figures(
            row_idx=row_idx,
            model_name=model_name,
            index=index,
            y_al=y_al,
            pred_shift=pred_shift,
            err=err,
            thr=thr_scaled,
            alarm=alarm,
            split_raw=split_raw,
            out_dir=OUT_IMG_DIR
        )


if __name__ == "__main__":
    main()