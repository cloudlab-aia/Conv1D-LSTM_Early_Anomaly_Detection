# -*- coding: utf-8 -*-
"""
1. 1-LSTM_power_dataset_DINAMICO.py     <--- Estamos aquí
2. 2-Calculo_de_metricas.py
3. 3-Graficas_power_mejores_resultados.py

Este script SOLO ENTRENA y GUARDA modelos LSTM para power_dataset.txt haciendo barrido con WS/H/BS.

Aquí definimos:
- Franjas anómalas FIJAS manualmente (A1/A2/A3, tal y como hacen en el paper ECG-SHORT)
- Eliminado W_EARLY (no se usa aquí, aquí anticipación poca por la naturaleza del dataset).
- Eliminamos cosa del umbral/anomalías (K, MAX, rolling, alarmas, etc.) porque aquí solo vamos a generar modelos entrenados
- Guardamos los modelos en "...\modelos_dinamico\otros_modelos" con el nombre hacienod referencia a las 3 variables WS,H,BS
- Primero comprueba si un modelo ya existe, y si existe no lo vuelve a entrenar.
- También guarda 'results_entrenamiento.csv con el resumen del entrenamiento

[ Script 1 ]
LSTM_POWER_DINAMICO_BARRIDO_2.py
    └── entrena modelos base
    └── guarda en /modelos_dinamico/otros_modelos

[ Script 2 ]
2-Calculo_de_metricas.py
    └── carga modelos base
    └── calibra detectores (k, MAX, W, F)
    └── calcula métricas
    └── genera:
         ├── \pruebas_extras_ajuste\csv_candidatos\candidatos_all.csv
         └── \pruebas_extras_ajuste\csv_candidatos\candidatos_pass_filter.csv
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping



# DATASET

SRC_TXT = "power_dataset.txt"
DATASET_NAME = Path(SRC_TXT).stem

"""
Cálculo de puntos por día:
    - 1 día = 24 horas
    - 1 hora = 60 minutos
    - 1 punto = 14 minutos
    - (24x60)/15 = 96 muestras por día
"""
PTS_PER_DAY = 96
PTS_PER_WEEK = 7 * PTS_PER_DAY

TRAIN_FRAC = 0.70
ALIGN_SPLIT_TO_WEEKS = True

# Franjas ANÓMALAS FIJAS (A1/A2/A3) [inicio, fin)
ANOM_RANGES = [
    (7872, 9216),      # A1
    (11232, 12576),    # A2
    (34080, 35040),    # A3
]

# Franjas NORMALES[inicio, fin)

NORM_RANGES = [
    (0, 7872),
    (9216, 11232),
    (12576, 34080),
]


# CONFIGURACIÓN DEL BARRIDO LSTM (SOLO ENTRENA)

Window_size = [64, 128, 256]
Horizon  = [30, 35, 40, 45, 50, 55, 60, 128, 256]
Batch_size = [16, 32]


# ENTRENAMIENTO

EPOCHS = 40
LR = 1e-4
SHUFFLE_BUFFER = 4000
EARLY_STOP_PATIENCE = 6
EARLY_STOP_MIN_DELTA = 1e-5


# SALIDAS

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "modelos_dinamico" / "otros_modelos"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = ROOT_DIR / "modelos_dinamico/results_entrenamiento.csv"



# HELPERS I/O

def load_series(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for p in s.replace("\t", " ").replace(",", " ").split():
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
    if not vals:
        raise ValueError("Dataset vacío o mal leído")
    return np.asarray(vals, dtype=float)


def split_index_by_frac(n: int, frac: float) -> int:
    return int(np.floor(n * frac))


def split_index_power(n: int, frac: float) -> int:
    """Split alineado a semanas completas (recomendado en Power)."""
    if not ALIGN_SPLIT_TO_WEEKS:
        return int(np.clip(split_index_by_frac(n, frac), 1, n - 1))

    nweeks = n // PTS_PER_WEEK
    if nweeks < 2:
        return int(np.clip(split_index_by_frac(n, frac), 1, n - 1))

    split_week = int(np.floor(nweeks * frac))
    split_week = int(np.clip(split_week, 1, nweeks - 1))
    return split_week * PTS_PER_WEEK


# =========================================================
# LSTM
# =========================================================
def windowed_dataset(series: np.ndarray, w: int, h: int, bs: int, shuffle: int) -> tf.data.Dataset:
    """
    Dataset supervisado:
      X = ventana de tamaño w
      y = valor en t+(h-1) relativo al final de la ventana (predicción a horizonte h)
    """
    s = tf.expand_dims(tf.convert_to_tensor(series, tf.float32), -1)
    ds = tf.data.Dataset.from_tensor_slices(s)
    ds = ds.window(w + h, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x.batch(w + h))
    ds = ds.shuffle(shuffle)
    ds = ds.map(lambda x: (x[:w], x[w + h - 1]))
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)


def build_model(lr: float) -> tf.keras.Model:
    m = Sequential([
        Conv1D(12, 5, padding="causal", activation="relu"),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    m.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr))
    return m


def train_if_needed(y_s: np.ndarray, split_raw: int, WS: int, H: int, BS: int) -> dict:
    """
    Entrena y guarda un modelo base si no existe.
    Devuelve un dict con info para el CSV.
    """
    model_name = f"base_{DATASET_NAME}_WS{WS}_H{H}_BS{BS}.h5"
    model_path = MODELS_DIR / model_name

    if model_path.exists():
        print(f"[SKIP] Ya existe: {model_path.name}")
        return {
            "dataset": DATASET_NAME,
            "model_file": model_path.name,
            "WS": WS, "H": H, "BS": BS,
            "status": "exists",
            "epochs_trained": 0,
        }

    train_ds = windowed_dataset(y_s[:split_raw], WS, H, BS, SHUFFLE_BUFFER)
    model = build_model(LR)

    es = EarlyStopping(
        monitor="loss",
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        restore_best_weights=True
    )

    history = model.fit(train_ds, epochs=EPOCHS, callbacks=[es], verbose=0)
    epochs_done = len(history.history.get("loss", []))

    model.save(model_path)
    print(f"[OK] Entrenado y guardado: {model_path.name} (epochs={epochs_done})")

    return {
        "dataset": DATASET_NAME,
        "model_file": model_path.name,
        "WS": WS, "H": H, "BS": BS,
        "status": "trained",
        "epochs_trained": epochs_done,
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


# =========================================================
# MAIN
# =========================================================
def main():
    tf.get_logger().setLevel("ERROR")

    print(f"[ROOT] {ROOT_DIR}")
    print(f"[OUT] modelos_dinamico\otros_modelos -> {MODELS_DIR}")

    y = load_series(SRC_TXT)
    n = len(y)

    split_raw = split_index_power(n, TRAIN_FRAC)
    print(f"[SPLIT] split_raw={split_raw} (train={split_raw} pts, test={n - split_raw} pts)")

    print("[ANOM] Franjas anómalas fijas [inicio, fin):")
    for i, (a, b) in enumerate(ANOM_RANGES, start=1):
        where = "TEST" if a >= split_raw else ("MIX" if b > split_raw else "TRAIN")
        print(f"  A{i}: [{a}, {b}) -> {where}")

    # Escalado con TRAIN
    y_train = y[:split_raw]
    ymin, ymax = float(np.min(y_train)), float(np.max(y_train))
    y_s = (y - ymin) / (ymax - ymin + 1e-8)

    all_rows = []
    for WS in Window_size:
        for H in Horizon:
            for BS in Batch_size:
                print(f"\n=== TRAIN {DATASET_NAME} | WS={WS} H={H} BS={BS} ===")
                row = train_if_needed(y_s, split_raw, WS, H, BS)
                row.update({
                    "split_raw": split_raw,
                    "train_pts": split_raw,
                    "test_pts": n - split_raw,
                    "ymin_train": ymin,
                    "ymax_train": ymax,
                })
                all_rows.append(row)

    write_csv(OUT_CSV, all_rows)
    print(f"\n[CSV] Resumen guardado: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
