"""
04_robustez.py
==============
Chequeo de robustez: el desempenio de los atributos acusticos no depende de
donde se ponga el corte de popularidad.

Motivacion: podria pensarse que el Random Forest rinde poco solo porque
definir un "super hit" como el top 1% genera una clase extremadamente
desbalanceada. Este programa reestima el modelo moviendo el umbral de
popularidad y muestra que el AUC se mantiene aproximadamente constante.

Entrada : output/bases/base_2018.csv , output/bases/base_2019.csv
Salida  : output/tables/robustez-umbral.tex
          output/figures/robustez-umbral.png

Uso:
    python 04_robustez.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
BASES = RAIZ / "output" / "bases"
TABLAS = RAIZ / "output" / "tables"
FIGURAS = RAIZ / "output" / "figures"

SEMILLA = 444
UMBRALES = [30, 40, 52, 60, 70, 80]

PREDICTORES = [
    "acousticness", "danceability", "duration_ms", "energy", "instrumentalness",
    "key", "liveness", "loudness", "mode", "speechiness", "tempo",
    "time_signature", "valence", "artist_count", "explicit_dummy", "song_age",
]


def main():
    TABLAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)

    df18 = pd.read_csv(BASES / "base_2018.csv", low_memory=False)
    df19 = pd.read_csv(BASES / "base_2019.csv", low_memory=False)
    panel = (
        df18[["track_id", "popularity_2018"] + PREDICTORES]
        .merge(df19[["track_id", "popularity_2019"]], on="track_id", how="inner")
        .drop_duplicates("track_id")
        .dropna(subset=["popularity_2018", "popularity_2019"] + PREDICTORES)
        .reset_index(drop=True)
    )

    filas = []
    for umbral in UMBRALES:
        y = (panel["popularity_2019"] >= umbral).astype(int)
        if y.sum() < 50:
            continue
        tr, te, ytr, yte = train_test_split(
            panel[PREDICTORES], y, test_size=.3, random_state=SEMILLA, stratify=y
        )
        rf = RandomForestClassifier(
            n_estimators=100, max_features=3, max_samples=.75,
            bootstrap=True, random_state=SEMILLA, n_jobs=-1,
        ).fit(tr, ytr)
        auc = roc_auc_score(yte, rf.predict_proba(te)[:, 1])
        # una regla trivial que predice siempre "no es hit" ya acierta esta fraccion
        acc_trivial = 1 - y.mean()
        filas.append({"Umbral": umbral, "Positivos (\\%)": y.mean() * 100,
                      "AUC": auc, "Aciertos triviales": acc_trivial})
        print(f"  umbral {umbral:>3}: {y.mean()*100:>5.2f}% positivos, AUC={auc:.4f}")

    res = pd.DataFrame(filas)
    (TABLAS / "robustez-umbral.tex").write_text(
        res.to_latex(
            index=False, float_format="%.3f", position="htbp", escape=False,
            caption=(
                "Desempeño del Random Forest sobre atributos acústicos según el "
                "umbral de popularidad usado para definir un hit. La última columna "
                "es la proporción de aciertos de una regla que nunca predice un hit."
            ),
            label="tab:robustez",
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res["Umbral"], res["AUC"], "o-", color="#2b6cb0", linewidth=1.8)
    ax.axhline(.5, linestyle="--", color="gray", linewidth=.8)
    ax.annotate("azar", xy=(res["Umbral"].iloc[0], .515), fontsize=9, color="gray")
    ax.set_ylim(.45, 1.0)
    ax.set_xlabel("Umbral de popularidad que define un hit", fontsize=12)
    ax.set_ylabel("AUC en la muestra de testeo", fontsize=12)
    ax.set_title("El poder predictivo de los atributos acústicos\nno depende del umbral",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURAS / "robustez-umbral.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nListo. Tabla en {TABLAS}, figura en {FIGURAS}")


if __name__ == "__main__":
    main()
