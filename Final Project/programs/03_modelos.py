"""
03_modelos.py
=============
Compara dos formas de identificar un "super hit" (cancion en el top 1% de
popularidad): una ley de potencia estimada por maxima verosimilitud sobre la
popularidad pasada, y un Random Forest sobre los atributos acusticos.

Entrada : output/bases/base_2018.csv , output/bases/base_2019.csv
Salida  : output/tables/resultados.tex
          output/tables/matrices-confusion.tex
          output/tables/parametros.tex
          output/figures/roc-comparacion.png
          output/figures/importancia-variables.png

Uso:
    python 03_modelos.py                # grilla completa de hiperparametros
    python 03_modelos.py --quick        # grilla reducida (para probar rapido)

------------------------------------------------------------------------------
NOTA METODOLOGICA
------------------------------------------------------------------------------
La version original de este analisis tenia tres problemas que este programa
corrige. Quedan documentados porque cambian las conclusiones del trabajo:

1. CIRCULARIDAD. El modelo de ley de potencia calculaba la probabilidad a
   partir de `popularity_2019`, que es la misma variable con la que se define
   la etiqueta (estar en el top 1% de 2019). Predecia la variable con si
   misma. Aca la ley de potencia se evalua sobre `popularity_2018`, de modo
   que la prediccion es genuinamente fuera de muestra en el tiempo.

2. FORMULA. La funcion de supervivencia de una ley de potencia es
   P(X >= x) = (x/xmin)^(1-alpha). El codigo original usaba
   (x/umbral_99)^(-(alpha+1)): exponente equivocado y xmin reemplazado por el
   percentil 99. El resultado era decreciente en la popularidad (a mas
   popular, menor probabilidad asignada) y tomaba valores mayores que 1, lo
   que es imposible para una probabilidad.

3. UMBRAL DE DECISION. Con una tasa base de 1%, cortar la probabilidad
   predicha en 0.5 hace que casi ningun caso se clasifique como positivo y
   las metricas de precision y recall colapsan a cero. Se reporta por eso un
   umbral calibrado a la tasa base, que es la comparacion informativa.

Ademas, la busqueda de hiperparametros original optimizaba accuracy. Con un
1% de positivos, cualquier modelo alcanza ~0.99 de accuracy prediciendo
siempre "no es hit", por lo que la grilla no discriminaba entre candidatos.
Aca se optimiza AUC.
------------------------------------------------------------------------------
"""

import argparse
import io
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
BASES = RAIZ / "output" / "bases"
TABLAS = RAIZ / "output" / "tables"
FIGURAS = RAIZ / "output" / "figures"

SEMILLA = 444
TEST_SIZE = 0.30
PCTL = 99          # un "super hit" es una cancion en el top 1%

PREDICTORES = [
    "acousticness", "danceability", "duration_ms", "energy", "instrumentalness",
    "key", "liveness", "loudness", "mode", "speechiness", "tempo",
    "time_signature", "valence", "artist_count", "explicit_dummy", "song_age",
]

ETIQUETAS = {
    "acousticness": "Acousticness", "danceability": "Danceability",
    "duration_ms": "Duración (ms)", "energy": "Energy",
    "instrumentalness": "Instrumentalness", "key": "Key", "liveness": "Liveness",
    "loudness": "Loudness", "mode": "Mode", "speechiness": "Speechiness",
    "tempo": "Tempo", "time_signature": "Time signature", "valence": "Valence",
    "artist_count": "Cantidad de artistas", "explicit_dummy": "Explícita",
    "song_age": "Edad de la canción",
}


# ---------------------------------------------------------------- datos ----
def armar_panel():
    """Panel de canciones presentes en las dos bases, con etiqueta de super hit."""
    df18 = pd.read_csv(BASES / "base_2018.csv", low_memory=False)
    df19 = pd.read_csv(BASES / "base_2019.csv", low_memory=False)

    panel = (
        df18[["track_id", "track_name", "artist_name", "popularity_2018"] + PREDICTORES]
        .merge(df19[["track_id", "popularity_2019"]], on="track_id", how="inner")
        .drop_duplicates("track_id")
        .dropna(subset=["popularity_2018", "popularity_2019"] + PREDICTORES)
        .reset_index(drop=True)
    )

    umbral = np.percentile(panel["popularity_2019"], PCTL)
    panel["y"] = (panel["popularity_2019"] >= umbral).astype(int)

    print(f"Panel: {len(panel)} canciones presentes en ambas bases")
    print(f"Umbral top 1% de 2019: {umbral:.0f}")
    print(f"Super hits: {panel['y'].sum()} ({panel['y'].mean() * 100:.2f}%)")
    return panel, umbral


# ----------------------------------------------------------- power law ----
def ajustar_power_law(x):
    """Estima alpha y xmin por maxima verosimilitud. Devuelve (alpha, xmin)."""
    with redirect_stdout(io.StringIO()):       # el paquete imprime una barra de progreso
        fit = powerlaw.Fit(x[x > 0], discrete=True, verbose=False)
    return fit.power_law.alpha, fit.power_law.xmin


def supervivencia(x, alpha, xmin):
    """P(X >= x) bajo la ley de potencia ajustada.

    Para x < xmin la ley no esta definida (el ajuste solo describe la cola);
    se asigna 1, es decir, ninguna evidencia de pertenecer al extremo.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x < xmin, 1.0, np.power(np.maximum(x, xmin) / xmin, 1.0 - alpha))


# -------------------------------------------------------------- metricas ----
def umbral_tasa_base(score, tasa):
    """Umbral que clasifica como positivo la misma proporcion que la tasa base."""
    k = max(1, int(round(tasa * len(score))))
    return np.sort(score)[::-1][k - 1]


def metricas(nombre, y, pred, score):
    return {
        "Modelo": nombre,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "AUC": roc_auc_score(y, score),
    }


def imprimir(m, cm):
    print(f"\n  {m['Modelo']}")
    print(f"    matriz de confusion: {cm.tolist()}")
    print(f"    accuracy={m['Accuracy']:.4f}  precision={m['Precision']:.4f}  "
          f"recall={m['Recall']:.4f}  f1={m['F1']:.4f}  auc={m['AUC']:.4f}")


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="grilla de hiperparametros reducida")
    args = ap.parse_args()

    TABLAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)

    panel, umbral_hit = armar_panel()

    train, test = train_test_split(
        panel, test_size=TEST_SIZE, random_state=SEMILLA, stratify=panel["y"]
    )
    y_test = test["y"].to_numpy()
    tasa_base = train["y"].mean()
    print(f"Train: {len(train)}   Test: {len(test)}")

    filas, matrices, curvas = [], {}, {}

    # ---------------- Modelo 1: ley de potencia sobre popularidad pasada ----
    print("\n[1] Ley de potencia (maxima verosimilitud sobre popularidad 2018)")
    alpha, xmin = ajustar_power_law(train["popularity_2018"].to_numpy())
    print(f"    alpha={alpha:.4f}  xmin={xmin:.0f}")

    S_test = supervivencia(test["popularity_2018"], alpha, xmin)
    score_pl = 1.0 - S_test                      # percentil estimado, creciente en x
    corte_pl = umbral_tasa_base(score_pl, tasa_base)
    pred_pl = (score_pl >= corte_pl).astype(int)

    m = metricas("Ley de potencia", y_test, pred_pl, score_pl)
    cm = confusion_matrix(y_test, pred_pl)
    filas.append(m); matrices["Ley de potencia"] = cm; imprimir(m, cm)
    curvas["Ley de potencia"] = (*roc_curve(y_test, score_pl)[:2], m["AUC"])

    # ---------------- Modelo 2: Random Forest sobre audio features ----------
    print("\n[2] Random Forest (atributos acusticos)")
    grilla = (
        {"max_features": [3, "sqrt"], "n_estimators": [100]}
        if args.quick
        else {
            "max_samples": [0.25, 0.5, 0.75],
            "max_features": [3, 6, 9, "sqrt", None],
            "n_estimators": [100, 300],
        }
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    busqueda = GridSearchCV(
        # n_jobs=1 en el bosque y -1 en la busqueda: si ambos piden todos los
        # nucleos se pisan entre si (oversubscription) y la grilla tarda mas.
        RandomForestClassifier(bootstrap=True, random_state=SEMILLA, n_jobs=1),
        param_grid=grilla,
        scoring="roc_auc",                       # no accuracy: la clase es 1%
        cv=cv,
        verbose=1,
        n_jobs=-1,
    )
    busqueda.fit(train[PREDICTORES], train["y"])
    rf = busqueda.best_estimator_                # ya viene ajustado (refit=True)
    print(f"    mejores hiperparametros: {busqueda.best_params_}")
    print(f"    AUC en validacion cruzada: {busqueda.best_score_:.4f}")

    score_rf = rf.predict_proba(test[PREDICTORES])[:, 1]
    corte_rf = umbral_tasa_base(score_rf, tasa_base)
    pred_rf = (score_rf >= corte_rf).astype(int)

    m = metricas("Random Forest", y_test, pred_rf, score_rf)
    cm = confusion_matrix(y_test, pred_rf)
    filas.append(m); matrices["Random Forest"] = cm; imprimir(m, cm)
    curvas["Random Forest"] = (*roc_curve(y_test, score_rf)[:2], m["AUC"])

    # ---------------- Referencia: ambas fuentes de informacion juntas -------
    print("\n[3] Random Forest + popularidad pasada (referencia)")
    pred_mix = PREDICTORES + ["popularity_2018"]
    rf_mix = RandomForestClassifier(
        **{k: v for k, v in busqueda.best_params_.items()},
        bootstrap=True, random_state=SEMILLA, n_jobs=-1,
    )
    rf_mix.fit(train[pred_mix], train["y"])
    score_mix = rf_mix.predict_proba(test[pred_mix])[:, 1]
    corte_mix = umbral_tasa_base(score_mix, tasa_base)
    pred_mix_y = (score_mix >= corte_mix).astype(int)

    m = metricas("RF + popularidad pasada", y_test, pred_mix_y, score_mix)
    cm = confusion_matrix(y_test, pred_mix_y)
    filas.append(m); matrices["RF + popularidad pasada"] = cm; imprimir(m, cm)
    curvas["RF + popularidad pasada"] = (*roc_curve(y_test, score_mix)[:2], m["AUC"])

    # ---------------- Tabla de resultados ----------------
    res = pd.DataFrame(filas)
    (TABLAS / "resultados.tex").write_text(
        res.to_latex(
            index=False, float_format="%.3f", position="htbp",
            caption=(
                "Desempeño en la muestra de testeo para identificar canciones del "
                "top 1\\% de popularidad en 2019. El umbral de clasificación de cada "
                "modelo se calibra a la tasa base de positivos."
            ),
            label="tab:resultados",
        ),
        encoding="utf-8",
    )

    # ---------------- Matrices de confusion ----------------
    # Se reportan en formato plano (una fila por modelo) para que las cuatro
    # celdas sean directamente comparables entre modelos.
    conf = pd.DataFrame(
        [
            {
                "Modelo": nombre,
                "VN": cm[0, 0], "FP": cm[0, 1],
                "FN": cm[1, 0], "VP": cm[1, 1],
            }
            for nombre, cm in matrices.items()
        ]
    )
    (TABLAS / "matrices-confusion.tex").write_text(
        conf.to_latex(
            index=False, position="htbp",
            caption=(
                "Matrices de confusión en la muestra de testeo. VN: verdaderos "
                "negativos, FP: falsos positivos, FN: falsos negativos, "
                "VP: verdaderos positivos."
            ),
            label="tab:matrices",
        ),
        encoding="utf-8",
    )

    # ---------------- Parametros para citar en el texto ----------------
    # Cada valor que el informe menciona en prosa se escribe aca como macro de
    # LaTeX, para que ningun numero del texto este tipeado a mano.
    def miles(n):
        """Separador de miles que LaTeX compone bien: 114{,}714."""
        return f"{n:,}".replace(",", "{,}")

    macros = {
        "plalpha": f"{alpha:.2f}",
        "plxmin": f"{xmin:.0f}",
        "nPanel": miles(len(panel)),
        "nTrain": miles(len(train)),
        "nTest": miles(len(test)),
        "umbralHit": f"{umbral_hit:.0f}",
        "tasaBase": f"{tasa_base * 100:.2f}",
        # Positivos efectivamente predichos. No coinciden exactamente con la
        # tasa base porque la popularidad es entera: el score de la ley de
        # potencia toma pocos valores distintos y hay empates en el corte.
        "nHitsTest": miles(int(y_test.sum())),
        "nPosPL": miles(int(pred_pl.sum())),
        "nPosRF": miles(int(pred_rf.sum())),
        "nObjetivo": miles(int(round(tasa_base * len(test)))),
    }
    (TABLAS / "parametros.tex").write_text(
        "% Generado por 03_modelos.py. No editar a mano.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()),
        encoding="utf-8",
    )

    # ---------------- Figura: curvas ROC ----------------
    plt.figure(figsize=(7, 6))
    colores = {"Ley de potencia": "#c53030", "Random Forest": "#2b6cb0",
               "RF + popularidad pasada": "#2f855a"}
    for nombre, (fpr, tpr, auc) in curvas.items():
        plt.plot(fpr, tpr, label=f"{nombre} (AUC = {auc:.3f})",
                 color=colores[nombre], linewidth=1.8)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=.8, label="Azar")
    plt.xlabel("Tasa de falsos positivos", fontsize=12)
    plt.ylabel("Tasa de verdaderos positivos", fontsize=12)
    plt.title("Capacidad de discriminar súper hits", fontsize=13)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURAS / "roc-comparacion.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------- Figura: importancia de variables ----------------
    imp = (pd.Series(rf.feature_importances_, index=PREDICTORES)
           .sort_values().rename(index=ETIQUETAS))
    plt.figure(figsize=(8, 6))
    plt.barh(imp.index, imp.to_numpy(), color="#2b6cb0")
    plt.xlabel("Importancia (reducción media de impureza)", fontsize=12)
    plt.title("Atributos acústicos más informativos (Random Forest)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURAS / "importancia-variables.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + res.to_string(index=False))
    print(f"\nListo. Tablas en {TABLAS}, figuras en {FIGURAS}")


if __name__ == "__main__":
    main()
