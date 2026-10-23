"""
02_descriptivas.py
==================
Estadistica descriptiva del panel de canciones y evidencia grafica de que
la popularidad sigue una distribucion de cola pesada.

Entrada : output/bases/base_2018.csv , output/bases/base_2019.csv
Salida  : output/tables/estadistica-descriptiva.tex
          output/tables/top-canciones.tex
          output/figures/popularidad.png
          output/figures/power-law-ajuste.png

Uso:
    python 02_descriptivas.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # backend sin pantalla: funciona en cualquier maquina
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

def escapar_latex(texto):
    """Escapa los caracteres con significado especial en LaTeX.

    Hace falta porque los nombres de canciones y artistas contienen
    ampersands y otros simbolos (p. ej. "Selena Gomez, Ozuna & Cardi B"),
    y a partir de pandas 2.x `to_latex` ya no escapa por defecto.
    """
    reemplazos = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for viejo, nuevo in reemplazos.items():
        texto = texto.replace(viejo, nuevo)
    return texto


def formatear_p(p):
    """p-valor para el texto del informe.

    Un p-valor de 1e-12 impreso con tres decimales queda como "0.000", que
    se lee como si fuera cero. Por debajo de 0.001 se reporta como cota.
    """
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
BASES = RAIZ / "output" / "bases"
TABLAS = RAIZ / "output" / "tables"
FIGURAS = RAIZ / "output" / "figures"


def main():
    TABLAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)

    df_2018 = pd.read_csv(BASES / "base_2018.csv", low_memory=False)
    df_2019 = pd.read_csv(BASES / "base_2019.csv", low_memory=False)

    p18 = df_2018["popularity_2018"]
    p19 = df_2019["popularity_2019"]

    # ---------------- Tabla 1: descriptivas ----------------
    desc = pd.DataFrame(
        {
            "Año": [2018, 2019],
            "N": [p18.count(), p19.count()],
            "Media": [p18.mean(), p19.mean()],
            "Desvío": [p18.std(ddof=1), p19.std(ddof=1)],
            "p50": [p18.median(), p19.median()],
            "p99": [np.percentile(p18, 99), np.percentile(p19, 99)],
            "Max": [p18.max(), p19.max()],
        }
    )
    (TABLAS / "estadistica-descriptiva.tex").write_text(
        desc.to_latex(
            index=False,
            caption="Estadística descriptiva del score de popularidad de Spotify.",
            label="tab:descriptivas",
            float_format="%.2f",
            position="htbp",
        ),
        encoding="utf-8",
    )
    print(desc.to_string(index=False))

    # ---------------- Tabla 2: top 3 de cada anio ----------------
    def top3(df, col):
        r = df.nlargest(3, col)
        return [
            f"{escapar_latex(str(s))} --- {escapar_latex(str(a))} ({int(p)})"
            for s, a, p in zip(r["track_name"], r["artist_name"], r[col])
        ]

    top = pd.DataFrame({"2018": top3(df_2018, "popularity_2018"),
                        "2019": top3(df_2019, "popularity_2019")})
    (TABLAS / "top-canciones.tex").write_text(
        top.to_latex(
            index=False,
            caption="Canciones más populares de cada año (score entre paréntesis).",
            label="tab:top",
            position="htbp",
            escape=False,          # ya escapado por escapar_latex()
            # Columnas de ancho fijo: los titulos son largos y con columnas
            # elasticas la tabla se desbordaba del margen.
            column_format="p{.45\\textwidth}p{.45\\textwidth}",
        ),
        encoding="utf-8",
    )
    print("\n", top.to_string(index=False))

    # ---------------- Figura 1: distribucion ----------------
    bins = np.linspace(min(p18.min(), p19.min()), max(p18.max(), p19.max()), 31)
    plt.figure(figsize=(10, 6))
    sns.histplot(p18, bins=bins, color="#2b6cb0", alpha=.45, label="2018", stat="density")
    sns.histplot(p19, bins=bins, color="#c53030", alpha=.45, label="2019", stat="density")
    sns.kdeplot(p18, color="#2b6cb0", bw_adjust=1)
    sns.kdeplot(p19, color="#c53030", bw_adjust=1)
    plt.title("Distribución de la popularidad de las canciones", fontsize=16)
    plt.xlabel("Score de popularidad", fontsize=13)
    plt.ylabel("Densidad", fontsize=13)
    plt.legend(title="Anio", title_fontsize=12, fontsize=11)
    plt.figtext(
        .5, .005,
        "Nota: el score de popularidad de Spotify va de 0 (menos popular) a 100 (más popular).",
        ha="center", fontsize=11, style="italic",
    )
    plt.tight_layout(rect=[0, .04, 1, 1])
    plt.savefig(FIGURAS / "popularidad.png", dpi=300)
    plt.close()

    # ---------------- Figura 2: ajuste power law ----------------
    import powerlaw
    datos = p18[p18 > 0].to_numpy()
    fit = powerlaw.Fit(datos, discrete=True, verbose=False)
    print(f"\nAjuste power law en 2018: alpha={fit.power_law.alpha:.4f}, "
          f"xmin={fit.power_law.xmin:.0f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    fit.plot_ccdf(ax=ax, color="black", linewidth=1.6, label="Datos empíricos")
    fit.power_law.plot_ccdf(ax=ax, color="#c53030", linestyle="--",
                            label=f"Power law ajustada ($\\alpha$={fit.power_law.alpha:.2f})")
    fit.lognormal.plot_ccdf(ax=ax, color="#2b6cb0", linestyle=":",
                            label="Lognormal ajustada")
    ax.set_xlabel("Score de popularidad (x)", fontsize=12)
    ax.set_ylabel(r"$P(X \geq x)$", fontsize=12)
    ax.set_title("Función de supervivencia de la popularidad (2018)", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURAS / "power-law-ajuste.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Test de comparacion de distribuciones: respalda el supuesto de cola pesada
    R, p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    print(f"Comparacion power law vs lognormal: R={R:.3f}, p={p:.4f}")
    (TABLAS / "comparacion-distribuciones.tex").write_text(
        "% Cociente de verosimilitud normalizado power law vs lognormal\n"
        f"% R = {R:.4f}, p = {p:.4f}\n"
        f"\\newcommand{{\\plR}}{{{R:.2f}}}\n"
        f"\\newcommand{{\\plp}}{{{formatear_p(p)}}}\n",
        encoding="utf-8",
    )
    print(f"\nListo. Tablas en {TABLAS}, figuras en {FIGURAS}")


if __name__ == "__main__":
    main()
