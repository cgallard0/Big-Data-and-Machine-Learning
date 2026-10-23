# ¿Cómo predecir un súper hit?

Un análisis comparativo de modelos de popularidad musical sobre datos de Spotify.

**Catalina Gallardo** y **Manuel Elorga** — E337: Big Data, Universidad de San Andrés.

---

## El problema

La popularidad musical en plataformas de streaming está extremadamente concentrada:
la canción promedio de nuestra muestra tiene un score de popularidad de 24 sobre 100,
pero el 1% más popular arranca en 70. Esa desigualdad plantea la pregunta del trabajo:
**¿se puede anticipar qué canciones van a llegar al 1% más popular, y con qué información?**

Comparamos dos enfoques sobre un panel de 114.714 canciones observadas en 2018 y 2019:

1. **Ley de potencia** ajustada por máxima verosimilitud sobre la popularidad del año
   anterior. No mira a la canción: usa solo la estructura estadística de la distribución.
2. **Random Forest** sobre los atributos acústicos de la canción (*danceability*,
   *energy*, *valence*, etc.). No usa la popularidad pasada.

## El resultado

| Modelo | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Ley de potencia (popularidad 2018) | 0.995 | 0.706 | 0.816 | 0.757 | **0.997** |
| Random Forest (atributos acústicos) | 0.980 | 0.048 | 0.051 | 0.049 | **0.761** |
| Random Forest + popularidad pasada | 0.995 | 0.744 | 0.748 | 0.746 | 0.986 |

La popularidad pasada identifica a los súper hits casi perfectamente: recupera 288 de los
353 súper hits de la muestra de testeo. Los atributos acústicos ordenan a las canciones
mejor que el azar, pero para el éxito extremo son casi inútiles: con el umbral calibrado
recuperan 18 de esos 353. El resultado se mantiene con cualquier umbral de popularidad
(ver `04_robustez.py`). Cuando se le entrega la popularidad pasada, el mismo Random Forest
alcanza el desempeño de la ley de potencia: la limitación está en la información
disponible, no en el método.

**El éxito extremo es muy persistente en el tiempo, pero no se explica por las
características sonoras de la canción.** No parece haber una receta acústica del hit.

## Estructura del repositorio

```
.
├── documents/          Informe en LaTeX
│   ├── informe.tex         fuente principal
│   ├── referencias.bib     bibliografía
│   └── original/           informe y presentación entregados, y tablas previas
├── input/              Datos crudos (no se modifican nunca)
│   ├── data_2018.csv
│   └── data_2019.csv
├── output/             Todo lo generado por el código
│   ├── bases/              bases procesadas (no versionadas, ver abajo)
│   ├── figures/            figuras del informe
│   └── tables/             tablas en LaTeX, importadas por informe.tex
└── programs/           Código, numerado en orden de ejecución
    ├── 01_construir_base.py
    ├── 02_descriptivas.py
    ├── 03_modelos.py
    ├── 04_robustez.py
    ├── requirements.txt
    └── original/           notebooks entregados (sin las credenciales)
```

Las tablas y los parámetros estimados se escriben como `.tex` y se importan desde
`informe.tex` con `\input{}`. Ningún número del informe está escrito a mano: al volver
a correr el código, el informe se actualiza solo.

## Cómo reproducirlo

### 1. Dependencias

```bash
pip install -r programs/requirements.txt
```

### 2. Datos

`input/` ya trae los datos crudos de [Kaggle](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features).
Las bases procesadas (`output/bases/`) **no están versionadas** porque pesan ~60 MB.
Hay dos formas de obtenerlas:

**Opción A — reconstruirlas desde la API de Spotify.** Requiere credenciales propias,
que se obtienen gratis en el [dashboard de Spotify](https://developer.spotify.com/dashboard):

```bash
export SPOTIFY_CLIENT_ID=tu_client_id
export SPOTIFY_CLIENT_SECRET=tu_client_secret
python programs/01_construir_base.py
```

Tarda un buen rato: son unos 180.000 identificadores consultados en lotes de 50.

**Opción B — usar las bases ya construidas**, si las tenés de una corrida anterior.
Copiarlas a `output/bases/` y saltar directamente al paso 3.

### 3. Análisis

```bash
python programs/02_descriptivas.py    # tablas descriptivas y figuras
python programs/03_modelos.py         # modelos y comparación  (~15 min)
python programs/04_robustez.py        # chequeo de robustez
```

`03_modelos.py` acepta `--quick` para usar una grilla de hiperparámetros reducida si
solo se quiere probar que corre.

### 4. Informe

```bash
cd documents
pdflatex informe.tex && bibtex informe && pdflatex informe.tex && pdflatex informe.tex
```

## Notas sobre los datos

- El endpoint de *audio features* de la API de Spotify fue discontinuado, así que los
  atributos acústicos **no se pueden volver a descargar** desde la fuente original. Por
  eso el repositorio conserva los CSV crudos en `input/`.
- El *popularity score* de Spotify es una variable propietaria: la plataforma no
  documenta cómo se construye. Se sabe que combina reproducciones recientes con
  recencia, pero su definición exacta es privada.

## Correcciones respecto de la versión entregada

Esta versión corrige tres problemas de la implementación original que **cambian las
conclusiones del trabajo**. Están documentados en el apéndice del informe y en los
comentarios de `03_modelos.py`. Los notebooks originales se conservan en
`programs/original/` tal como fueron entregados, con una sola excepción: las
credenciales de la API de Spotify que estaban escritas en el código fueron reemplazadas
por `<REDACTADO>`.

1. **Circularidad.** El modelo de ley de potencia calculaba la probabilidad a partir de
   `popularity_2019`, la misma variable con la que se define la etiqueta de súper hit.
   Predecía la variable con la variable misma. Ahora se evalúa sobre `popularity_2018`,
   de modo que la predicción es fuera de muestra en el tiempo.
2. **Fórmula.** Se usaba `(x/x99)^-(alpha+1)`, con el exponente equivocado y el
   percentil 99 en lugar de `xmin`. La expresión resultante era *decreciente* en la
   popularidad y tomaba valores mayores que 1, imposible para una probabilidad. La
   función de supervivencia correcta es `(x/xmin)^(1-alpha)`.
3. **Umbral de decisión.** Con una tasa base del 1%, cortar en 0.5 hace que casi ningún
   caso se clasifique como positivo y precisión y recall colapsen a cero. Ahora el
   umbral de cada modelo se calibra a la tasa base.

Además: la búsqueda de hiperparámetros optimizaba *accuracy*, que con un 1% de positivos
no discrimina entre candidatos (cualquier modelo llega a 0.99 prediciendo siempre "no es
hit"); ahora optimiza AUC. Las credenciales de la API, que estaban escritas en el
notebook, se leen de variables de entorno. Las rutas absolutas se reemplazaron por rutas
relativas al repositorio.

Un hallazgo adicional: el test formal de comparación de distribuciones favorece a una
**lognormal** sobre la ley de potencia (R = -83.7, p < 0.001). La popularidad tiene cola
pesada, pero la ley de potencia es un resumen paramétrico útil de la cola, no el proceso
generador. El informe lo discute explícitamente.

## Licencia

Código bajo licencia MIT. Los datos pertenecen a sus respectivas fuentes y se incluyen
únicamente con fines de reproducibilidad académica.
