# Big Data — Trabajos Prácticos (Otoño 2025)

Cada TP tiene la misma estructura:

```
TPn/
├── documents/   consigna, informe entregado y main.tex (fuente LaTeX)
├── input/       datos que lee el código
├── output/      gráficos, tablas .tex y bases generadas
└── programs/    notebook
```

## Cómo correr

Los notebooks resuelven las rutas solos: abrilos desde `programs/` y leen de
`../input/` y escriben en `../output/`. No hace falta tocar ningún `os.chdir`.

## Orden de ejecución (los TPs están encadenados)

```
TP1  (EPH cruda)  ──>  output/base.csv
                            │
TP3  (base.csv)   ──>  output/respondieron.csv
                            │
                            ├──>  TP4  (respondieron.csv)
                            └──>  TP5  (respondieron.csv)

TP2  lee la EPH cruda directamente, es independiente.
```

`base.csv` y `respondieron.csv` ya están puestos en el `input/` que los necesita,
así que podés correr cualquier TP por separado sin rehacer los anteriores.

## Informes

Cada `documents/main.tex` compila por sí solo y toma las figuras de `../output/`
vía `\graphicspath`. Necesita el paquete `babel` con español (Overleaf y MacTeX
lo traen). El TP4 usa además `images.png` (el logo), que está en su propio
`documents/`.

Se conservan los PDF originales entregados, con el sufijo `_entregado`, y en el
TP1 también la versión con la devolución del profesor (`_corregido`).
