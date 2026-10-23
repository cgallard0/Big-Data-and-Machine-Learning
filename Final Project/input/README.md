# Datos crudos

No modificar los archivos de esta carpeta. Todo lo que el código genera va a `output/`.

## `data_2018.csv`, `data_2019.csv`

Score de popularidad y atributos acústicos de canciones de Spotify, medidos en 2018 y
2019 respectivamente.

- **Fuente:** [Spotify Audio Features](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features),
  publicado en Kaggle por Tomi Gelo.
- **Nivel de observación:** canción.
- **Columnas:** `artist_name`, `track_id`, `track_name`, `popularity` y los atributos
  acústicos `acousticness`, `danceability`, `duration_ms`, `energy`, `instrumentalness`,
  `key`, `liveness`, `loudness`, `mode`, `speechiness`, `tempo`, `time_signature`,
  `valence`.

Estas bases fueron construidas originalmente con el endpoint de *audio features* de la
API de Spotify, **que desde entonces fue discontinuado**. Los atributos acústicos ya no
pueden volver a descargarse desde la fuente original, y por eso estos CSV se conservan
en el repositorio en lugar de regenerarse.

El resto de las variables que usa el análisis (metadata de álbum y de artista, edad de
la canción) sí se obtienen de endpoints que siguen activos, y las agrega
`programs/01_construir_base.py`.
