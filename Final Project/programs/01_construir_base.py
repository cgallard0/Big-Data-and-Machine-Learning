"""
01_construir_base.py
====================
Construye las bases de trabajo a partir de los datos crudos de Kaggle,
enriqueciendolas con metadata de album y artista tomada de la API de Spotify.

Entrada : input/data_2018.csv , input/data_2019.csv
Salida  : output/bases/base_2018.csv , output/bases/base_2019.csv

Las credenciales de la API se leen de variables de entorno. NUNCA se
escriben en el codigo (ver README).

    export SPOTIFY_CLIENT_ID=...
    export SPOTIFY_CLIENT_SECRET=...

Uso:
    python 01_construir_base.py            # no hace nada si las bases ya existen
    python 01_construir_base.py --force    # reconstruye desde cero
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# --- rutas relativas al repo, no absolutas: el proyecto es portable -------
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
INPUT = RAIZ / "input"
BASES = RAIZ / "output" / "bases"

BATCH = 50          # maximo de ids por request admitido por la API
PAUSA = 1.0         # segundos entre requests, para no exceder el rate limit


def credenciales():
    """Lee las credenciales de la API desde el entorno."""
    cid = os.environ.get("SPOTIFY_CLIENT_ID")
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret:
        sys.exit(
            "ERROR: faltan las credenciales de la API de Spotify.\n"
            "  export SPOTIFY_CLIENT_ID=tu_client_id\n"
            "  export SPOTIFY_CLIENT_SECRET=tu_client_secret\n"
            "Se obtienen en https://developer.spotify.com/dashboard"
        )
    return cid, secret


def traer_metadata(track_ids, cid, secret):
    """Consulta la API de Spotify y devuelve un DataFrame con metadata."""
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(client_id=cid, client_secret=secret),
        requests_timeout=100,
        retries=5,
    )

    filas = []
    total = (len(track_ids) + BATCH - 1) // BATCH
    for i, x in enumerate(range(0, len(track_ids), BATCH), start=1):
        lote = track_ids[x : x + BATCH]
        try:
            resultados = sp.tracks(lote)["tracks"]
        except Exception as e:                      # un lote que falla no corta la corrida
            print(f"  lote {i}/{total} fallo ({e}); se omite")
            continue

        for track in resultados:
            if track is None:                      # id invalido o cancion dada de baja
                continue
            album = track["album"]
            filas.append(
                {
                    "track_id": track["id"],
                    "artist_id": track["artists"][0]["id"],
                    "album_id": album["id"],
                    "album_name": album["name"],
                    "date": album["release_date"],
                    "album_type": album["album_type"],
                    "track_number": track["track_number"],
                    "explicit": track["explicit"],
                    "artist_count": len(track["artists"]),
                }
            )

        if i % 20 == 0:
            print(f"  {i}/{total} lotes ({len(filas)} canciones)")
        time.sleep(PAUSA)

    return pd.DataFrame(filas)


def limpiar(df, anio, col_pop):
    """Limpieza y variables derivadas para una base-anio."""
    n0 = len(df)

    df["explicit_dummy"] = df["explicit"].astype(int)

    # Fecha de publicacion: lo no parseable queda como NaT y se descarta.
    # (El codigo original comparaba la fecha contra el entero 0, lo cual
    #  nunca era verdadero y por lo tanto no filtraba nada.)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df = df[df["date"].notna()]
    n1 = len(df)

    df["song_age"] = anio - df["date"].dt.year

    # Una cancion publicada despues del anio de medicion es una reversion
    # o una reedicion: su edad seria negativa y sesgaria el analisis.
    df = df[df["date"].dt.year <= anio]
    n2 = len(df)

    df = df.dropna(subset=[col_pop])
    print(
        f"  {anio}: {n0} -> {n1} (con fecha) -> {n2} (publicadas <= {anio}) "
        f"-> {len(df)} (con popularidad)"
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="reconstruye las bases aunque ya existan")
    args = ap.parse_args()

    BASES.mkdir(parents=True, exist_ok=True)
    salidas = [BASES / "base_2018.csv", BASES / "base_2019.csv"]

    if all(p.exists() for p in salidas) and not args.force:
        print("Las bases ya existen en output/bases/. Nada que hacer.")
        print("Usa --force para reconstruirlas (requiere la API de Spotify).")
        return

    print("Leyendo datos crudos...")
    df_2018 = pd.read_csv(INPUT / "data_2018.csv")
    df_2019 = pd.read_csv(INPUT / "data_2019.csv")

    df_2018 = df_2018.rename(columns={"popularity": "popularity_2018"})
    df_2019 = df_2019.rename(columns={"popularity": "popularity_2019"})

    df_2018 = df_2018.drop_duplicates(subset="track_id")
    df_2019 = df_2019.drop_duplicates(subset="track_id")
    print(f"  2018: {len(df_2018)} canciones unicas")
    print(f"  2019: {len(df_2019)} canciones unicas")

    ids = pd.concat([df_2018["track_id"], df_2019["track_id"]]).drop_duplicates()
    track_ids = ids.tolist()
    print(f"  {len(track_ids)} ids unicos a consultar en la API")

    cid, secret = credenciales()
    print("Consultando la API de Spotify (esto tarda)...")
    info = traer_metadata(track_ids, cid, secret)
    print(f"  metadata obtenida para {len(info)} canciones")

    df_2018 = df_2018.merge(info, on="track_id", how="inner")
    df_2019 = df_2019.merge(info, on="track_id", how="inner")

    print("Limpiando...")
    df_2018 = limpiar(df_2018, 2018, "popularity_2018")
    df_2019 = limpiar(df_2019, 2019, "popularity_2019")

    df_2018.to_csv(BASES / "base_2018.csv", index=False)
    df_2019.to_csv(BASES / "base_2019.csv", index=False)
    print(f"Listo. Bases guardadas en {BASES}")


if __name__ == "__main__":
    main()
