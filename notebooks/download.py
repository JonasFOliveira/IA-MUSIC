import os
import re
from pathlib import Path

import pandas as pd
from IPython.display import clear_output
from youtube_search import YoutubeSearch

# sucess = pd.read_csv("data/sucess_sample.csv")
# # sucess.head(2)
# fail = pd.read_csv("data/fail_sample.csv")

dataframe = pd.read_csv("data/dataframe_15_anos.csv")


def download(track, artist, locate) -> None:

    # Limpa os nomes
    artistclean = re.sub(r"[^a-zA-Z0-9]", "_", artist)
    trackclean = re.sub(r"[^a-zA-Z0-9]", "_", track)
    filename = f"{trackclean}_{artistclean}"
    if Path(f"{locate}/wav/{filename}.wav").is_file():
        return None
    print(filename)
    # Pesquisa no YouTube
    search_query = track + " - " + artist
    results = YoutubeSearch(search_query, max_results=1).to_dict()
    if not results:
        print(f"Nenhum resultado encontrado para {search_query}")

    url_ = "https://www.youtube.com" + results[0]["url_suffix"]

    # Define caminhos
    webm_path = f"{locate}/webm/{filename}.webm"
    wav_path = f"{locate}/wav/{filename}.wav"

    # Baixa o arquivo .webm
    os.system(f"yt-dlp -f bestaudio -o {webm_path} {url_}")

    # Converte para .wav
    os.system(f"ffmpeg -i {webm_path} -ar 44100 -ac 1 -y {wav_path}")

    # limpar o terminal
    clear_output()

    # # (Opcional) Remove o arquivo .webm após conversão
    # try:
    #     if Path(webm_path).is_file:
    #         os.remove(webm_path)
    # except:
    #     pass


dataframe.apply(lambda x: download(x["track"], x["artist"], "data/musics"), axis=1)

# fail.apply(lambda x: download(x["track"], x["artist"], "data/musics"), axis=1)
