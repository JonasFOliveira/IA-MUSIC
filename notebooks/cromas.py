import os

import h5py
import librosa

# Diretório com os arquivos de áudio
audio_dir = "data/musics/wav"
output_dir = "data/musics/croma"
os.makedirs(output_dir, exist_ok=True)

# Parâmetros padrão para extração
sr = 22050  # taxa de amostragem
hop_length = 512  # passo da janela
with h5py.File("todos_os_cromagramas.h5", "w") as f:
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(audio_dir, filename)
            y, _ = librosa.load(filepath, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

            # Nome do dataset = nome do arquivo sem extensão
            dataset_name = os.path.splitext(filename)[0]
            f.create_dataset(dataset_name, data=chroma)
