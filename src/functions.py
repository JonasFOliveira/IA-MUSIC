import math

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write as write_wav
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, mean_squared_error


def extract_chroma(path):
    # Carregar o áudio
    y, sr = librosa.load(path)
    # Extrair o cromagrama
    return librosa.feature.chroma_cqt(y=y, sr=sr)


def chroma_to_notes(chromagram, sr=22050, hop_length=512, min_duration=0.2):
    """
    Converte um cromagrama para uma sequência de acordes com tempos de início e fim,
    removendo acordes de curta duração.

    Args:
        chromagram (np.ndarray): O cromagrama (12 x T).
        sr (int): Taxa de amostragem do áudio original.
        hop_length (int): Número de samples entre quadros do cromagrama.
        min_duration (float): Duração mínima em segundos para um acorde ser considerado relevante.

    Returns:
        list: Uma lista de tuplas no formato (chord, start_time_sec, end_time_sec).
    """
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    num_frames = chromagram.shape[1]
    chords = []
    current_chord = None
    current_start_time = 0.0

    def get_chord_from_chroma(chroma_vector, threshold=0.8):
        """Simplificada: Retorna a nota com maior energia se acima do threshold."""
        max_val = np.max(chroma_vector)
        if max_val > threshold:
            max_index = np.argmax(chroma_vector)
            return notes[max_index]
        return None

    for i in range(num_frames):
        time_sec = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
        chord = get_chord_from_chroma(chromagram[:, i])

        if chord != current_chord:
            if current_chord is not None:
                duration = time_sec - current_start_time
                if duration >= min_duration:  # Verifica a duração mínima
                    chords.append((current_chord, current_start_time, time_sec))
            current_chord = chord
            current_start_time = time_sec

    # Adiciona o último acorde (se durar o suficiente)
    if current_chord is not None:
        duration = (
            librosa.frames_to_time(num_frames, sr=sr, hop_length=hop_length)
            - current_start_time
        )
        if duration >= min_duration:
            chords.append(
                (
                    current_chord,
                    current_start_time,
                    librosa.frames_to_time(num_frames, sr=sr, hop_length=hop_length),
                )
            )

    return chords


def generate_piano_sound(frequency, duration, sample_rate=44100, amplitude=0.1):
    """Gera uma onda senoidal com harmônicos para simular uma nota de piano."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    note = amplitude * np.sin(2 * np.pi * frequency * t)
    harmonic1 = 0.5 * amplitude * np.sin(2 * np.pi * 2 * frequency * t)
    harmonic2 = 0.25 * amplitude * np.sin(2 * np.pi * 4 * frequency * t)
    return (note + harmonic1 + harmonic2).astype(np.float32)


def get_note_frequency(note_name):
    """Retorna a frequência de uma nota musical (oitava 4 como padrão)."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    base_frequency_a4 = 440.0
    if len(note_name) > 0 and note_name[0].upper() in notes:
        n = notes.index(note_name[0].upper())
        accidental = 0
        if len(note_name) > 1:
            if note_name[1] == "#":
                accidental = 1
            elif note_name[1] == "b":
                accidental = -1
        octave = 4
        if len(note_name) > 2 and note_name[2].isdigit():
            octave = int(note_name[2])
        elif len(note_name) > 2 and note_name[2] == "-":
            if len(note_name) > 3 and note_name[3].isdigit():
                octave = -int(note_name[3])
                base_frequency_a4 /= 2**4

        semitones_from_a4 = (n - 9) + accidental + (octave - 4) * 12
        return base_frequency_a4 * (2 ** (semitones_from_a4 / 12))
    return None


import contextlib
import wave


def get_wav_duration(wav_path):
    """Retorna a duração do arquivo WAV em segundos."""
    with contextlib.closing(wave.open(wav_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def transform_chords_to_piano(
    chord_sequence,
    original_wav_path,
    output_wav_path="piano_music.wav",
    sample_rate=44100,
):
    """
    Gera uma música de piano usando apenas os acordes, sem silêncios entre eles,
    mantendo a mesma duração da música original (detectada a partir de um .wav).
    """
    if not chord_sequence:
        raise ValueError("A sequência de acordes está vazia.")

    # Detectar duração do áudio original automaticamente
    total_duration_sec = get_wav_duration(original_wav_path)
    print(f"Duração da música original: {total_duration_sec:.2f} segundos")

    chord_map_notes = {
        "C": ["C4", "E4", "G4"],
        "C#": ["C#4", "F4", "G#4"],
        "Db": ["Db4", "F4", "Ab4"],
        "D": ["D4", "F#4", "A4"],
        "D#": ["D#4", "G4", "A#4"],
        "Eb": ["Eb4", "G4", "Bb4"],
        "E": ["E4", "G#4", "B4"],
        "F": ["F4", "A4", "C5"],
        "F#": ["F#4", "A#4", "C#5"],
        "Gb": ["Gb4", "Bb4", "Db5"],
        "G": ["G4", "B4", "D5"],
        "G#": ["G#4", "C5", "D#5"],
        "Ab": ["Ab4", "C5", "Eb5"],
        "A": ["A4", "C#5", "E5"],
        "A#": ["A#4", "D5", "F5"],
        "Bb": ["Bb4", "D5", "F5"],
        "B": ["B4", "D#5", "F#5"],
        "E#": ["E#4", "Gx4", "B#4"],
        "Cmaj7": ["C4", "E4", "G4", "B4"],
        "Cm": ["C4", "Eb4", "G4"],
        "C#m": ["C#4", "E4", "G#4"],
        "Dbm": ["Db4", "Fb4", "Ab4"],
        "Dm": ["D4", "F4", "A4"],
        "D#m": ["D#4", "F#4", "A#4"],
        "Ebm": ["Eb4", "Gb4", "Bb4"],
        "Em": ["E4", "G4", "B4"],
        "Fm": ["F4", "Ab4", "C5"],
        "F#m": ["F#4", "A4", "C#5"],
        "Gbm": ["Gb4", "Bbb4", "Db5"],  # Bbb é A natural
        "Gm": ["G4", "Bb4", "D5"],
        "G#m": ["G#4", "B4", "D#5"],
        "Abm": ["Ab4", "Cb5", "Eb5"],  # Cb é B natural
        "Am": ["A4", "C5", "E5"],
        "A#m": ["A#4", "C#5", "F5"],
        "Bbm": ["Bb4", "Db5", "F5"],
        "Bm": ["B4", "D5", "F#5"],
        "C7": ["C4", "E4", "G4", "Bb4"],
        "C#7": ["C#4", "F4", "G#4", "B4"],
        "Db7": ["Db4", "F4", "Ab4", "Cb5"],
        "D7": ["D4", "F#4", "A4", "C5"],
        "D#7": ["D#4", "G4", "A#4", "C#5"],
        "Eb7": ["Eb4", "G4", "Bb4", "Db5"],
        "E7": ["E4", "G#4", "B4", "D5"],
        "F7": ["F4", "A4", "C5", "Eb5"],
        "F#7": ["F#4", "A#4", "C#5", "E5"],
        "Gb7": ["Gb4", "Bb4", "Db5", "Fb5"],
        "G7": ["G4", "B4", "D5", "F5"],
        "G#7": ["G#4", "C5", "D#5", "F#5"],
        "Ab7": ["Ab4", "C5", "Eb5", "Gb5"],
        "A7": ["A4", "C#5", "E5", "G5"],
        "A#7": ["A#4", "D5", "F5", "G#5"],
        "Bb7": ["Bb4", "D5", "F5", "Ab5"],
        "B7": ["B4", "D#5", "F#5", "A5"],
        "Csus4": ["C4", "F4", "G4"],
        "Dsus4": ["D4", "G4", "A4"],
        "Esus4": ["E4", "A4", "B4"],
        "Fsus4": ["F4", "Bb4", "C5"],
        "Gsus4": ["G4", "C5", "D5"],
        "Asus4": ["A4", "D5", "E5"],
        "Bsus4": ["B4", "E5", "F#5"],
        "C#sus4": ["C#4", "F#4", "G#4"],
        "D#sus4": ["D#4", "G#4", "A#4"],
        "Fs#us4": ["F#4", "B4", "C#5"],
        "Gs#us4": ["G#4", "C#5", "D#5"],
        "As#us4": ["A#4", "D#5", "F5"],
        "Bs#us4": ["B#4", "E#5", "Fx5"],
    }

    original_total_sound_duration = sum(end - start for _, start, end in chord_sequence)
    if original_total_sound_duration <= 0:
        raise ValueError("A duração total dos acordes é inválida.")

    audio_data = np.array([], dtype=np.float32)

    for chord, start, end in chord_sequence:
        original_duration = end - start
        proportion = original_duration / original_total_sound_duration
        new_duration = proportion * total_duration_sec
        new_samples = int(new_duration * sample_rate)

        if chord in chord_map_notes:
            freqs = [get_note_frequency(n) for n in chord_map_notes[chord]]
            chord_audio = np.zeros(new_samples, dtype=np.float32)
            for f in freqs:
                chord_audio += generate_piano_sound(f, new_duration, sample_rate)
            chord_audio /= len(freqs)
        else:
            print(f"Acorde não mapeado: {chord}. Usando silêncio.")
            chord_audio = np.zeros(new_samples, dtype=np.float32)

        audio_data = np.concatenate((audio_data, chord_audio))

    # Ajustar para bater exatamente com a duração desejada
    total_samples_target = int(total_duration_sec * sample_rate)
    audio_data = np.resize(audio_data, total_samples_target)

    audio_data = np.clip(audio_data, -1.0, 1.0)
    write_wav(output_wav_path, sample_rate, audio_data)
    print(f"Áudio de piano gerado com duração igual ao original: {output_wav_path}")
    return output_wav_path


def extract_chroma(audio_path, sr=22050, hop_length=512):
    """Extrai o cromagrama de um arquivo de áudio."""
    y, sr = librosa.load(audio_path, sr=sr)
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length), sr


import librosa
import librosa.display
import matplotlib.pyplot as plt


def visualizar_cromagrama(original_chroma, chromas_2, time=(75, 90)):
    """
    Função para visualizar o cromagrama original e o cromagrama gerado, com corte na seção central.

    Parâmetros:
    - original_chroma: Cromagrama original a ser visualizado
    - chromas_2: Cromagrama do áudio gerado a ser visualizado
    - time: intervalo de tempo em segundos
    """

    # --- Visualização do cromagrama original ---
    plt.figure(figsize=(12, 4))  # Aumenta a largura da imagem (width=12)
    plt.xlim(time[0], time[1])
    librosa.display.specshow(original_chroma, y_axis="chroma", x_axis="time")
    plt.colorbar(format="%+.2f dB")
    plt.title("Cromagrama Original")
    plt.tight_layout()
    plt.show()

    # --- Visualização do cromagrama do áudio gerado ---
    plt.figure(figsize=(12, 4))  # Aumenta a largura da imagem (width=12)
    plt.xlim(time[0], time[1])
    librosa.display.specshow(chromas_2, y_axis="chroma", x_axis="time")
    plt.colorbar(format="%+0.2f dB")
    plt.title("Cromagrama do Áudio Gerado")
    plt.tight_layout()
    plt.show()


import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def calcular_similaridade(cromagrama_original, cromagrama_gerado, limiar_db=-30):
    """
    Função para calcular as métricas de similaridade entre cromagramas ignorando valores muito baixos.

    Parâmetros:
    - cromagrama_original: Cromagrama original (matriz de dB)
    - cromagrama_gerado: Cromagrama do áudio gerado (matriz de dB)
    - limiar_db: Limite em dB abaixo do qual os valores serão ignorados. O padrão é -30 dB.

    Retorno:
    - Um dicionário com as métricas de similaridade.
    """

    # Aplicando o limiar (ignorando valores abaixo de -30 dB)
    cromagrama_original_filtrado = np.where(
        cromagrama_original > limiar_db, cromagrama_original, 0
    )
    cromagrama_gerado_filtrado = np.where(
        cromagrama_gerado > limiar_db, cromagrama_gerado, 0
    )

    # Inicializar o dicionário de resultados
    resultados_similaridade = {}

    # 1. Similaridade de Correlação de Pearson
    pearson_corr, _ = pearsonr(
        cromagrama_original_filtrado.flatten(), cromagrama_gerado_filtrado.flatten()
    )
    resultados_similaridade["Correlação de Pearson"] = pearson_corr

    # 2. Distância Euclidiana
    dist_euclidiana = euclidean(
        cromagrama_original_filtrado.flatten(), cromagrama_gerado_filtrado.flatten()
    )
    resultados_similaridade["Distância Euclidiana"] = dist_euclidiana

    # 3. Similaridade Coseno
    cos_sim = cosine_similarity(
        cromagrama_original_filtrado.flatten().reshape(1, -1),
        cromagrama_gerado_filtrado.flatten().reshape(1, -1),
    )
    resultados_similaridade["Similaridade Coseno"] = cos_sim[0][0]

    # 4. Distância de Manhattan
    dist_manhattan = np.sum(
        np.abs(
            cromagrama_original_filtrado.flatten()
            - cromagrama_gerado_filtrado.flatten()
        )
    )
    resultados_similaridade["Distância de Manhattan"] = dist_manhattan

    return resultados_similaridade


# Exemplo de uso:
# Calcular as métricas de similaridade entre cromagramas (original e gerado)
# resultados = calcular_similaridade(original_chroma, chromas_2, limiar_db=-30)
# print(resultados)
