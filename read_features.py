import numpy as np
import pandas as pd

# midi_extractor.py에서 나온 결과인 *.npy를 읽어서 음악 특징 벡터 테이블(vgmidi_emotion.csv)을 만드는 코드
if __name__ == '__main__':

    key = np.load("output/extracted/key.npy")
    global_key = np.load("output/extracted/global_key.npy")
    chord = np.load("output/extracted/chord.npy")
    melodic_contour = np.load("output/extracted/melodic_contour.npy")
    note_density = np.load("output/extracted/note_density.npy")
    note_octave = np.load("output/extracted/note_octave.npy")
    note_velocity = np.load("output/extracted/note_velocity.npy")
    rhythm_density = np.load("output/extracted/rhythm.npy")
    roman_numeral = np.load("output/extracted/roman_numeral_chord.npy")
    tempo = np.load("output/extracted/tempo.npy")
    valence = np.load("output/extracted/valence.npy")
    arousal = np.load("output/extracted/arousal.npy")

    f = open("output/extracted/metadata", "r")
    metadata = []
    while True:
        line = f.readline()
        if len(line) > 0:
            full_filename = line.split(" ")[1]
            measure = full_filename.split("_")[0]
            filename = full_filename[full_filename.index("_") + 1:-1]
            metadata.append([measure, filename])
        if not line: break
    f.close()

    print(key.shape)                # (seq_len, )
    print(global_key.shape)         # (seq_len, )
    print(chord.shape)
    print(melodic_contour.shape)
    print(note_density.shape)
    print(note_octave.shape)
    print(note_velocity.shape)
    print(rhythm_density.shape)
    print(roman_numeral.shape)
    print(tempo.shape)
    print(valence.shape)            # (seq_len, )
    print(arousal.shape)            # (seq_len, )

    data = []
    header = ["ID", "song", "measure", "empty", "key.local.major",
        "key.global.major", "chord.maj", "chord.min", "chord.aug",
        "chord.dim", "chord.sus4", "chord.dom7", "chord.min7",
        "note.density", "note.octave", "note.velocity", "rhythm.density",
        "tempo", "valence", "arousal"]
    #data.append(header)
    
    
    seq_len = key.shape[0]
    for i in range(seq_len):
        entity = [i, metadata[i][1], int(metadata[i][0])]

        # empty: key가 0 -> 1, 1 이상 -> 0
        if key[i] == 0:
            entity.append(1)
        else:
            entity.append(0)

        # key.local.major: 1 ~ 12 -> 1 (major), 13 ~ 24 -> 0 (minor)
        key_local_major = 0
        if key[i] < 13:
            key_local_major = 1
        entity.append(key_local_major)

        # key.global.major: 1 ~ 12 -> 1 (major), 13 ~ 24 -> 0 (minor)
        key_global_major = 0
        if global_key[i] < 13:
            key_global_major = 1
        entity.append(key_global_major)

        # chord: 1 ~ 12 -> maj, 13 ~ 24 -> min, 25 ~ 36 -> aug, 37 ~ 48 -> dim, 49 ~ 60: sus4, 61 ~ 72: dom7, 73 ~ 84: min7
        chord_maj = 0
        chord_min = 0
        chord_aug = 0
        chord_dim = 0
        chord_sus4 = 0
        chord_dom7 = 0
        chord_min7 = 0
        for j in range(16):
            if 1 <= chord[i][j] <= 12: chord_maj += 1
            elif 13 <= chord[i][j] <= 24: chord_min += 1
            elif 25 <= chord[i][j] <= 36: chord_aug += 1
            elif 37 <= chord[i][j] <= 48: chord_dim += 1
            elif 49 <= chord[i][j] <= 60: chord_sus4 += 1
            elif 61 <= chord[i][j] <= 72: chord_dom7 += 1
            elif 73 <= chord[i][j] <= 84: chord_min7 += 1
        entity.append(chord_maj)
        entity.append(chord_min)
        entity.append(chord_aug)
        entity.append(chord_dim)
        entity.append(chord_sus4)
        entity.append(chord_dom7)
        entity.append(chord_min7)

        # note.density
        entity.append(np.mean(note_density[i]))

        # note.octave
        entity.append(np.mean(note_octave[i][np.nonzero(note_octave[i])]) - 1)

        # note.velocity
        entity.append(np.mean(note_velocity[i]))

        # rhythm.density
        entity.append(16 - np.count_nonzero(rhythm_density[i] - 1))

        # tempo
        entity.append(np.mean(tempo[i]))

        # valence, arousal
        entity.append(valence[i])
        entity.append(arousal[i])

        data.append(entity)

    #print(data[0])
    #print(data[1947])

    df = pd.DataFrame(data)
    df.to_csv("./output/extracted/vgmidi_emotion.csv", index=False, header=header)
        

