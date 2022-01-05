import numpy as np
import pandas as pd
import argparse
from change_modality_utils import midi_feature_to_emotion

VGMIDI = True
IN_DIR = "output/extracted/"

chord_types = ["maj", "min", "aug", "dim", "sus4", "dom7", "min7"]

def pitch_class(note_position):
    p = note_position % 12
    switcher = {
        0: 'C',
        1: 'C#',
        2: 'D',
        3: 'D#',
        4: 'E',
        5: 'F',
        6: 'F#',
        7: 'G',
        8: 'G#',
        9: 'A',
        10: 'A#',
        11: 'B'
    }
    return switcher.get(p)

def roman_numeral_label(relative_pitch, quality, key_major):
    p = relative_pitch % 12
    major_switcher = {
        0: 'I',
        1: '#I',
        2: 'II',
        3: '#II',
        4: 'III',
        5: 'IV',
        6: '#IV',
        7: 'V',
        8: '#V',
        9: 'VI',
        10: '#VI',
        11: 'VII',
        12: 'i',
        13: '#i',
        14: 'ii',
        15: '#ii',
        16: 'iii',
        17: 'iv',
        18: '#iv',
        19: 'v',
        20: '#v',
        21: 'vi',
        22: '#vi',
        23: 'vii'
    }
    minor_switcher = {
        0: 'I',
        1: '#I',
        2: 'II',
        3: 'III',
        4: '#III',
        5: 'IV',
        6: '#IV',
        7: 'V',
        8: 'VI',
        9: '#VI',
        10: 'VII',
        11: '#VII',
        12: 'i',
        13: '#i',
        14: 'ii',
        15: 'iii',
        16: '#iii',
        17: 'iv',
        18: '#iv',
        19: 'v',
        20: 'vi',
        21: '#vi',
        22: 'vii',
        23: '#vii'
    }

    switcher = minor_switcher
    if key_major:
        switcher = major_switcher
    
    if quality == "maj":
        return switcher.get(p)
    elif quality == "min":
        return switcher.get(p + 12)
    elif quality == "aug":
        return switcher.get(p) + str("+")
    elif quality == "dim":
        return switcher.get(p + 12) + str("o")
    elif quality == "sus4":
        return switcher.get(p) + str("sus4")
    elif quality == "dom7":
        return switcher.get(p) + str("7")
    else:  # quality == "min7":
        return switcher.get(p + 12) + str("7")

def valence_category(valence):
    if VGMIDI:
        if valence < 0.0:
            return "L"
        elif valence < 0.5290476:
            return "M"
        else:
            return "H"
    else: # TODO
        if valence < 0.0:
            return "L"
        elif valence < 0.5290476:
            return "M"
        else:
            return "H"

def arousal_category(arousal):
    if VGMIDI:
        if arousal < -0.1404167:
            return "L"
        elif arousal < 0.3381111:
            return "M"
        else:
            return "H"
    else: # TODO
        if arousal < -0.1404167:
            return "L"
        elif arousal < 0.3381111:
            return "M"
        else:
            return "H"

def predicted_valence_category(valence):
    if isinstance(valence, (np.ndarray, np.generic)):
        v = np.full_like(valence, "H", dtype=np.str)
        v[valence < 0.3204725] = "M"
        v[valence < 0.1229805] = "L"
        return v
    else:
        if valence < 0.1229805:
            return "L"
        elif valence < 0.3204725:
            return "M"
        else:
            return "H"

def predicted_arousal_category(arousal):
    if isinstance(arousal, (np.ndarray, np.generic)):
        a = np.full_like(arousal, "H", dtype=np.str)
        a[arousal < 0.20592906] = "M"
        a[arousal < 0.04047026] = "L"
        return a
    else:
        if arousal < 0.04047026:
            return "L"
        elif arousal < 0.20592906:
            return "M"
        else:
            return "H"

# midi_extractor.py에서 나온 결과인 *.npy를 읽어서 음악 특징 벡터 테이블(vgmidi_emotion.csv 또는 yamaha_emotion.csv)을 만드는 코드
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, default=0, help='yamaha folder number')
    args = parser.parse_args()

    print(type(np.array(["H", "L"])))

    if args.number > 0:
        VGMIDI = False
        IN_DIR = "output_yamaha/" + str(args.number) + "/extracted/"

    key = np.load(IN_DIR + "key.npy")
    global_key = np.load(IN_DIR + "global_key.npy")
    chord = np.load(IN_DIR + "chord.npy")
    melodic_contour = np.load(IN_DIR + "melodic_contour.npy")
    note_density = np.load(IN_DIR + "note_density.npy")
    note_octave = np.load(IN_DIR + "note_octave.npy")
    note_velocity = np.load(IN_DIR + "note_velocity.npy")
    rhythm_density = np.load(IN_DIR + "rhythm.npy")
    roman_numeral = np.load(IN_DIR + "roman_numeral_chord.npy")
    tempo = np.load(IN_DIR + "tempo.npy")
    mean_note_pitch = np.load(IN_DIR + "mean_note_pitch.npy")
    valence = np.load(IN_DIR + "valence.npy")
    arousal = np.load(IN_DIR + "arousal.npy")

    f = open(IN_DIR + "metadata", "r")
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
    print(mean_note_pitch.shape)
    print(valence.shape)            # (seq_len, )
    print(arousal.shape)            # (seq_len, )

    data = []
    header = ["ID", "song", "measure", "ending", "empty", "key.local.major",
        "key.global.major", "tonic.local", "tonic.global",
        "chord.maj", "chord.min", "chord.aug",
        "chord.dim", "chord.sus4", "chord.dom7", "chord.min7",
        "roman.numeral", "roman.numeral.label",
        "prev.roman.numeral", "prev.roman.numeral.label",
        "note.density", "note.pitch.mean", "note.velocity", "rhythm.density",
        "tempo", "valence", "arousal", "valence.category", "arousal.category",
        "prev.valence", "prev.arousal", "prev.valence.category", "prev.arousal.category",
        "predicted.valence", "predicted.arousal",
        "predicted.valence.category", "predicted.arousal.category"]
    #data.append(header)
    
    
    seq_len = key.shape[0]
    for i in range(seq_len):
        entity = [i, metadata[i][1], int(metadata[i][0]), 0]

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

        # tonic.local: 1, 13 -> 'C' / 2, 14 -> 'D' / ... / 12, 24 -> 'B'
        tonic_local = pitch_class(key[i] - 1)
        entity.append(tonic_local)

        # tonic.global: 1, 13 -> 'C' / 2, 14 -> 'D' / ... / 12, 24 -> 'B'
        tonic_global = pitch_class(global_key[i] - 1)
        entity.append(tonic_global)

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

        # roman.numeral
        entity.append(roman_numeral[i])

        # roman.numeral.label
        if roman_numeral[i] == 0:
            entity.append("no_chord")
        else:
            entity.append(roman_numeral_label(roman_numeral[i] - 1, chord_types[(roman_numeral[i] - 1) // 12], global_key[i] < 13))

        # add dummy data with ending=1, roman.numeral=0, prev.roman.numeral=(last roman numeral of this song)
        if i != 0 and metadata[i][1] != metadata[i - 1][1]:
            data.append([-i, metadata[i - 1][1], int(metadata[i - 1][0]) + 1, 1, 1,
                1, 1, 'C', 'C', 0, 0, 0, 0, 0, 0, 0,
                0, 'no_chord',
                roman_numeral[i - 1], roman_numeral_label(roman_numeral[i - 1] - 1, chord_types[(roman_numeral[i - 1] - 1) // 12], global_key[i - 1] < 13),
                0, 0, 0, 0, np.mean(tempo[i - 1]),
                valence[i - 1], arousal[i - 1], valence_category(valence[i - 1]), arousal_category(arousal[i - 1]),
                valence[i - 1], arousal[i - 1], valence_category(valence[i - 1]), arousal_category(arousal[i - 1])])

        # prev.roman.numeral
        prev_roman_numeral = 0
        if i == 0 or int(metadata[i][0]) == 0:
            entity.append(0)
        else:
            entity.append(roman_numeral[i - 1])
            prev_roman_numeral = roman_numeral[i - 1]

        # prev.roman.numeral.label
        if i == 0 or int(metadata[i][0]) == 0 or roman_numeral[i - 1] == 0:
            entity.append("no_chord")
        else:
            entity.append(roman_numeral_label(roman_numeral[i - 1] - 1, chord_types[(roman_numeral[i - 1] - 1) // 12], global_key[i - 1] < 13))

        # note.density
        entity.append(np.mean(note_density[i]))

        # note.pitch.mean
        entity.append(mean_note_pitch[i] - 1)

        # note.velocity
        entity.append(np.mean(note_velocity[i]))

        # rhythm.density
        entity.append(16 - np.count_nonzero(rhythm_density[i] - 1))

        # tempo
        entity.append(np.mean(tempo[i]))

        # valence, arousal
        entity.append(valence[i])
        entity.append(arousal[i])

        # valence.category
        entity.append(valence_category(valence[i]))

        # arousal.category
        entity.append(arousal_category(arousal[i]))
        
        # prev.valence, prev.arousal, prev.valence.category, prev.arousal.category
        if i == 0 or int(metadata[i][0]) == 0:
            entity.append(valence[i])
            entity.append(arousal[i])
            entity.append(valence_category(valence[i]))
            entity.append(arousal_category(arousal[i]))
        else:
            entity.append(valence[i - 1])
            entity.append(arousal[i - 1])
            entity.append(valence_category(valence[i - 1]))
            entity.append(arousal_category(arousal[i - 1]))

        """
        # predicted.valence, predicted.arousal, predicted.valence.category, predicted.arousal.category
        predicted = midi_feature_to_emotion(key_local_major, key_global_major,
            chord_maj, chord_min, chord_aug, chord_dim, chord_sus4, chord_dom7,
            chord_min7, np.mean(note_density[i]), mean_note_pitch[i] - 1,
            np.mean(note_velocity[i]), 16 - np.count_nonzero(rhythm_density[i] - 1),
            np.mean(tempo[i]), roman_numeral[i], prev_roman_numeral, True)

        entity.append(predicted["valence"][0])
        entity.append(predicted["arousal"][0])
        entity.append(predicted_valence_category(predicted["valence"][0]))
        entity.append(predicted_arousal_category(predicted["arousal"][0]))
        """

        data.append(entity)

    i = seq_len
    data.append([-i, metadata[i - 1][1], int(metadata[i - 1][0]) + 1, 1, 1,
        1, 1, 'C', 'C', 0, 0, 0, 0, 0, 0, 0,
        0, 'no_chord',
        roman_numeral[i - 1], roman_numeral_label(roman_numeral[i - 1] - 1, chord_types[(roman_numeral[i - 1] - 1) // 12], global_key[i - 1] < 13),
        0, 0, 0, 0, np.mean(tempo[i - 1]),
        valence[i - 1], arousal[i - 1], valence_category(valence[i - 1]), arousal_category(arousal[i - 1]),
        valence[i - 1], arousal[i - 1], valence_category(valence[i - 1]), arousal_category(arousal[i - 1])])

    #print(data[0])
    #print(data[1947])

    df = pd.DataFrame(data)

    """
    [0"ID", 1"song", 2"measure", 3"ending", 4"empty", 5"key.local.major",
        6"key.global.major", 7"tonic.local", 8"tonic.global",
        9"chord.maj", 10"chord.min", 11"chord.aug",
        12"chord.dim", 13"chord.sus4", 14"chord.dom7", 15"chord.min7",
        16"roman.numeral", 17"roman.numeral.label",
        18"prev.roman.numeral", 19"prev.roman.numeral.label",
        20"note.density", 21"note.pitch.mean", 22"note.velocity", 23"rhythm.density",
        24"tempo", 25"valence", 26"arousal", 27"valence.category", 28"arousal.category",
        29"prev.valence", 30"prev.arousal", 31"prev.valence.category", 32"prev.arousal.category",
        33"predicted.valence", 34"predicted.arousal",
        35"predicted.valence.category", 36"predicted.arousal.category"]
    """

    predicted = midi_feature_to_emotion(df.iloc[:, 5].values, df.iloc[:, 6].values,
            df.iloc[:, 9].values, df.iloc[:, 10].values, df.iloc[:, 11].values,
            df.iloc[:, 12].values, df.iloc[:, 13].values, df.iloc[:, 14].values,
            df.iloc[:, 15].values, df.iloc[:, 20].values, df.iloc[:, 21].values,
            df.iloc[:, 22].values, df.iloc[:, 23].values,
            df.iloc[:, 24].values, df.iloc[:, 16].values, df.iloc[:, 18].values, True)
    df[33] = predicted["valence"]
    df[34] = predicted["arousal"]
    df[35] = predicted_valence_category(predicted["valence"])
    df[36] = predicted_arousal_category(predicted["arousal"])

    if VGMIDI:
        df.to_csv("./" + IN_DIR + "vgmidi_emotion.csv", index=False, header=header)
    else:
        df.to_csv("./" + IN_DIR + "yamaha_emotion.csv", index=False, header=header)
        

