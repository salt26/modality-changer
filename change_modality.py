from midi_extractor import parse_events, make_piano_roll_sequences, sequences_to_midi_file, extract_features
import os
import argparse
import numpy as np
from scipy import stats

def clamp(value, min_value, max_value, integer=False):
    if integer:
        return max(min_value, min(max_value, round(value)))
    else:
        return max(min_value, min(max_value, value))

# ending_measure: inclusive
def midi_to_midi_feature(midi_file_path, starting_measure=0, ending_measure=-1, verbose=False):

    def is_midi_file(filename):
        ext = filename[filename.rindex('.'): len(filename)]
        return ext == '.mid' or ext == '.MID' or ext == '.midi' or ext == '.MIDI'
    
    assert(is_midi_file(midi_file_path))

    fname = os.path.basename(midi_file_path).rstrip('.mid').rstrip('.MID').rstrip('.midi').rstrip('.MIDI')

    try:
        events, notes, ticks_per_beat = \
            parse_events(midi_file_path.replace('\\', '/'))

        sequences, onset_sequences, sequences_length = \
            make_piano_roll_sequences(events, notes, ticks_per_beat, verbose)

        sequence_files, valid_sequences = sequences_to_midi_file(events, sequences_length, ticks_per_beat,
                                                                 midi_file_path.replace('\\', '/'))

        # extract_features의 인자로 valid_sequences에 numpy boolean array(shape=(sequence_length,))를 넘기면
        # 음표가 들어있지 않은 시퀀스를 자동으로 제외하고 low-level feature vector 배열을 생성하여
        # 그 길이가 sequence_files의 길이와 같게 된다.
        # 반면, None을 넘기면 음표가 들어있지 않은 시퀀스도 포함하여 그 길이가 sequence_length가 된다.
        features = extract_features(sequences, onset_sequences, sequences_length, events, verbose, False,
                                    valid_sequences=valid_sequences)

    except IOError as e:
        print(e)
        return {}

    if (starting_measure < 0): starting_measure = sequences_length + starting_measure
    if (starting_measure < 0): starting_measure = 0
    elif (starting_measure >= sequences_length): starting_measure = sequences_length - 1

    if (ending_measure < 0): ending_measure = sequences_length + ending_measure + 1
    if (ending_measure <= starting_measure): ending_measure = starting_measure + 1
    elif (ending_measure > sequences_length): ending_measure = sequences_length
        
    f = {}

    f["song"] = fname
    f["measure.start"] = starting_measure
    f["measure.end"] = ending_measure - 1
    if (starting_measure == ending_measure - 1): f["measure"] = starting_measure

    not_empty = np.nonzero(features["Key"][starting_measure : ending_measure])[0]
    not_empty_len = len(not_empty)
    f["empty"] = 0
    if not_empty_len == 0:
        f["empty"] = 1
        return f

    key_local = stats.mode(features["Key"][starting_measure : ending_measure][not_empty])[0][0]
    f["key.local.major"] = 0
    if key_local < 13: f["key.local.major"] = 1

    f["key.global.major"] = 0
    if features["Global_key"][0] < 13: f["key.global.major"] = 1

    # chord: 1 ~ 12 -> maj, 13 ~ 24 -> min, 25 ~ 36 -> aug, 37 ~ 48 -> dim, 49 ~ 60: sus4, 61 ~ 72: dom7, 73 ~ 84: min7
    chord = features["Chord"][starting_measure : ending_measure, :][not_empty, ...]
    chord_oh = np.eye(85)[chord]
    f["chord.maj"] = np.sum(chord_oh[:, :, 1:13]) / not_empty_len
    f["chord.min"] = np.sum(chord_oh[:, :, 13:25]) / not_empty_len
    f["chord.aug"] = np.sum(chord_oh[:, :, 25:37]) / not_empty_len
    f["chord.dim"] = np.sum(chord_oh[:, :, 37:49]) / not_empty_len
    f["chord.sus4"] = np.sum(chord_oh[:, :, 49:61]) / not_empty_len
    f["chord.dom7"] = np.sum(chord_oh[:, :, 61:73]) / not_empty_len
    f["chord.min7"] = np.sum(chord_oh[:, :, 73:85]) / not_empty_len

    f["note.density"] = np.sum(features["Note_density"][starting_measure : ending_measure, :][not_empty, ...]) / not_empty_len / 16
    temp_octave_vector = features["Note_octave"][starting_measure : ending_measure, :][not_empty, ...]
    f["note.octave"] = np.mean(temp_octave_vector[np.nonzero(temp_octave_vector)]) - 1
    f["note.velocity"] = np.sum(features["Note_velocity"][starting_measure : ending_measure, :][not_empty, ...]) / not_empty_len / 16
    f["rhythm.density"] = 16 - (np.count_nonzero(features["Rhythm_density"][starting_measure : ending_measure, :][not_empty, ...] - 1) / not_empty_len)
    f["tempo"] = np.sum(features["Tempo"][starting_measure : ending_measure, :][not_empty, ...]) / not_empty_len / 16

    return f

    

def emotion_to_midi_feature(valence, arousal):
    if (valence < -1): valence = -1
    elif (valence > 1): valence = 1
    if (arousal < -1): arousal = -1
    elif (arousal > 1): arousal = 1

    f = {}

    f["key.local.major"] = clamp(0.482460 + 0.144274 * valence - 0.040919 * arousal, 0, 1, True)
    f["key.global.major"] = clamp(0.403267 + 0.382065 * valence - 0.052849 * arousal, 0, 1, True)

    f["chord.maj"] = clamp(4.14292 + 1.47985 * valence - 0.27026 * arousal, 0, 16) #max(0, min(16, round(4.14292 + 1.47985 * valence - 0.27026 * arousal)))
    f["chord.min"] = clamp(3.43880 - 1.30647 * valence + 0.61270 * arousal, 0, 16)
    f["chord.aug"] = clamp(0.49831 - 0.15036 * valence + 0.12709 * arousal, 0, 16)
    f["chord.dim"] = clamp(0.85197 - 0.67637 * valence + 0.39020 * arousal, 0, 16)
    f["chord.sus4"] = clamp(3.07394 + 0.26588 * valence - 0.69071 * arousal, 0, 16)
    f["chord.dom7"] = clamp(1.39375 + 0.25181 * arousal, 0, 16)
    f["chord.min7"] = clamp(2.11296 + 0.48170 * valence - 0.32964 * arousal, 0, 16)

    f["note.density"] = clamp(3.34349 - 0.25946 * valence, 0, 15)
    f["note.octave"] = 5.508  # sd = 0.7763639
    f["note.velocity"] = clamp(95.4541 - 5.0989 * valence + 6.7917 * arousal, 0, 127)
    f["rhythm.density"] = clamp(7.42540 + 0.25139 * valence + 1.77139 * arousal, 0, 16)
    f["tempo"] = max(0, 567652 - 65097 * valence - 188871 * arousal)

    return f

def emotion_dict_to_midi_feature(emotion_dict):
    return emotion_to_midi_feature(emotion_dict["valence"], emotion_dict["arousal"])


def midi_feature_to_emotion(key_local_major, key_global_major, chord_maj, chord_min, chord_aug,
                             chord_dim, chord_sus4, chord_dom7, chord_min7, note_density, 
                             note_octave, note_velocity, rhythm_density, tempo):
    key_local_major = clamp(key_local_major, 0, 1, True)
    key_global_major = clamp(key_global_major, 0, 1, True)
    chord_maj = clamp(chord_maj, 0, 16)
    chord_min = clamp(chord_min, 0, 16)
    chord_aug = clamp(chord_aug, 0, 16)
    chord_dim = clamp(chord_dim, 0, 16)
    chord_sus4 = clamp(chord_sus4, 0, 16)
    chord_dom7 = clamp(chord_dom7, 0, 16)
    chord_min7 = clamp(chord_min7, 0, 16)
    note_density = clamp(note_density, 0, 15)
    note_octave = clamp(note_octave, 0, 10)
    note_velocity = clamp(note_velocity, 0, 127)
    rhythm_density = clamp(rhythm_density, 0, 16)
    tempo = max(0, tempo)

    e = {}

    e["valence"] = clamp(0.2400 + 0.02279 * key_local_major + 0.2481 * key_global_major
        + 0.02295 * chord_maj + 0.01646 * chord_min + 0.01172 * chord_aug + 0.02009 * chord_sus4
        + 0.01946 * chord_dom7 + 0.02490 * chord_min7 - 0.01556 * note_density
        - 0.002267 * note_velocity - 0.0000003606 * tempo, -1, 1)

    e["arousal"] = clamp(0.2899 - 0.02058 * key_global_major - 0.01803 * chord_maj
        - 0.01171 * chord_min - 0.01145 * chord_aug - 0.007169 * chord_dim
        - 0.01942 * chord_sus4 - 0.01507 * chord_dom7 - 0.01826 * chord_min7
        + 0.02191 * note_octave + 0.001899 * note_velocity + 0.05750 * rhythm_density
        - 0.001401 * (rhythm_density ** 2) - 0.000001077 * tempo, -1, 1)

    return e

def midi_feature_dict_to_emotion(midi_feature_dict):
    return midi_feature_to_emotion(midi_feature_dict["key.local.major"], midi_feature_dict["key.global.major"],
        midi_feature_dict["chord.maj"], midi_feature_dict["chord.min"], midi_feature_dict["chord.aug"],
        midi_feature_dict["chord.dim"], midi_feature_dict["chord.sus4"],
        midi_feature_dict["chord.dom7"], midi_feature_dict["chord.min7"],
        midi_feature_dict["note.density"], midi_feature_dict["note.octave"], 
        midi_feature_dict["note.velocity"], midi_feature_dict["rhythm.density"], midi_feature_dict["tempo"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--starting_measure', default=0, type=int, help='starting measure number of MIDIs')
    parser.add_argument('-e', '--ending_measure', default=-1, type=int, help='ending measure number of MIDIs')
    parser.add_argument('midi_files', nargs='*', help='input MIDI files')
    args = parser.parse_args()

    for i, d in enumerate(args.midi_files):
        f = midi_to_midi_feature(args.midi_files[i], int(args.starting_measure), int(args.ending_measure))
        print(f)
        e = midi_feature_dict_to_emotion(f)
        print(e)
        f2 = emotion_dict_to_midi_feature(e)
        print(f2)

    while True:
        file_path = input("Input MIDI(.mid) file path (input \"exit\" to terminate): ")
        file_path.strip("\"")
        if file_path == "exit": break
        f = midi_to_midi_feature(file_path)
        print(f)
        e = midi_feature_dict_to_emotion(f)
        print(e)
        f2 = emotion_dict_to_midi_feature(e)
        print(f2)
        print()
