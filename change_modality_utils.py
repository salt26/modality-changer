from pickle import NONE
import numpy as np
import torch
from torch._C import device
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from scipy import stats

def clamp(value, min_value, max_value, integer=False):
    if isinstance(value, (np.ndarray, np.generic)):
        return np.clip(value, min_value, max_value)
    else:
        if integer:
            return max(min_value, min(max_value, round(value)))
        else:
            return max(min_value, min(max_value, value))


def tempo_bin(tempo):
    if isinstance(tempo, (np.ndarray, np.generic)):
        t = np.zeros_like(tempo, dtype=np.int32)
        t[tempo <= 1200000] = 1
        t[tempo <= 1000000] = 2
        t[tempo <= 800000] = 3
        t[tempo <= 600000] = 4
        t[tempo < 444445] = 5
        t[tempo < 333334] = 6
        return t
    else:
        if tempo > 1200000:    # bpm < 50
            return 0
        elif tempo > 1000000:  # bpm < 60
            return 1
        elif tempo > 800000:   # bpm < 75
            return 2
        elif tempo > 600000:   # bpm < 100
            return 3
        elif tempo >= 444445:  # bpm < 135
            return 4
        elif tempo >= 333334:  # bpm < 180
            return 5
        else:                  # bpm > 180
            return 6


# non-vectorized, aggregated
def extracted_features_to_aggregated_midi_feature(fname, features, sequences_length, starting_measure=0, ending_measure=-1):
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
    temp_pitch_mean_vector = features["Mean_note_pitch"][starting_measure : ending_measure][not_empty]
    f["note.pitch.mean"] = np.mean(temp_pitch_mean_vector[np.nonzero(temp_pitch_mean_vector)])
    f["note.velocity"] = np.sum(features["Note_velocity"][starting_measure : ending_measure, :][not_empty, ...]) / not_empty_len / 16
    f["rhythm.density"] = 16 - (np.count_nonzero(features["Rhythm_density"][starting_measure : ending_measure, :][not_empty, ...] - 1) / not_empty_len)
    f["tempo"] = np.sum(features["Tempo"][starting_measure : ending_measure, :][not_empty, ...]) / not_empty_len / 16

    # There are no `f["roman.numeral"]` and `f["prev.roman.numeral"]`.
    # If you want these features, use `extracted_features_to_midi_feature()`.

    return f


# vectorized for each measure, non-aggregated
def extracted_features_to_midi_feature(fname, features, sequences_length):
        
    f = {}

    f["song"] = fname
    f["measure"] = np.array(range(sequences_length), dtype=np.int64)

    f["empty"] = np.array(features["Key"] == 0, dtype=np.int64)

    f["key.local.major"] = np.array(features["Key"] < 13, dtype=np.int64)

    f["key.global.major"] = np.array(features["Global_key"] < 13, dtype=np.int64)

    # chord: 1 ~ 12 -> maj, 13 ~ 24 -> min, 25 ~ 36 -> aug, 37 ~ 48 -> dim, 49 ~ 60: sus4, 61 ~ 72: dom7, 73 ~ 84: min7
    chord_oh = np.eye(85)[features["Chord"]]
    f["chord.maj"] = np.sum(np.sum(chord_oh[:, :, 1:13], axis=2), axis=1)
    f["chord.min"] = np.sum(np.sum(chord_oh[:, :, 13:25], axis=2), axis=1)
    f["chord.aug"] = np.sum(np.sum(chord_oh[:, :, 25:37], axis=2), axis=1)
    f["chord.dim"] = np.sum(np.sum(chord_oh[:, :, 37:49], axis=2), axis=1)
    f["chord.sus4"] = np.sum(np.sum(chord_oh[:, :, 49:61], axis=2), axis=1)
    f["chord.dom7"] = np.sum(np.sum(chord_oh[:, :, 61:73], axis=2), axis=1)
    f["chord.min7"] = np.sum(np.sum(chord_oh[:, :, 73:85], axis=2), axis=1)

    f["note.density"] = np.sum(features["Note_density"], axis=1) / 16
    f["note.pitch.mean"] = features["Mean_note_pitch"]
    f["note.velocity"] = np.sum(features["Note_velocity"], axis=1) / 16
    f["rhythm.density"] = 16 - np.count_nonzero(features["Rhythm_density"] - 1, axis=1)
    f["tempo"] = np.sum(features["Tempo"], axis=1) / 16

    f["roman.numeral"] = features["Roman_numeral_chord"]
    f["prev.roman.numeral"] = np.concatenate(([0], features["Roman_numeral_chord"][:-1]))

    return f


def emotion_to_midi_feature(valence, arousal):
    if (valence < -1): valence = -1
    elif (valence > 1): valence = 1
    if (arousal < -1): arousal = -1
    elif (arousal > 1): arousal = 1

    f = {}

    f["key.local.major"] = clamp(0.482460 + 0.144274 * valence - 0.040919 * arousal, 0, 1, True)
    f["key.global.major"] = clamp(0.403267 + 0.382065 * valence - 0.052849 * arousal, 0, 1, True)

    f["chord.maj"] = clamp(4.14292 + 1.47985 * valence - 0.27026 * arousal, 0, 16)
    f["chord.min"] = clamp(3.43880 - 1.30647 * valence + 0.61270 * arousal, 0, 16)
    f["chord.aug"] = clamp(0.49831 - 0.15036 * valence + 0.12709 * arousal, 0, 16)
    f["chord.dim"] = clamp(0.85197 - 0.67637 * valence + 0.39020 * arousal, 0, 16)
    f["chord.sus4"] = clamp(3.07394 + 0.26588 * valence - 0.69071 * arousal, 0, 16)
    f["chord.dom7"] = clamp(1.39375 + 0.25181 * arousal, 0, 16)
    f["chord.min7"] = clamp(2.11296 + 0.48170 * valence - 0.32964 * arousal, 0, 16)

    f["note.density"] = clamp(3.34349 - 0.25946 * valence, 0, 15)
    f["note.pitch.mean"] = clamp(60.57052 - 0.44633 * arousal, 0, 127)
    f["note.velocity"] = clamp(95.4541 - 5.0989 * valence + 6.7917 * arousal, 0, 127)
    f["rhythm.density"] = clamp(7.42540 + 0.25139 * valence + 1.77139 * arousal, 0, 16)
    f["tempo"] = max(0, 567652 - 65097 * valence - 188871 * arousal)

    return f

def emotion_dict_to_midi_feature(emotion_dict):
    return emotion_to_midi_feature(emotion_dict["valence"], emotion_dict["arousal"])


def midi_feature_to_emotion(key_local_major, key_global_major, chord_maj, chord_min, chord_aug,
                             chord_dim, chord_sus4, chord_dom7, chord_min7, note_density, 
                             note_pitch_mean, note_velocity, rhythm_density, tempo,
                             roman_numeral=0, prev_roman_numeral=0, use_nn_model=False, verbose=False):
    
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
    note_pitch_mean = clamp(note_pitch_mean, 0, 127)
    note_velocity = clamp(note_velocity, 0, 127)
    rhythm_density = clamp(rhythm_density, 0, 16)
    tempo = clamp(tempo, 0, float("inf"))
    roman_numeral = clamp(roman_numeral, 0, 84, True)
    prev_roman_numeral = clamp(prev_roman_numeral, 0, 84, True)

    e = {}

    if use_nn_model:
        if verbose:
            print("Use neural regression model")

        torch.manual_seed(100)

        if torch.cuda.is_available() and verbose:
            print("Use GPU")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = torch.nn.Sequential(
            torch.nn.Linear(190, 4096),
            torch.nn.BatchNorm1d(4096),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4096, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 2),
        ).to(device)
        net.load_state_dict(torch.load('./regression_model.pth'))

        entity = np.vstack((key_local_major, key_global_major,
            chord_maj / 16.0, chord_min / 16.0,
            chord_aug / 16.0, chord_dim / 16.0,
            chord_sus4 / 16.0, chord_dom7 / 16.0,
            chord_min7 / 16.0, note_density / 16.0,
            note_pitch_mean / 127.0, note_velocity / 127.0,
            rhythm_density / 16.0))
        entity = np.vstack((entity, np.transpose(np.eye(7)[tempo_bin(tempo)].reshape((-1, 7)))))
        entity = np.vstack((entity, np.transpose(np.eye(85)[roman_numeral].reshape((-1, 85)))))
        entity = np.vstack((entity, np.transpose(np.eye(85)[prev_roman_numeral].reshape((-1, 85)))))

        x = Variable(torch.from_numpy(np.array(np.transpose(entity), dtype=np.float32)))
        test_loader = Data.DataLoader(dataset=x)

        e["valence"] = []
        e["arousal"] = []

        with torch.no_grad():
            net.eval()
            for data in test_loader:
                test_x = data.to(device)
                prediction = net(test_x)
                e["valence"].append(prediction[0, 0].item() * 2 - 1)
                e["arousal"].append(prediction[0, 1].item() * 2 - 1)
        
        e["valence"] = np.array(e["valence"])
        e["arousal"] = np.array(e["arousal"])

    else:
        if verbose:
            print("Use linear regression model")
        e["valence"] = clamp(0.1653 + key_local_major * 0.02307 + key_global_major * 0.2474
            + chord_maj * 0.02263 + chord_min * 0.01604 + chord_aug * 0.01141
            + chord_sus4 * 0.01969 + chord_dom7 * 0.01894 + chord_min7 * 0.02430
            - note_density * 0.01632 + note_pitch_mean * 0.001199
            - note_velocity * 0.002259 + rhythm_density * 0.001919 - tempo * 0.0000003687, -1, 1)

        e["arousal"] = clamp(0.3681 + key_local_major * 0.01262 - key_global_major * 0.02040
            - chord_maj * 0.01758 - chord_min * 0.01048 - chord_aug * 0.01072
            - chord_dim * 0.006241 - chord_sus4 * 0.01869 - chord_dom7 * 0.01427 - chord_min7 * 0.01732
            + note_density * 0.005358 + note_velocity * 0.001878 + rhythm_density * 0.05882
            - (rhythm_density ** 2) * 0.001457 - tempo * 0.000001073, -1, 1)

    return e

def midi_feature_dict_to_emotion(midi_feature_dict, use_nn_model):
    return midi_feature_to_emotion(midi_feature_dict["key.local.major"], midi_feature_dict["key.global.major"],
        midi_feature_dict["chord.maj"], midi_feature_dict["chord.min"], midi_feature_dict["chord.aug"],
        midi_feature_dict["chord.dim"], midi_feature_dict["chord.sus4"],
        midi_feature_dict["chord.dom7"], midi_feature_dict["chord.min7"],
        midi_feature_dict["note.density"], midi_feature_dict["note.pitch.mean"], 
        midi_feature_dict["note.velocity"], midi_feature_dict["rhythm.density"], midi_feature_dict["tempo"],
        midi_feature_dict["roman.numeral"], midi_feature_dict["prev.roman.numeral"], use_nn_model)