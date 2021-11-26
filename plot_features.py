import json
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from sklearn.metrics import confusion_matrix
import midi_extractor as extractor

if __name__ == '__main__':

    key = np.load("output/extracted/key.npy")
    chord = np.load("output/extracted/chord.npy")
    melodic_contour = np.load("output/extracted/melodic_contour.npy")
    note_density = np.load("output/extracted/note_density.npy")
    note_octave = np.load("output/extracted/note_octave.npy")
    note_velocity = np.load("output/extracted/note_velocity.npy")
    rhythm_density = np.load("output/extracted/rhythm.npy")
    roman_numeral = np.load("output/extracted/roman_numeral_chord.npy")
    tempo = np.load("output/extracted/tempo.npy")
    valence = np.load("output/extracted/valence.npy")

    print(key.shape)                # (seq_len, )
    print(chord.shape)
    print(melodic_contour.shape)
    print(note_density.shape)
    print(note_octave.shape)
    print(note_velocity.shape)
    print(rhythm_density.shape)
    print(roman_numeral.shape)
    print(tempo.shape)
    print(valence.shape)            # (seq_len, )

    seq_len = key.shape[0]

    # aggregate
    a_key = np.zeros(seq_len)
    a_chord = np.zeros(seq_len)
    a_melodic_contour = np.zeros(seq_len)
    a_note_density = np.zeros(seq_len)
    a_note_octave = np.zeros(seq_len)
    a_note_velocity = np.zeros(seq_len)
    a_rhythm_density = np.zeros(seq_len)
    a_roman_numeral = np.zeros(seq_len)
    a_chord_quality = np.zeros(seq_len)
    a_tempo = np.zeros(seq_len)
    a_valence = np.zeros(seq_len)

    for i in range(seq_len):
        if key[i] <= 0:
            a_key[i] = 0
        elif key[i] <= 12:
            a_key[i] = 1 # 장조
        else:
            a_key[i] = 2 # 단조

        a_chord[i] = np.bincount(chord[i][:]).argmax()

        r = 0
        r_sum = 0
        l = 0
        l_sum = 0
        for j in range(melodic_contour.shape[1]):
            if j < melodic_contour.shape[1] // 2: # and melodic_contour[i][j] > 0:
                l += 1
                l_sum += melodic_contour[i][j]
            elif j >= melodic_contour.shape[1] // 2: # and melodic_contour[i][j] > 0:
                r += 1
                r_sum += melodic_contour[i][j]
        a_melodic_contour[i] = r_sum / r - l_sum / l

        a_note_density[i] = np.sum(note_density[i][:]) / note_density.shape[1]

        a_note_octave[i] = np.sum(note_octave[i][:]) / note_octave.shape[1]

        a_note_velocity[i] = np.sum(note_velocity[i][:]) / note_velocity.shape[1]

        a_rhythm_density[i] = Counter(list(rhythm_density[i][:]))[1] / rhythm_density.shape[1]

        a_roman_numeral[i] = np.bincount(roman_numeral[i][:]).argmax()

        if a_chord[i] == 0:
            a_chord_quality[i] = 0
        else:
            a_chord_quality[i] = ((a_chord[i] - 1) // 12) + 1

        a_tempo[i] = np.bincount(tempo[i][:]).argmax()

        if valence[i] < -0.1:
            a_valence[i] = -1
        elif valence[i] > 0.1:
            a_valence[i] = 1
        else:
            a_valence[i] = 0

    alpha = 0.005 # min is 0.002

    plt.figure()
    plt.subplot(3, 3, 1)
    plt.scatter(valence, a_key, s=1, alpha=alpha)
    plt.title("key")
    plt.vlines(-0.1, np.min(a_key), np.max(a_key), colors='b')
    plt.vlines(0.1, np.min(a_key), np.max(a_key), colors='r')
    plt.subplot(3, 3, 2)
    plt.scatter(valence, a_chord, s=1, alpha=alpha)
    plt.title("chord")
    plt.vlines(-0.1, np.min(a_chord), np.max(a_chord), colors='b')
    plt.vlines(0.1, np.min(a_chord), np.max(a_chord), colors='r')
    plt.subplot(3, 3, 3)
    plt.scatter(valence, a_melodic_contour, s=1, alpha=alpha)
    plt.title("melodic_contour")
    plt.vlines(-0.1, np.min(a_melodic_contour), np.max(a_melodic_contour), colors='b')
    plt.vlines(0.1, np.min(a_melodic_contour), np.max(a_melodic_contour), colors='r')
    plt.subplot(3, 3, 4)
    plt.scatter(valence, a_note_density, s=1, alpha=alpha)
    plt.title("note_density")
    plt.vlines(-0.1, np.min(a_note_density), np.max(a_note_density), colors='b')
    plt.vlines(0.1, np.min(a_note_density), np.max(a_note_density), colors='r')
    plt.subplot(3, 3, 5)
    plt.scatter(valence, a_note_octave, s=1, alpha=alpha)
    plt.title("note_octave")
    plt.vlines(-0.1, np.min(a_note_octave), np.max(a_note_octave), colors='b')
    plt.vlines(0.1, np.min(a_note_octave), np.max(a_note_octave), colors='r')
    plt.subplot(3, 3, 6)
    plt.scatter(valence, a_note_velocity, s=1, alpha=alpha)
    plt.title("note_velocity")
    plt.vlines(-0.1, np.min(a_note_velocity), np.max(a_note_velocity), colors='b')
    plt.vlines(0.1, np.min(a_note_velocity), np.max(a_note_velocity), colors='r')
    plt.subplot(3, 3, 7)
    plt.scatter(valence, a_rhythm_density, s=1, alpha=alpha)
    plt.title("rhythm_density")
    plt.vlines(-0.1, np.min(a_rhythm_density), np.max(a_rhythm_density), colors='b')
    plt.vlines(0.1, np.min(a_rhythm_density), np.max(a_rhythm_density), colors='r')
    plt.subplot(3, 3, 8)
    plt.scatter(valence, a_roman_numeral, s=1, alpha=alpha)
    plt.title("roman_numeral_chord")
    plt.vlines(-0.1, np.min(a_roman_numeral), np.max(a_roman_numeral), colors='b')
    plt.vlines(0.1, np.min(a_roman_numeral), np.max(a_roman_numeral), colors='r')
    plt.subplot(3, 3, 9)
    plt.scatter(valence, a_tempo, s=1, alpha=alpha)
    plt.title("tempo")
    plt.vlines(-0.1, np.min(a_tempo), np.max(a_tempo), colors='b')
    plt.vlines(0.1, np.min(a_tempo), np.max(a_tempo), colors='r')

    if not os.path.exists("plot"):
        os.mkdir("plot")
    plt.savefig("plot/plot.png", dpi=600)

    def matrix(x, y, binning=False, bin=None, title="matrix", bin_size=9, x_valence=True):
        plt.figure(figsize=(12, bin_size + 1))
        y_bin = y.copy()

        y_max = np.max(y)
        y_min = np.min(y)
        y_bin = np.floor(np.multiply(np.divide(np.subtract(y_bin, y_min), y_max - y_min), bin_size - 1))
        y_bin = y_bin.astype(np.int64)

        cm = np.zeros((3, bin_size), dtype=np.int64)
        idx = np.array(1)
        if x_valence:
            idx = np.array([np.add(x, 1), y_bin]).astype(np.int64)
        else:
            idx = np.array([x, y_bin]).astype(np.int64)
        print(idx.shape)
        print(tuple(idx))
        np.add.at(cm, tuple(idx), 1)
        cm = cm.T
        plt.subplot(1, 2, 1)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, origin='lower')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(bin_size)
        if not binning and bin is not None:
            plt.yticks(tick_marks, bin)
        else:
            plt.yticks(tick_marks, np.linspace(y_min, y_max, bin_size))
        if x_valence:
            plt.xticks(np.arange(3), ['low_valence', 'mid_valence', 'high_valence'])
        else:
            plt.xticks(np.arange(3), ['no_key', 'major', 'minor'])
        thresh = cm.max() / 2.
        fmt = 'd'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        if x_valence:
            plt.xlabel("valence")
        else:
            plt.xlabel("min_maj_2")
            plt.ylabel("min_maj_1")
        plt.tight_layout()

        cm2 = np.divide(cm, np.sum(cm, axis=0))
        if x_valence:
            cm2 = np.delete(cm2, (1), axis=1)
        else:
            cm2 = np.delete(cm2, (0), axis=1)


        plt.subplot(1, 2, 2)
        plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.Blues, origin='lower')
        plt.title(title + " proportion")
        plt.colorbar()
        tick_marks = np.arange(bin_size)
        if not binning and bin is not None:
            plt.yticks(tick_marks, bin)
        else:
            plt.yticks(tick_marks, np.linspace(y_min, y_max, bin_size))
        if x_valence:
            plt.xticks(np.arange(2), ['low_valence', 'high_valence'])
        else:
            plt.xticks(np.arange(2), ['major', 'minor'])
        thresh2 = cm2.max() / 2.
        fmt = '.3f'
        for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
            plt.text(j, i, format(cm2[i, j], fmt), horizontalalignment="center", color="white" if cm2[i, j] > thresh2 else "black")
        if x_valence:
            plt.xlabel("valence")
        else:
            plt.xlabel("min_maj_2")
            plt.ylabel("min_maj_1")
        plt.tight_layout()

        plt.savefig("plot/" + title + ".png", dpi=600)

    chord_bin = []
    for i, q in enumerate(['maj', 'min', 'aug', 'dim', 'sus4', 'dom7', 'min7']):
        for j, r in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            chord_bin.append(r + " " + q)
    chord_bin.insert(0, 'No chord')

    roman_numeral_bin = []
    for i, q in enumerate(['maj', 'min', 'aug', 'dim', 'sus4', 'dom7', 'min7']):
        for j, r in enumerate(['I', '#I', 'II', '#II', 'III', 'IV', '#IV', 'V', '#V', 'VI', '#VI', 'VII']):
            roman_numeral_bin.append(r + " " + q)
    roman_numeral_bin.insert(0, 'No chord')

    matrix(a_valence, a_key, title="key_matrix", bin_size=3, bin=['None', 'Major', 'Minor'])
    matrix(a_valence, a_chord, title="chord_matrix", bin_size=85, bin=chord_bin)
    matrix(a_valence, a_roman_numeral, title="roman_numeral_chord_matrix", bin_size=85, bin=roman_numeral_bin)
    matrix(a_valence, a_chord_quality, title="chord_quality_matrix", bin_size=8, bin=['No chord', 'maj', 'min', 'aug', 'dim', 'sus4', 'dom7', 'min7'])
    matrix(a_valence, a_tempo, title="tempo_matrix", bin_size=7)

    matrix(a_valence, a_melodic_contour, binning=True, title="melodic_contour_matrix")
    matrix(a_valence, a_note_density, binning=True, title="note_density_matrix")
    matrix(a_valence, a_note_octave, binning=True, title="note_octave_matrix")
    matrix(a_valence, a_note_velocity, binning=True, title="note_velocity_matrix")
    matrix(a_valence, a_rhythm_density, binning=True, title="rhythm_density_matrix")

    print(extractor.aggregate_maj_min_2(key))
    matrix(extractor.aggregate_maj_min_2(key), extractor.aggregate_maj_min_1(chord), title="maj_min_matrix", binning=True, bin_size=33, x_valence=False)

    print(np.max(extractor.aggregate_note_density(note_density) - a_note_density))
    print(np.max(extractor.aggregate_tempo(tempo) - a_tempo))
    matrix(a_valence, extractor.extract_tonic(key), title="tonic_matrix", bin_size=13, bin=['None', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
