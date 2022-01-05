from midi_extractor import parse_events, make_piano_roll_sequences, sequences_to_midi_file, extract_features
from change_modality_utils import *
import os
import argparse
import numpy as np

# ending_measure: inclusive
def midi_to_midi_feature(midi_file_path, aggregation, starting_measure=0, ending_measure=-1, verbose=False):

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

    if aggregation:
        return extracted_features_to_aggregated_midi_feature(fname, features, sequences_length, starting_measure, ending_measure)
    else:
        return extracted_features_to_midi_feature(fname, features, sequences_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--starting_measure', default=0, type=int, help='starting measure number of MIDIs')
    parser.add_argument('-e', '--ending_measure', default=-1, type=int, help='ending measure number of MIDIs')
    parser.add_argument('--use_nn', action='store_true', help='use neural regression model to predict emotion')
    parser.add_argument('midi_files', nargs='*', help='input MIDI files')
    args = parser.parse_args()

    for i, d in enumerate(args.midi_files):
        f = midi_to_midi_feature(args.midi_files[i], int(args.starting_measure), int(args.ending_measure))
        print(f)
        e = midi_feature_dict_to_emotion(f, args.use_nn)
        print(e)
        f2 = emotion_dict_to_midi_feature(e)
        print(f2)

    while True:
        file_path = input("Input MIDI(.mid) file path (input \"exit\" to terminate): ")
        file_path.strip("\"")
        if file_path == "exit": break
        v = []
        a = []
        empty = []
        """
        for i in range(68):
            f = midi_to_midi_feature(file_path, True, i, i + 1)
            print(f)
            empty.append(f["empty"])
            if f["empty"] == 1:
                v.append(-2)
                a.append(-2)
            else:
                e = midi_feature_dict_to_emotion(f)
                print(e)
                v.append(e["valence"])
                a.append(e["arousal"])
        """
        f = midi_to_midi_feature(file_path, False)
        #print(f)
        empty = f["empty"]
        e = midi_feature_dict_to_emotion(f, args.use_nn)
        print(e)
        v = e["valence"]
        a = e["arousal"]
        """
        f2 = emotion_dict_to_midi_feature(e)
        print(f2)
        print()
        """
        
        #print(empty)
        #print(v)
        #print(a)
        print()
