import os
import sys
import argparse
import numpy as np
import csv
import json

from tslearn.clustering import TimeSeriesKMeans
from IPython import embed


def parse_json(filename):
    file = open(filename, "r", encoding='UTF8')
    parsed_json = json.loads(file.read())
    return parsed_json


def parse_annotation(annotations_path):
    annotation_rounds = []

    for filename in os.listdir(annotations_path):
        if os.path.splitext(filename)[1] != ".json":
            continue

        data = parse_json(os.path.join(annotations_path, filename))

        pieces = {}
        for annotation_id in data['annotations']:
            piece_id = annotation_id.split("_")[0]
            if piece_id not in data['pieces']:
                continue

            if piece_id not in pieces:
                pieces[piece_id] = {}
                pieces[piece_id]["name"] = data["pieces"][piece_id]["name"]
                pieces[piece_id]["midi"] = data["pieces"][piece_id]["midi"]
                pieces[piece_id]["measures"] = data["pieces"][piece_id]["measures"]
                pieces[piece_id]["duration"] = data["pieces"][piece_id]["duration"]
                pieces[piece_id]["arousal"] = []
                pieces[piece_id]["valence"] = []
                pieces[piece_id]["musicianship"] = []

            if data['annotations'][annotation_id]['musicianship'] >= 3:
                pieces[piece_id]["arousal"].append(data['annotations'][annotation_id]['arousal'])
                pieces[piece_id]["valence"].append(data['annotations'][annotation_id]['valence'])
                pieces[piece_id]["musicianship"].append(data['annotations'][annotation_id]['musicianship'])
            else: 
                pass

        annotation_rounds.append(pieces)

    joint_pieces = {}

    joint_piece_id = 0
    for pieces in annotation_rounds:
        for id, piece in pieces.items():
            joint_pieces["piece_" + str(joint_piece_id)] = piece
            joint_piece_id += 1

    return joint_pieces


def parse_emotion_dimension(piece, dimension_name, max_variance=0.1):

    # Clamp examples with length greater than the min length
    min_length = min([len(d) for d in piece[dimension_name]])
    data_dimension = np.array([d[:min_length] for d in piece[dimension_name]])

    rows_to_delete = np.where(np.var(data_dimension, axis=1) > max_variance)
    data_dimension = np.delete(data_dimension, rows_to_delete, axis=0)

    return data_dimension


def get_average_clustered_mean_values(valence_values):

    clusters = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0).fit_predict(valence_values)

    c1, c2,  c3 = [], [], []
    
    for j in range(len(clusters)):
        if clusters[j] == 0:
            c1.append(valence_values[j])
        elif clusters[j] == 1:
            c2.append(valence_values[j])
        elif clusters[j] == 2:
            c3.append(valence_values[j])
    
    var1 = np.mean(np.var(c1, axis=0))
    var2 = np.mean(np.var(c2, axis=0))
    var3 = np.mean(np.var(c3, axis=0))
    # min_var = min(min(var1, var2), var3)
    
    # not max var(2nd or 3rd) , more number ..
    if var1 >= var2 and var1 >= var3:
        if len(c2) > len(c3):
            valence_values = c2
        else:
            valence_values = c3
    elif var2 >= var1 and var2 >= var3:
        if len(c1) > len(c3):
            valence_values = c1
        else:
            valence_values = c3
    elif var3 >= var2 and var3 >= var1:
        if len(c2) > len(c1):
            valence_values = c2
        else:
            valence_values = c1
    
    valence_values = np.mean(valence_values, axis=0)
    return valence_values

# ANN_DIR ="./data/vgmidi_annotations/"


def get_preprocessed_valence_arousal_dict(dir):
    pieces = parse_annotation(dir)
    valence_preprocessed_result = {}
    arousal_preprocessed_result = {}
    for i, piece_id in enumerate(pieces):
        midi_name = os.path.basename(pieces[piece_id]["midi"])

        print("Processing...", midi_name)

        valence_data = parse_emotion_dimension(pieces[piece_id], "valence")
        arousal_data = parse_emotion_dimension(pieces[piece_id], "arousal", 0.2)
        # musicianship_data = np.array(pieces[piece_id]["musicianship"])

        clustered_mean_valence_data = get_average_clustered_mean_values(valence_data)
        fname = pieces[piece_id]["midi"].rstrip('.mid').rstrip('.MID').rstrip('.midi').rstrip('.MIDI')
        valence_preprocessed_result[fname] = clustered_mean_valence_data
        
        clustered_mean_arousal_data = get_average_clustered_mean_values(arousal_data)
        arousal_preprocessed_result[fname] = clustered_mean_arousal_data


    return (valence_preprocessed_result, arousal_preprocessed_result)

# # Parse arguments
# parser = argparse.ArgumentParser(description='train_generative.py')
# parser.add_argument('--annotations', type=str, required=True, help="Dir with annotation files.")
# parser.set_defaults(rmdup=True)

# opt = parser.parse_args()

# Parse music annotaion into a dict of pieces
# pieces = parse_annotation(opt.annotations)
# valence_preprocessed_result= {}
# for i, piece_id in enumerate(pieces):
#     midi_name = os.path.basename(pieces[piece_id]["midi"])

#     print("Processing...", midi_name)

#     valence_data = parse_emotion_dimension(pieces[piece_id], "valence")
#     arousal_data = parse_emotion_dimension(pieces[piece_id], "arousal")
#     # musicianship_data = np.array(pieces[piece_id]["musicianship"])

#     clustered_mean_valence_data= get_average_clustered_mean_values(valence_data)
#     valence_preprocessed_result[pieces[piece_id]["midi"]] = clustered_mean_valence_data

# embed()
# exit()