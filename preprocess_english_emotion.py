import pandas as pd
from read_features import valence_category, arousal_category
from build_chord_transition_matrix import category_to_index

if __name__ == '__main__':
    df = pd.read_csv("./data/english_emotion/BRM-emot-submit.csv")
    data = {"Word": df["Word"],
        "Valence": (df["V.Mean.Sum"].to_numpy() - 5) / 4.0,
        "ValenceCategory": valence_category((df["V.Mean.Sum"].to_numpy() - 5) / 4.0),
        "Arousal": (df["A.Mean.Sum"].to_numpy() - 5) / 4.0,
        "ArousalCategory": arousal_category((df["A.Mean.Sum"].to_numpy() - 5) / 4.0),
        "CategoryIndex": category_to_index(valence_category((df["V.Mean.Sum"].to_numpy() - 5) / 4.0),
            arousal_category((df["A.Mean.Sum"].to_numpy() - 5) / 4.0))}
    df2 = pd.DataFrame(data)
    df2.to_csv("./output/NewEnglishSentiment.csv", index=False)
