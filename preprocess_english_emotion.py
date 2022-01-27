import numpy as np
import pandas as pd
from build_chord_transition_matrix import category_to_index

# for English_emotion data
def valence_category(valence):
    if isinstance(valence, (np.ndarray, np.generic)):
        a = np.full_like(valence, "H", dtype=np.str)
        a[valence <= 0.3325] = "M"
        a[valence <= 0.1125] = "L"
        return a
    else:
        if valence <= 0.1125:
            return "L"
        elif valence <= 0.3325:
            return "M"
        else:
            return "H"

# for English_emotion data
def arousal_category(arousal):
    if isinstance(arousal, (np.ndarray, np.generic)):
        a = np.full_like(arousal, "H", dtype=np.str)
        a[arousal < -0.1500] = "M"
        a[arousal <= -0.3350] = "L"
        return a
    else:
        if arousal <= -0.3350:
            return "L"
        elif arousal < -0.1500:
            return "M"
        else:
            return "H"

if __name__ == '__main__':
    # Amy Beth Warriner, Victor Kuperman, and Marc Brysbaert. Norms of valence, arousal, and dominance for 13,915 English lemmas. Behavior research methods 45(4): 1191–1207, 2013.
    df = pd.read_csv("./data/english_emotion/BRM-emot-submit.csv")
    data = {"Word": df["Word"],
        "Valence": (df["V.Mean.Sum"].to_numpy() - 5) / 4.0,
        "ValenceCategory": valence_category((df["V.Mean.Sum"].to_numpy() - 5) / 4.0),
        "Arousal": (df["A.Mean.Sum"].to_numpy() - 5) / 4.0,
        "ArousalCategory": arousal_category((df["A.Mean.Sum"].to_numpy() - 5) / 4.0),
        "CategoryIndex": category_to_index(valence_category((df["V.Mean.Sum"].to_numpy() - 5) / 4.0),
            arousal_category((df["A.Mean.Sum"].to_numpy() - 5) / 4.0))}
    df2 = pd.DataFrame(data)
    df2.to_csv("./output/EnglishSentiment2.csv", index=False)
