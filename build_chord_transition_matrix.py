import numpy as np
import pandas as pd
import json
import codecs

def category_to_index(valence_category, arousal_category):
    if isinstance(valence_category, (np.ndarray, np.generic)) and isinstance(arousal_category, (np.ndarray, np.generic))\
        and valence_category.shape[0] == arousal_category.shape[0]:
        ind = np.zeros_like(valence_category, dtype=int)
        ind[valence_category == 'M'] += 3
        ind[valence_category == 'H'] += 6
        ind[arousal_category == 'M'] += 1
        ind[arousal_category == 'H'] += 2
        return ind
    else:
        switcher = {
            'L': 0,
            'M': 1,
            'H': 2
        }
        return switcher.get(valence_category) * 3 + switcher.get(arousal_category)

if __name__ == '__main__':
    df = pd.read_csv("./output/extracted/vgmidi_emotion.csv")
    for i in range(13):
        yamaha_df = pd.read_csv("./output_yamaha/" + str(i + 1) + "/extracted/yamaha_emotion.csv")
        df = df.append(yamaha_df, ignore_index=True)
    #df.to_csv('./output/all_emotion.csv', index=False)

    """
    df['prev.valence.category']
    df['prev.arousal.category']
    df['valence.category']
    df['arousal.category']
    df['key.global.major']
    df['prev.roman.numeral']
    df['roman.numeral']
    """

    epsilon = 0.001

    df['sentiment.index'] = category_to_index(df['valence.category'].to_numpy(), df['arousal.category'].to_numpy())
    df['prev.sentiment.index'] = category_to_index(df['prev.valence.category'].to_numpy(), df['prev.arousal.category'].to_numpy())

    
    bct = np.zeros((2, 85, 85), dtype=np.float32)         # basic chord transition matrix
    sct = np.zeros((9, 2, 85, 85), dtype=np.float32)      # sentimental chord transition matrix
    svct = np.zeros((9, 9, 2, 85, 85), dtype=np.float32)  # sentiment-variational chord transition matrix

    for mode in range(2):
        print("mode = " + str(mode))
        temp_df = df.loc[df['key.global.major'] == mode, :]
        #temp_bct = pd.crosstab(temp_df['prev.roman.numeral'], temp_df['roman.numeral']).to_numpy()
        #bct[mode, :, :] = temp_bct
        for i in range(85):
            for j in range(85):
                bct[mode, i, j] = temp_df.loc[(temp_df['prev.roman.numeral'] == i) & \
                    (temp_df['roman.numeral'] == j), :].to_numpy().shape[0] + epsilon
            bct[mode, i, :] /= np.sum(bct[mode, i, :])

        for sentiment in range(9):
            print("  sentiment = " + str(sentiment))
            temp_df2 = temp_df.loc[(temp_df['ending'] == 0) & (temp_df['sentiment.index'] == sentiment), :]
            #temp_sct = pd.crosstab(temp_df2['prev.roman.numeral'], temp_df2['roman.numeral']).to_numpy()
            #sct[sentiment, mode, :, :] = temp_sct
            for i in range(85):
                for j in range(85):
                    sct[sentiment, mode, i, j] = temp_df2.loc[(temp_df2['prev.roman.numeral'] == i) & \
                        (temp_df2['roman.numeral'] == j), :].to_numpy().shape[0]
                if np.sum(sct[sentiment, mode, i, :]) != 0:
                    sct[sentiment, mode, i, :] /= np.sum(sct[sentiment, mode, i, :])

            for prev_sentiment in range(9):
                print("    prev_sentiment = " + str(prev_sentiment))
                temp_df3 = temp_df2.loc[temp_df2['prev.sentiment.index'] == prev_sentiment, :]
                #temp_svct = pd.crosstab(temp_df3['prev.roman.numeral'], temp_df3['roman.numeral']).to_numpy()
                #svct[prev_sentiment, sentiment, mode, :, :] = temp_svct
                for i in range(85):
                    for j in range(85):
                        svct[prev_sentiment, sentiment, mode, i, j] = temp_df3.loc[(temp_df3['prev.roman.numeral'] == i) & \
                            (temp_df3['roman.numeral'] == j), :].to_numpy().shape[0]
                    if np.sum(svct[prev_sentiment, sentiment, mode, i, :]) != 0:
                        svct[prev_sentiment, sentiment, mode, i, :] /= np.sum(svct[prev_sentiment, sentiment, mode, i, :])

    print("dump json...")
    matrices = {'basic': bct.tolist(), 'sentimental': sct.tolist(), 'sentiment-variational': svct.tolist()}
    json.dump(matrices, codecs.open('./output/chord_transition.json', 'w', encoding='utf-8'))

    

    
