"""
This script preprocesses data and prepares data to be actually used in training
emotion_dict = {'ang': 0,
                'hap': 1,
                'sad': 2,
                'neu': 4}
# original
# emotion_dict = {'ang': 0,
#                 'hap': 1,
#                 'exc': 2,
#                 'sad': 3,
#                 'fru': 4,
#                 'fea': 5,
#                 'sur': 6,
#                 'neu': 7,
#                 'xxx': 8,
#                 'oth': 8}

"""
import re
import os
import pickle
import unicodedata
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

logging.basicConfig(filename="memo_1.txt", level=logging.INFO)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """
    Lowercase, trim, and remove non-letter characters
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def transcribe_sessions():
    """
    audio: transcription
    """
    file2transcriptions = {}
    useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)
    transcript_path = '/home/Data/IEMOCAP_session_only/Session{}/dialog/transcriptions/'
    for sess in range(1, 6):
        transcript_path_i = transcript_path.format(sess)
        for f in os.listdir(transcript_path_i):
            with open('{}{}'.format(transcript_path_i, f), 'r') as f:
                all_lines = f.readlines()
            for l in all_lines:
                logging.info(l)
                audio_code = useful_regex.match(l).group()
                transcription = l.split(':')[-1].strip()
                # assuming that all the keys would be unique and hence no `try`
                file2transcriptions[audio_code] = transcription
    with open('../data/t2e/audiocode2text.pkl', 'wb') as file:
        pickle.dump(file2transcriptions, file)
    return file2transcriptions


def prepare_text_data(audiocode2text, id_emo_fts):
    """
    In audio_fts.csv, do
    1. Remove useless label
    2. map extreme rare emotion to related one
    3. oversample rare emotion
    4. create gender information
    5. Normalization
    6. split train/test data

    based on processed wav_id in audio_fts.csv, do
    1. Normalize txt
    2. split train/test of txt fts with audiocode2text

    """
    # Prepare text data
    df = pd.read_csv(id_emo_fts, header=None)
    if df.columns[0] == 0:
        df.columns = ["wav_file",
                      "label",
                      "rmse_m",
                      "rmse_std",
                      "rmse_range",
                      "harm_m",
                      "harm_std",
                      "pitch_m",
                      "pitch_std",
                      "pitch_range"]
    df.label = pd.to_numeric(df.label, errors='coerce')
    print("Before filter:", df.shape)
    # Delete fea
    #df = df[df['label'].isin([0, 1, 2, 3, 4, 5, 6, 7])]

    # ang hap exc sad fru neu
    df = df[df['label'].isin([0, 1, 2, 3, 4, 7])]

    # change 7 to 6
    #df['label'] = df['label'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5})
    # ang hap exc->hap sad fru->sad neu
    df['label'] = df['label'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 7: 3})
    print("after filter:", df.shape)
    df_group = df.groupby(by=["label"])

    print(df_group.aggregate(np.mean))

    df.to_csv('test_csv/data/pre-processed/audio_feature_filtered.csv', index=False)

    x_train, x_test = train_test_split(df, test_size=0.20)
    x_train.to_csv('test_csv/data/s2e/audio_train.csv', index=False)
    x_test.to_csv('test_csv/data/s2e/audio_test.csv', index=False)

    text_train = pd.DataFrame()
    text_train['wav_file'] = x_train['wav_file']
    text_train['label'] = x_train['label']
    text_train['transcription'] = [normalizeString(audiocode2text[code])
                                   for code in x_train['wav_file']]

    text_test = pd.DataFrame()
    text_test['wav_file'] = x_test['wav_file']
    text_test['label'] = x_test['label']
    text_test['transcription'] = [normalizeString(audiocode2text[code])
                                  for code in x_test['wav_file']]

    text_train.to_csv('test_csv/data/t2e/text_train.csv', index=False)
    text_test.to_csv('test_csv/data/t2e/text_test.csv', index=False)
    
    # Generate corpus for extracting tfidf of other db
    text = pd.concat([text_train, text_test], axis=0)
    text = text.drop(columns=["label"])
    text.to_csv("test_csv/data/combined/corpus_text.csv", index=False)

    print(text_train.shape, text_test.shape)


def pring_df(df_group):
    print("groupby label", df_group)


def main():
    prepare_text_data(transcribe_sessions())

def combine_fts():
    x_train_text = pd.read_csv('test_csv/data/t2e/text_train.csv')
    x_test_text = pd.read_csv('test_csv/data/t2e/text_test.csv')

    y_train_text = x_train_text['label']
    y_test_text = x_test_text['label']

    x_train_audio = pd.read_csv('test_csv/data/s2e/audio_train.csv')
    x_test_audio = pd.read_csv('test_csv/data/s2e/audio_test.csv')

    y_train_audio = x_train_audio['label']
    y_test_audio = x_test_audio['label']

    y_train = y_train_audio  # since y_train_audio == y_train_text
    y_test = y_test_audio  # since y_train_audio == y_train_text

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features_text = tfidf.fit_transform(x_train_text.append(x_test_text).transcription).toarray()

    x_train_text = features_text[:x_train_text.shape[0]]
    x_test_text = features_text[-x_test_text.shape[0]:]

    print(features_text.shape, x_train_text.shape, x_test_text.shape)

    combined_x_train = np.concatenate((np.array(x_train_audio[x_train_audio.columns[2:]]), x_train_text), axis=1)
    combined_x_test = np.concatenate((np.array(x_test_audio[x_test_audio.columns[2:]]), x_test_text), axis=1)

    print(combined_x_train.shape, combined_x_test.shape)

    combined_features_dict = {}

    combined_features_dict['x_train'] = combined_x_train
    combined_features_dict['x_test'] = combined_x_test
    combined_features_dict['y_train'] = np.array(y_train)
    combined_features_dict['y_test'] = np.array(y_test)

    with open('test_csv/data/combined/combined_features.pkl', 'wb') as f:
        pickle.dump(combined_features_dict, f)


if __name__ == '__main__':

    main()
