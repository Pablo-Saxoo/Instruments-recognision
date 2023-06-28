import pandas as pd
import sqlite3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import librosa
import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, models
from sklearn.model_selection import train_test_split

def sq_connect(sq_base):
    # Connect to sql base
    conn = None
    try:
        conn = sqlite3.connect(sq_base)
        print('Connection successful')
    except Error as e:
            print(f'Error {e} occured')
    return conn


def sq_exec(conn, query):
    curr = conn.cursor()
    result = None
    try:
          curr.execute(query)
          result = curr.fetchall()
          print('Exec successful')
          return result
    except Error as e:
        print(f'Error {e} occurred')


def data_prep(query, conn):
     # Data preprocessing

    dataset = pd.read_sql_query(query, conn)
    dataset.drop(columns='id', inplace=True)

    drop_ind = [27, 28, 29, 40, 41]
    for i in drop_ind:
        dataset.drop(dataset[dataset.filename == f'sound_files/cello_margarita_pitch_stability/akg/00{i}.wav'].index, inplace=True)


    dataset.reset_index(drop=True, inplace=True)


    y = dataset.pop('instrument')
    instruments = list(y.unique())
    inst_num = np.arange(len(instruments))
    inst_dict = dict(zip(instruments, inst_num))
    y.replace(inst_dict, inplace=True)


    return dataset, y



def get_features(dataset):
    """ Create Pandas Frame and add columns with diffrent features """
    features_frame = pd.DataFrame()
    mel_list = []
    chroma_list = []
    mfcc_list = []

    for i in range(len(dataset)):
        path = dataset.loc[i, "filename"]
        y, sr = librosa.load(path)

        # Return melspectogram and take a mean arr
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec = np.mean(mel_spec, axis=1)
        mel_list.append(mel_spec)
    
        # Return chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma = np.mean(chroma, axis=1)
        chroma_list.append(chroma)

        # Return mel-freq cepstral coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        mfcc = np.mean(mfcc, axis=1)
        mfcc_list.append(mfcc)


    features_frame['Melspectogram'] = mel_list
    features_frame['Chromagram'] = chroma_list
    features_frame['MFCC'] = mfcc_list

    return features_frame


def unpack_features(features):
    
    mfcc_data = pd.DataFrame(features.Melspectogram.values.tolist(), index=features.index)
    mfcc_data = mfcc_data.add_prefix('mfcc_')

    chroma_data = pd.DataFrame(features.Chromagram.values.tolist(), index=features.index)
    chroma_data = chroma_data.add_prefix('chroma_')

    mel_data = pd.DataFrame(features.Melspectogram.values.tolist(), index=features.index)
    mel_data = mel_data.add_prefix('mel_')

    all_data = pd.concat([mel_data, mfcc_data, chroma_data], axis=1)

    return all_data






if __name__ == '__main__':

    SQ_BASE = 'database.sqlite'
    QUERY = 'SELECT * FROM  `sounds+akg`'

    # Load sql table to pandas dataframe
    conn = sq_connect(SQ_BASE)
    data, y = data_prep(QUERY, conn)

    # Get training dataset
    features = get_features(data)
    training_data = unpack_features(features)


    X_train, X_valid, y_train, y_valid = train_test_split(training_data, y, train_size=0.7, random_state=42)