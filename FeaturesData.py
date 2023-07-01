import pandas as pd
import sqlite3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import librosa
import pathlib


class FeaturesDataset():
    def __init__(self, sq_base, query):
        self.sq_base = sq_base
        self.query = query

    def sq_connect(self):
        # Connect to sql base
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.sq_base)
            print("Connection successful")
        except Error as e:
            print(f"Error {e} occured")
        return self.conn

    def sq_exec(self):
        self.curr = self.conn.cursor()
        self.result = None
        try:
            self.curr.execute(self.query)
            self.result = self.curr.fetchall()
            print("Exec successful")
            return self.result
        except Error as e:
            print(f"Error {e} occurred")

    def data_prep(self):
        # Data preprocessing

        self.dataset = pd.read_sql_query(self.query, self.conn)
        self.dataset.drop(columns="id", inplace=True)

        drop_ind = [27, 28, 29, 40, 41]
        for i in drop_ind:
            self.dataset.drop(
                self.dataset[
                    self.dataset.filename
                    == f"sound_files/cello_margarita_pitch_stability/akg/00{i}.wav"
                ].index,
                inplace=True,
            )

        self.dataset.reset_index(drop=True, inplace=True)

        self.y = self.dataset.pop("instrument")
        self.instruments = list(self.y.unique())
        self.inst_num = np.arange(len(self.instruments))
        self.inst_dict = dict(zip(self.instruments, self.inst_num))
        self.y.replace(self.inst_dict, inplace=True)

        return self.dataset, self.y

    def get_features(self):
        """Create Pandas Frame and add columns with diffrent features"""
        self.features_frame = pd.DataFrame()
        mel_list = []
        chroma_list = []
        mfcc_list = []

        for i in range(len(self.dataset)):
            path = self.dataset.loc[i, "filename"]
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

        self.features_frame["Melspectogram"] = mel_list
        self.features_frame["Chromagram"] = chroma_list
        self.features_frame["MFCC"] = mfcc_list

        return self.features_frame

    def unpack_features(self):
        mfcc_data = pd.DataFrame(
            self.features_frame.Melspectogram.values.tolist(),
            index=self.features_frame.index,
        )
        mfcc_data = mfcc_data.add_prefix("mfcc_")

        chroma_data = pd.DataFrame(
            self.features_frame.Chromagram.values.tolist(),
            index=self.features_frame.index,
        )
        chroma_data = chroma_data.add_prefix("chroma_")

        mel_data = pd.DataFrame(
            self.features_frame.Melspectogram.values.tolist(),
            index=self.features_frame.index,
        )
        mel_data = mel_data.add_prefix("mel_")

        self.all_data = pd.concat([mel_data, mfcc_data, chroma_data], axis=1)

        return self.all_data

    def save_features(self, path):
        self.all_data.to_csv(path, index=False)