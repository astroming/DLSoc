import os, pickle, random, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

warnings.filterwarnings('ignore')

class SOCloader(Dataset):
    """
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, flag=None):
        self.args = args
        self.flag = flag
        # pre_process
        self.feature_df, self.df_static, self.df_task, self.labels_df = self.load_all(flag='TRAIN')
        # fit scaler with training data
        self.scaler_ts = RobustScaler().fit(self.feature_df)
        self.scaler_static = RobustScaler().fit(self.df_static)
        self.scaler_l = MinMaxScaler().fit(self.labels_df)

        if flag == 'TEST':
            self.feature_df, self.df_static, self.df_task, self.labels_df = self.load_all(flag=flag)

        # scaler transform on data
        self.feature_df.iloc[:] = self.scaler_ts.transform(self.feature_df)
        self.df_static.iloc[:] = self.scaler_static.transform(self.df_static)
        self.labels_df.iloc[:, 0] = self.scaler_l.transform(self.labels_df)

        # all ids for data loader #
        self.feature_names = self.feature_df.columns
        self.all_IDs = self.feature_df.index.unique()

        print(f'{flag} sample size:', len(self.all_IDs))

    def load_all(self, flag=None):
        # time_now = time.time()
        file = open(self.args.processed_path, 'rb')
        all_cores = pickle.load(file)
        file.close()

        all_cores.reset_index(drop=True, inplace=True)
        idx = all_cores.index.values
        random.seed(self.args.cv_seed)
        random.shuffle(idx)
        bin = int(len(idx) / self.args.cv_folders)
        start = bin * self.args.cv_id
        end = min(bin * (self.args.cv_id + 1), len(idx))
        valid_idx = idx[start:end]
        train_idx = np.setdiff1d(idx, valid_idx, assume_unique=True)

        if flag == 'TRAIN':
            df_idx = train_idx
        elif flag == 'TEST':
            df_idx = valid_idx
        df_cores = all_cores.iloc[df_idx]
        df_cores.reset_index(drop=True, inplace=True)

        df_ts, df_static = df_cores[self.args.ts_bands], df_cores[self.args.static_f]
        df_task = df_cores['task']
        df_y = df_cores[self.args.predtarget]

        lengths = df_ts.applymap(lambda x: len(x)).values
        self.max_seq_len = lengths[0, 0]

        df = pd.concat((pd.DataFrame({col: df_ts.loc[row, col] for col in df_ts.columns}).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df_ts.shape[0])), axis=0)

        return df, df_static, df_task.to_frame(), df_y.to_frame()

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        static_x = self.df_static.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        tasks = self.df_task.loc[self.all_IDs[ind]].values

        return torch.from_numpy(batch_x), torch.from_numpy(static_x), torch.from_numpy(tasks), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)