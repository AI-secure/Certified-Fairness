from os import path
from urllib import request

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets import AbstractDataset


class AdultDataset_Covariate_Shifting(AbstractDataset):
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
        'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    train_labels_map = {'<=50K': 0, '>50K': 1}
    test_labels_map = {'<=50K.': 0, '>50K.': 1}

    def __init__(self, split, args, normalize=True):
        super().__init__('adult', split)

        train_data_file = path.join(self.data_dir, 'adult.data')
        test_data_file = path.join(self.data_dir, 'adult.test')

        if not path.exists(train_data_file):
            request.urlretrieve(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', train_data_file
            )
        if not path.exists(test_data_file):
            request.urlretrieve(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', test_data_file
            )


        P_dataset = pd.read_csv(train_data_file, sep=',', header=None, names=AdultDataset_Covariate_Shifting.column_names)
        Q_dataset = pd.read_csv(test_data_file, sep=',', header=0, names=AdultDataset_Covariate_Shifting.column_names)
        train_dataset = pd.concat([P_dataset,Q_dataset])


        # preprocess strings
        train_dataset = train_dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # drop missing values
        train_dataset.replace(to_replace='?', value=np.nan, inplace=True)
        train_dataset.dropna(axis=0, inplace=True)

        # encode labels
        train_dataset.replace(AdultDataset_Covariate_Shifting.train_labels_map, inplace=True)
        train_dataset.replace(AdultDataset_Covariate_Shifting.test_labels_map, inplace=True)


        # split features and labels
        train_features, train_labels = train_dataset.drop('income', axis=1), train_dataset['income']

        continuous_vars = []
        self.categorical_columns = []
        for col in train_features.columns:
            if train_features[col].isnull().sum() > 0:
                train_features.drop(col, axis=1, inplace=True)
            else:
                if train_features[col].dtype == np.object:
                    self.categorical_columns += [col]
                else:
                    continuous_vars += [col]

        # print(continuous_vars)
        # print(self.categorical_columns)

        protected_att = args.protected_att if args.protected_att is not None else 'sex'
        self.protected_unique = train_features[protected_att].nunique()
        protected_train = np.logical_not(pd.Categorical(train_features[protected_att]).codes)

        # Male: False

        # one-hot encode categorical data
        train_features = pd.get_dummies(train_features, columns=self.categorical_columns, prefix_sep='=')
        self.continuous_columns = [train_features.columns.get_loc(var) for var in continuous_vars]

        # print(len(train_features.columns))
        # print(len(test_features.columns))

        # add missing column to test dataset
        # test_features.insert(
        #     loc=train_features.columns.get_loc('native_country=Holand-Netherlands'),
        #     column='native_country=Holand-Netherlands', value=0
        # )

        self.one_hot_columns = {}
        for column_name in self.categorical_columns:
            ids = [i for i, col in enumerate(train_features.columns) if col.startswith('{}='.format(column_name))]
            if len(ids) > 0:
                assert len(ids) == ids[-1] - ids[0] + 1
            self.one_hot_columns[column_name] = ids
        # print('categorical features: ', self.one_hot_columns.keys())

        self.column_ids = {col: idx for idx, col in enumerate(train_features.columns)}

        features = torch.tensor(train_features.values.astype(np.float32), device=self.device)
        labels = torch.tensor(train_labels.values.astype(np.int64), device=self.device)
        protected = torch.tensor(protected_train.astype(np.bool), device=self.device)

        max_age = torch.max(features[:,0]).item()
        min_age = torch.min(features[:, 0]).item()
        # mean_age = torch.mean(features[:, 0]).item()
        mean_age = min_age + int((max_age-min_age)*0.2)
        print(f'maximal age: {max_age}')
        print(f'minimal age: {min_age}')
        print(f'mean age: {mean_age}')

        train_features = features[features[:, 0] < mean_age]
        train_labels = labels[features[:, 0] < mean_age]
        train_protected = protected[features[:, 0] < mean_age]

        test_features = features[features[:, 0] >= mean_age]
        test_labels = labels[features[:, 0] >= mean_age]
        test_protected = protected[features[:, 0] >= mean_age]


        self.X_train, self.X_val, self.y_train, self.y_val, self.protected_train, self.protected_val = train_test_split(
            train_features, train_labels, train_protected, test_size=0.2, random_state=0
        )

        self.X_test = torch.tensor(test_features, dtype=torch.float32, device=self.device)
        self.y_test = torch.tensor(test_labels, dtype=torch.int64, device=self.device)
        self.protected_test = torch.tensor(test_protected, dtype=torch.bool, device=self.device)


        if normalize:
            self._normalize(self.continuous_columns)

        self._assign_split()
