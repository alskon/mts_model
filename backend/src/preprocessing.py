import pandas as pd


features = ['сумма', 'сегмент_arpu', 'частота', 'объем_данных', 'on_net', 'продукт_1',
           'продукт_2', 'зона_1', 'зона_2', 'pack_freq', 'частота_пополнения', 'секретный_скор']

target = 'binary_target'
path_file_train = './/train//train.csv'


class PreprocessingData:
    def __init__(self, features, train_data):
        self.features = features
        self.target = target
        self.nan_values = {}
        self.calc_nan(train_data)

    def calc_nan(self, train, type_fill='median'):
        for feature in self.features:
            if type_fill == 'median':
                self.nan_values[feature] = train[feature].median()
            elif type_fill == 'mean':
                self.nan_values[feature] = train[feature].mean()
            else:
                self.nan_values[feature] = -9999

    def prep_input(self, input):
        input.set_index('client_id', drop=True, inplace=True)
        input_copy = input.copy()
        non_features = [feat for feat in list(input_copy.columns) if feat not in self.features]
        input_copy.drop(non_features, axis=1, inplace=True)
        for feature in self.features:
            input_copy[feature] = input_copy[feature].fillna(self.nan_values[feature])
        return input_copy


def main_preprocessing(path_file_input):
    train_data = pd.read_csv(path_file_train)
    input_data = pd.read_csv(path_file_input)
    prep_data = PreprocessingData(features, train_data)
    prep_input = prep_data.prep_input(input_data)
    return prep_input

