import pandas as pd
import tensorflow as tf

TRAIN_PATH = '.\\bank-additional\\bank-additional-full.csv'
TEST_PATH = '.\\bank-additional\\bank-additional.csv'

CSV_COLUMN_NAMES = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                    'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate',
                    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'result']
CATEGORICAL_COLUMNS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                       'poutcome', 'result']
RESULTS = ['no', 'yes']

def load_data(y_name='result'):
    train = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0, delimiter=';')
    train_x, train_y = train, train.pop(y_name)
    train_y = train_y.map(dict(zip(RESULTS, range(len(RESULTS)))))

    test = pd.read_csv(TEST_PATH, names=CSV_COLUMN_NAMES, header=0, delimiter=';')
    test_x, test_y = test, test.pop(y_name)
    test_y = test_y.map(dict(zip(RESULTS, range(len(RESULTS)))))

    return (train_x, train_y), (test_x, test_y)


def feature_columns(features):
    # Feature columns describe how to use the input.
    feature_columns_list = []
    for key in features.keys():
        if key in CATEGORICAL_COLUMNS:
            feature_columns_list.append(
                tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
                    key=key,
                    vocabulary_list=features[key].unique()
                ))
            )
        else:
            feature_columns_list.append(tf.feature_column.numeric_column(key=key))

    return feature_columns_list
