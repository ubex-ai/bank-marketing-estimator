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

    f_columns = feature_columns(train_x)

    return (train_x, train_y), (test_x, test_y), f_columns


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


def get_train_input(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(40000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset