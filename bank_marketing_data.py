import pandas as pd

TRAIN_PATH = '.\\bank-additional\\bank-additional-full.csv'
TEST_PATH = '.\\bank-additional\\bank-additional.csv'

def load_data(y_name='result'):
    train = pd.read_csv(TRAIN_PATH, header=0, delimiter=';')
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(TEST_PATH, header=0, delimiter=';')
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)
