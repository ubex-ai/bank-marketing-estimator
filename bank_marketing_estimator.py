from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import bank_marketing_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=400, type=int, help='batch size')
parser.add_argument('--train_steps', default=10000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y), feature_columns = bank_marketing_data.load_data()

    # Build 2 hidden layer DNN with 60, 60 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[60, 60]
    )

    # Train the Model.
    classifier.train(
        input_fn=lambda: bank_marketing_data.get_train_input(
            train_x,
            train_y,
            args.batch_size
        ),
        steps=args.train_steps
    )

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
