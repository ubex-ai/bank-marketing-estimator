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

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: bank_marketing_data.get_eval_input(
            test_x,
            test_y,
            args.batch_size
        )
    )

    print('\nValidation accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Try to predict results
    expected = test_y
    predictions = classifier.predict(
        input_fn=lambda: bank_marketing_data.get_eval_input(
            test_x,
            labels=None,
            batch_size=len(test_x)
        )
    )

    template = ('Prediction is "{}" ({:.1f}%), expected "{}"')
    totN, correctN, totY, correctY = 0, 0, 0, 0
    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        if expec:
            totY += 1
            if class_id == expec:
                correctY += 1
            print(template.format(
                bank_marketing_data.RESULTS[class_id],
                100 * probability,
                bank_marketing_data.RESULTS[expec]
            ))
        else:
            totN += 1
            if class_id == expec:
                correctN += 1

    print('\nNegative total {}, predicted {} ({:.1f}%); Positive total {}, predicted {} ({:.1f}%)'.format(
        totN, correctN, correctN / totN * 100,
        totY, correctY, correctY / totY * 100
    ))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
