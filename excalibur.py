#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import volun_data



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = volun_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=5)

    # Train the Model.
    classifier.train(
        input_fn=lambda:volun_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    ##dont need this
    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:volun_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model

    print("Hi, we want to get to know you a little bit better so we can place you with the best volunteer oppertunity!!\n")
    expected = ['Classroom language link volunteer', 'distribute groceries to those in need','Intake assistant for hopital', 'Visiting elderly at senior facility','ETHS science fair']
    predict_x = {
        'Age':[int(input("What is your age? "))],
        'children':[int(input("Would you like to help children? (Scale 1 to 10): "))],
        'teenagers':[int(input("Would you like to help teenagers? (Scale 1 to 10): "))],
        'elderly':[int(input("Would you like to help the elderly? (Scale 1 to 10): "))],
        'outdoors':[int(input("Would you like to work outdoors? (Scale 1 to 10): "))],
        'sick':[int(input("Would you like to help the sick? (Scale 1 to 10): "))],
        'evanston':[int(input("Would you like to help to clean up Evanston? (Scale 1 to 10): "))],
         'schools':[int(input("Would you like to help in schools? (Scale 1 to 10): "))],
    }


    predictions = classifier.predict(
        input_fn=lambda:volun_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    # we can change this easily to show top 3 reccomendations or something
    
    for pred_dict, expec in zip(predictions, expected):
        template = ('\nWe recconmend you volunteer with {} ({:.1f}%)')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(volun_data.Organization[class_id],
                              100 * probability))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
tf.app.run(main)