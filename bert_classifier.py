
# Import
import ktrain
from ktrain import text
import numpy as np
import os
import codecs
import argparse

# Classes
classes = ['male', 'female']

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--datadir")
parser.add_argument("--k", default=10)
parser.add_argument("--length", type=int, default=3072)
args = parser.parse_args()

# Average accuracy
average_accuracy = np.zeros(args.k)

# For each fold
for k in range(args.k):
    # Validation directory
    fold_dir = os.path.join(args.datadir, "k{}".format(k))
    fold_val_dir = os.path.join(fold_dir, "val")

    # Load training and validation data from a folder
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(
        fold_dir,
        maxlen=512,
        preprocess_mode='bert',
        classes=classes
    )

    # Load BERT
    learner = ktrain.get_learner(
        text.text_classifier('bert', (x_train, y_train)),
        train_data=(x_train, y_train),
        val_data=(x_test, y_test),
        batch_size=16
    )

    # Get good learning rate
    learner.lr_find()

    # Plot
    learner.lr_plot()

    # Train the model
    learner.fit(2e-5, 20, early_stopping=5)
    # learner.fit_onecycle(2e-5, 1)

    # Get the predictor
    predictor = ktrain.get_predictor(learner.model, preproc)

    # Counting
    count = 0
    total = 0

    # For each class
    for c in ['male', 'female']:
        # Class directory
        class_val_dir = os.path.join(fold_val_dir, c)

        # For each file
        for class_file in os.listdir(class_val_dir):
            # Read file
            document_text = codecs.open(os.path.join(class_val_dir, class_file), "r", encoding="utf-8").read()

            # Data list
            data = list()

            # For each part
            for pos in range(0, len(document_text), args.length):
                data.append(document_text[pos:pos+args.length])
            # end for

            # Predict class
            pred = np.average(predictor.predict(data, return_proba=True), axis=1)

            # Predicted class
            if pred[0] >= pred[1]:
                predicted_class = 'male'
            else:
                predicted_class = 'female'
            # end if

            # Accuracy
            if predicted_class == c:
                count += 1
            # end if

            # Total
            total += 1
        # end for
    # end for

    # Accuracy
    accuracy = count / total

    # Print success rate
    print("ACCURACY TEST : {}".format(accuracy))
    average_accuracy[k] = accuracy
# end for

# Print average
print("Average accuracy : {}".format(np.mean(average_accuracy)))