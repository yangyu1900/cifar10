import numpy as np
from time import time
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import KFold
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils

# This is the function to unpickle raw data
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

# This is the function to plot confusion matrix
def plot_confusion_matrix(cm, classes, i):
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.xticks(fontsize=8, rotation='vertical')
    plt.yticks(fontsize=8, rotation='horizontal')
    plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_{i}.png', format='png')

# Extract data as matrix
def extract_data():
    data_batches = [f"../data/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
    raw_data = [unpickle(path) for path in data_batches]
    X = np.vstack([batch[b'data'] for batch in raw_data])
    y = np.hstack([batch[b'labels'] for batch in raw_data])
    label_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    return X, y, label_names

# Normalize data
def normalize_data(X):
    return X / 255.0

# The CNN classifier
if __name__ == '__main__':
    # Set parameters
    batch_size = 50
    num_classes = 10
    epochs = 60

    # Get normalized data
    X, y, label_names = extract_data()
    data_normalized = normalize_data(X)
    data = np.reshape(data_normalized, (50000, 32, 32, 3))
    label = np_utils.to_categorical(y, num_classes)

    # Set output file path
    result_file = 'cnn_report.csv'
    with open(result_file, 'a') as output:
        # Set the proportion of training data and validation data
        kf = KFold(n_splits=10)

        # Set a variable to record the number of cross-validation performed
        rd = 0

        # Set a variable to record average confusion matrix
        cm_avg = np.zeros((num_classes, num_classes))

        print("start")

        # Train classifier for 10-fold cross-validation
        for train, valid in kf.split(data):
            rd += 1
            t0 = time()

            # Create model and add layers
            model = Sequential([
                Conv2D(32, (5, 5), padding='same', input_shape=data.shape[1:]),
                Activation('relu'),
                Dropout(0.25),
                Conv2D(32, (5, 5)),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                Conv2D(64, (3, 3), padding='same'),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                Flatten(),
                Dense(512),
                Activation('tanh'),
                Dense(num_classes),
                Activation('softmax')
            ])

            # Initiate RMSprop optimizer
            opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            # Split data into training data and validation data
            X_train, X_valid, y_train, y_valid = data[train], data[valid], label[train], label[valid]

            # Fit model to data
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid), shuffle=True)

            # Predict validation data
            actual = y[valid]
            predicted = model.predict_classes(X_valid, batch_size=batch_size, verbose=1)
            t1 = time()

            # Output results
            try:
                output.write(f"round: {rd}\n")
                output.write(f"Classification report for classifier {model}:\n{metrics.classification_report(actual, predicted)}\n")
                cm = metrics.confusion_matrix(actual, predicted)
                cm_avg += cm
                confusionMatrixFilename = f'confusion_matrix_{rd}'
                confusion_matrix_file = pd.DataFrame(cm)
                confusion_matrix_file.to_csv(confusionMatrixFilename, sep=',', index=False, header=True)
                plot_confusion_matrix(cm, label_names, rd)
                output.write(f"Confusion matrix:\n{cm}\n\n")
                output.write(f"starting time: {t0}\n")
                output.write(f"ending time: {t1}\n")
                output.write(f"time consumed: {t1-t0}\n\n\n")
            except IOError:
                print('IOError')