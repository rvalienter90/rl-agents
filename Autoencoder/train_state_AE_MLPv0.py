# from tensorflow.keras.datasets import mnist

from autoencoder import DeepAutoencoder
import pickle
import os, fnmatch
import numpy as np
import matplotlib.pyplot as plt
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
EPOCHS = 50


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):
    deep_autoencoder = DeepAutoencoder(
        input_shape=42,
        latent_space_dim=8
    )
    deep_autoencoder.summary()
    deep_autoencoder.compile(learning_rate)
    history = deep_autoencoder.train(x_train, batch_size, epochs)
    return deep_autoencoder,history

def load_dataset_MLP(pathbase="D:\Rodolfo\States\Dataset\Kinematics", samples= None):
    file_list = []
    for path, folders, files in os.walk(pathbase):
        for folder in folders:
            folder_path = pathbase + '/' + folder
            for path, foldersx, files in os.walk(folder_path):
                for file in files:
                    if fnmatch.fnmatch(file, '*.pickle'):
                        file_list.append(os.path.join(path, file))
    xtrain_list = []
    for idxf, file in enumerate(file_list):
        # if idxf >100:
        #     break
        if samples:
            if idxf >samples:
                break
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
            for idx, d in enumerate(data):
                if idx<3:
                    continue
                state = d['state']
                state = state.flatten()
                # state = state.reshape((11,11,7))
                # state = np.moveaxis(state, 0, -1)
                xtrain_list.append(state)
    xtrain = np.array(xtrain_list)
    return xtrain


def plot_performance(history):
    metric = 'mean_squared_error'
    # metric = 'mean_squared_error' 'mean_absolute_error' 'mean_absolute_percentage_error'
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.title('Model train')
    plt.ylabel('mean_squared_error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    print("test")
    x_train= load_dataset_MLP(samples=None)
    # x_train, _, _, _ = load_mnist()
    # deep_autoencoder,history  = train(x_train[:100], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    deep_autoencoder,history = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    deep_autoencoder.evaluate(x_train)
    # Plot training & validation accuracy values
    plot_performance(history)
    deep_autoencoder.save("model")


# tensorboard --logdir ./code/logs/training/