from tensorflow.keras.datasets import mnist

from autoencoder import Autoencoder,DeepAutoencoder
import pickle
import os, fnmatch
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import argparse
import logging
import tensorflow as tf
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--latent_space_dim', type=int, required=False,default=16)
parser.add_argument('--pathbase', type=str, required=False,default="D:\Rodolfo\States\Dataset\Kinematics")
parser.add_argument('--datatype', type=str, required=False,default="Grid")
parser.add_argument('--samples', type=str, required=False,default='None')
parser.add_argument('--learning_rate', type=float, required=False,default=0.0005)
parser.add_argument('--batch_size', type=int, required=False,default=128)
parser.add_argument('--epochs', type=int, required=False,default=5)
# Parse the argument
args = parser.parse_args()
#
# LEARNING_RATE = 0.0005
# BATCH_SIZE = 128
# EPOCHS = 5
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test

def train_image(x_train, learning_rate, batch_size, epochs,latent_space_dim=64):
    autoencoder = Autoencoder(
        input_shape=(64, 128, 1),
        conv_filters=(16, 32, 32, 32),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 1, 1, 1),
        latent_space_dim=latent_space_dim,
        name="Autoencoder_CNN_Image_" + str(latent_space_dim)
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    history = autoencoder.train(x_train, batch_size, epochs)
    return autoencoder,history
def train_grid(x_train, learning_rate, batch_size, epochs,latent_space_dim=32):
    autoencoder = Autoencoder(
        input_shape=(23, 23, 5),
        conv_filters=(16, 32, 32, 32),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 1, 1, 1),
        latent_space_dim=latent_space_dim,
        name="Autoencoder_CNN_Grid_" + str(latent_space_dim)
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    history = autoencoder.train(x_train, batch_size, epochs)
    return autoencoder,history
def train_MLP(x_train, learning_rate, batch_size, epochs,latent_space_dim=16):
    deep_autoencoder = DeepAutoencoder(
        input_shape=42,
        latent_space_dim=latent_space_dim,
        name="Autoencoder_MLP_Kinematics_" + str(latent_space_dim)
    )
    deep_autoencoder.summary()
    deep_autoencoder.compile(learning_rate)
    history = deep_autoencoder.train(x_train, batch_size, epochs)
    return deep_autoencoder,history


# "D:\Rodolfo\States\Dataset\Grid"
def load_dataset_Image(pathbase="D:\Rodolfo\States\Dataset\Image", samples= None):
    file_list = []
    for path, folders, files in os.walk(pathbase):
        for folder in folders:
            folder_path = pathbase + '/' + folder
            for path, foldersx, files in os.walk(folder_path):
                for file in files:
                    if fnmatch.fnmatch(file, '*.pickle'):
                        file_list.append(os.path.join(path, file))
    xtrain_list = []
    if samples:
        sample_index = np.random.choice(range(len(file_list)), samples)
        file_list_sample = np.array(file_list)[sample_index]
    else:
        file_list_sample = file_list

    for idxf, file in enumerate(file_list_sample):
        # if idxf >100:
        #     break
        # if samples:
        #     if idxf >samples:
        #         break
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
            for idx, d in enumerate(data):
                if idx<3:
                    continue
                state = d['state']
                # state = state.reshape((11,11,7))
                state = state[0,:,:]
                # state = np.moveaxis(state, 0, -1)
                # img = Image.fromarray(state * 255).show()
                state = state.reshape((64,128,1))

                xtrain_list.append(state)
    xtrain = np.array(xtrain_list)
    return xtrain
def load_dataset_Grid(pathbase="D:\Rodolfo\States\Dataset\Grid", samples= None):
    file_list = []
    for path, folders, files in os.walk(pathbase):
        for folder in folders:
            folder_path = pathbase + '/' + folder
            for path, foldersx, files in os.walk(folder_path):
                for file in files:
                    if fnmatch.fnmatch(file, '*.pickle'):
                        file_list.append(os.path.join(path, file))
    xtrain_list = []
    if samples:
        sample_index = np.random.choice(range(len(file_list)), samples)
        file_list_sample = np.array(file_list)[sample_index]
    else:
        file_list_sample = file_list
    for idxf, file in enumerate(file_list_sample):
        # if idxf >100:
        #     break
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
            for idx, d in enumerate(data):
                if idx<3:
                    continue
                state = d['state']
                # state = state.reshape((11,11,7))
                # Compute latent space , convert to H,W,C input to keras
                state = np.moveaxis(state, 0, -1)
                xtrain_list.append(state)
    xtrain = np.array(xtrain_list)
    return xtrain
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
    if samples:
        sample_index = np.random.choice(range(len(file_list)), samples)
        file_list_sample = np.array(file_list)[sample_index]
    else:
        file_list_sample = file_list

    for idxf, file in enumerate(file_list_sample):
        # if idxf >100:
        #     break
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



if __name__ == "__main__":
    print("************************Start************************")
    gpu_available = tf.test.is_gpu_available()
    is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
    is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3, 0))
    built_with_cuda= tf.test.is_built_with_cuda()
    print('gpu_available: ',gpu_available)
    print('is_cuda_gpu_available: ',is_cuda_gpu_available)
    print('is_cuda_gpu_min_3: ',is_cuda_gpu_min_3)
    print('built_with_cuda: ',built_with_cuda)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #
    # print("test")
    pathbase = args.pathbase
    latent_space_dim = args.latent_space_dim
    datatype=args.datatype
    learning_rate=args.learning_rate
    batch_size=args.batch_size
    epochs=args.epochs
    if args.samples == 'None':
        samples = None
    else:
        samples = int(args.samples)

    if datatype == "Grid":
        x_train= load_dataset_Grid(pathbase=pathbase, samples=samples)
        autoencoder, history = train_grid(x_train, learning_rate, batch_size, epochs,latent_space_dim=latent_space_dim)
        autoencoder.evaluate(x_train)

    elif datatype == "Image":
        x_train = load_dataset_Image(pathbase=pathbase, samples=samples)
        autoencoder, history = train_image(x_train, learning_rate, batch_size, epochs, latent_space_dim=latent_space_dim)
        autoencoder.evaluate(x_train)

    elif datatype == "K":
        x_train = load_dataset_MLP(pathbase=pathbase, samples=samples)
        autoencoder, history = train_MLP(x_train, learning_rate, batch_size, epochs, latent_space_dim=latent_space_dim)
        autoencoder.evaluate(x_train)
    logger_file = os.path.join(autoencoder.base_folder, 'app.log')
    logging.basicConfig(filename=logger_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logging.info('Logging')
    logging.info(args)

    # x_train= load_dataset_Grid(samples=None)

    # x_train, _, _, _ = load_mnist()
    # autoencoder = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)

    # Plot training & validation accuracy values
    autoencoder.save()
    print("************************End************************")

# tensorboard --logdir ./code/logs/training/