import numpy as np
import matplotlib.pyplot as plt

from autoencoder import Autoencoder , DeepAutoencoder
from train import load_mnist
from train_state_AE_MLP import load_dataset_MLP
from  train_state_AE import load_dataset_Grid, load_dataset

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def select_observations(obs, num_samples=10):
    sample_index = np.random.choice(range(len(obs)), num_samples)
    return obs[sample_index]

def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    MLP = False
    Grid = False
    if MLP:
        MLPmodel = "model/DeepAutoencoder_MLPdate-2021-09-29-19-24-15"
        autoencoderMLP = DeepAutoencoder.load(MLPmodel)
        x_train = load_dataset_MLP(samples=2000)
        sample_obs = select_observations(x_train, num_samples=2)
        reconstructed_obs, latent_representations = autoencoderMLP.reconstruct(sample_obs)
        print(reconstructed_obs.reshape(2,6,7))
        print(sample_obs.reshape(2,6,7))

        # num_obs= 1000
        # sample_obs = select_observations(x_train, num_samples=num_obs)
        # _, latent_representations = autoencoderMLP.reconstruct(sample_obs)


    if Grid:
        Gridmodel = "model/AutoencoderCNN_date-2021-09-29-19-35-05"
        autoencoderGrid= Autoencoder.load(Gridmodel)
        x_train = load_dataset_Grid(samples=2000)
        sample_obs = select_observations(x_train, num_samples=2)
        reconstructed_obs, _ = autoencoderGrid.reconstruct(sample_obs)

    image = True
    if image:
        Gridmodel = "model/AutoencoderCNN_Image_date-2021-09-29-23-49-48"
        autoencoder = Autoencoder.load(Gridmodel)
        x_train = load_dataset(samples=100)

        num_sample_images_to_show = 8
        sample_obs = select_observations(x_train, num_samples=2)
        reconstructed_images, latent_representations = autoencoder.reconstruct(sample_obs)
        plot_reconstructed_images(sample_obs, reconstructed_images)




    mnist = False
    if mnist:
        autoencoder = Autoencoder.load("model/modelmnist")
        x_train, y_train, x_test, y_test = load_mnist()

        num_sample_images_to_show = 8
        sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
        reconstructed_images, _ = autoencoder.reconstruct(sample_images)
        plot_reconstructed_images(sample_images, reconstructed_images)

        # plot 2D latent space
        num_images = 6000
        sample_images, sample_labels = select_images(x_test, y_test, num_images)
        _, latent_representations = autoencoder.reconstruct(sample_images)
        plot_images_encoded_in_latent_space(latent_representations, sample_labels)






















