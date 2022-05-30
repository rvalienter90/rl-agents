import  os, pickle
import post_process.visualization.visualization_utils as vutils
from post_process.applications.applications import *
base_path = "D:/Data/Data/Prediction/Results"
model_base_folder_pred = os.path.join(base_path, 'autoencoder', 'modelpredictions')
model_base_folder_predv2 = os.path.join(base_path, 'autoencoder', 'modelpredictionsv2')
Imagemodel64_pred2 = os.path.join(model_base_folder_pred,
                                  "Autoencoder_CNN_Image_64_date-2022-02-08-22-57-56")
history_path = os.path.join(Imagemodel64_pred2, "history.pkl")
with open(history_path, "rb") as f:
    history = pickle.load(f)

loss= history['loss'][99]
val_loss= history['val_loss'][99]
# parameter_path = os.path.join(Imagemodel64_pred2, "parameters.pkl")
# with open(parameter_path, "rb") as f:
#     parameters = pickle.load(f)

print('test')

def compare_loss(files,x_label,x_ticks_labels, fontsize=8):
    loss_list = []
    val_loss_list=[]
    for f in files:
        history_path = os.path.join(f, "history.pkl")
        with open(history_path, "rb") as f:
            history = pickle.load(f)

        loss = np.average(history['loss'][80:])
        val_loss = np.average(history['val_loss'][80:])
        loss_list.append(loss)
        val_loss_list.append(val_loss)

    fig, [ax1,ax2] = plt.subplots(1,2)
    x_axes = np.arange(0,len(files))
    ax1.plot(x_axes,loss_list)
    ax2.plot(x_axes,val_loss_list)

    ax1.set_ylabel('loss', fontsize=fontsize)
    ax2.set_ylabel('val_loss', fontsize=fontsize)
    ax1.set_xlabel(x_label, fontsize=fontsize)
    ax2.set_xlabel(x_label, fontsize=fontsize)

    ax1.set_xticks(x_axes)
    ax1.set_xticklabels(x_ticks_labels, fontsize=fontsize)

    ax2.set_xticks(x_axes)
    ax2.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    plt.show()


def hp_batch_size_plot():
    #64 latent_space_dim=64, learning_rate=0.0005
    if V1:
        batch_size_64 = os.path.join(model_base_folder_pred,
                                          "Autoencoder_CNN_Image_64_date-2022-02-08-22-57-56")
        history_path = os.path.join(batch_size_64, "history.pkl")


        batch_size_128 = os.path.join(model_base_folder_pred,
                                     "Autoencoder_CNN_Image_64_date-2022-02-08-23-06-11")




        batch_size_256 = os.path.join(model_base_folder_pred,
                                      "Autoencoder_CNN_Image_64_date-2022-02-08-23-06-12")

        files  = [batch_size_64,batch_size_128,batch_size_256]

        x_label = 'batch_size'
        x_ticks_labels = [64,128,256]
        compare_loss(files,x_label,x_ticks_labels)
    if V2:
        #latent_space_dim 128
        batch_size_8 = os.path.join(model_base_folder_predv2,
                                     "Autoencoder_CNN_Image_128_date-2022-02-13-14-39-38")

        batch_size_16 = os.path.join(model_base_folder_predv2,
                                     "Autoencoder_CNN_Image_128_date-2022-02-13-14-41-10")


        batch_size_32 = os.path.join(model_base_folder_predv2,
                                    "Autoencoder_CNN_Image_128_date-2022-02-13-14-39-39")



        batch_size_64 = os.path.join(model_base_folder_predv2,
                                     "Autoencoder_CNN_Image_128_date-2022-02-13-14-39-40")

        batch_size_64v1 = os.path.join(model_base_folder_pred,
                                     "Autoencoder_CNN_Image_64_date-2022-02-08-22-57-56")


        batch_size_128 = os.path.join(model_base_folder_pred,
                                     "Autoencoder_CNN_Image_64_date-2022-02-08-23-06-11")

        batch_size_256 = os.path.join(model_base_folder_pred,
                                      "Autoencoder_CNN_Image_64_date-2022-02-08-23-06-12")

        files = [batch_size_8,batch_size_16,batch_size_32,batch_size_64,batch_size_64v1,batch_size_128,batch_size_256]

        x_label = 'batch_size'
        x_ticks_labels = [8,16,32,64,128,256,512]
        compare_loss(files, x_label, x_ticks_labels)

    # # 32 latent_space_dim=256, learning_rate=0.0005
    # batch_size_32 = os.path.join(model_base_folder_pred,
    #                                   "Autoencoder_CNN_Image_256_date-2022-02-08-23-43-45")
    #
    # batch_size_64 = os.path.join(model_base_folder_pred,
    #                              "Autoencoder_CNN_Image_256_date-2022-02-08-23-43-43")
    #
    # # batch_size_128 = os.path.join(model_base_folder_pred,
    # #                              "Autoencoder_CNN_Image_64_date-2022-02-08-23-06-11")
    #
    # batch_size_256 = os.path.join(model_base_folder_pred,
    #                               "Autoencoder_CNN_Image_256_date-2022-02-08-23-14-37")
    #
    #
    # files  = [batch_size_32,batch_size_64,batch_size_256]
    #
    # compare_loss(files)


def hp_latent_size_plot():
    #batch_size=64, learning_rate=0.0005
    if V1:
        latent_size_64 = os.path.join(model_base_folder_pred,
                                          "Autoencoder_CNN_Image_64_date-2022-02-08-22-57-56")


        latent_size_256 = os.path.join(model_base_folder_pred,
                                      "Autoencoder_CNN_Image_256_date-2022-02-08-23-43-43")

        latent_size_1024 = os.path.join(model_base_folder_pred,
                                       "Autoencoder_CNN_Image_1024_date-2022-02-08-23-46-45")
        files  = [latent_size_64,latent_size_256,latent_size_1024]

        x_label = 'latent size'
        x_ticks_labels = [64, 256, 512]
        compare_loss(files, x_label, x_ticks_labels)

    if V2:
        # Namespace(batch_size=16, datatype='Image', epochs=100, latent_space_dim=16, learning_rate=0.0005, pathbase='/home/rvalienteromero/Coop/autoencoder/Dataset/Image', samples='None')
        latent_size_16 = os.path.join(model_base_folder_predv2,
                                    "Autoencoder_CNN_Image_16_date-2022-02-13-14-41-11")

        latent_size_32 = os.path.join(model_base_folder_predv2,
                                      "Autoencoder_CNN_Image_32_date-2022-02-13-14-41-11")

        latent_size_64 = os.path.join(model_base_folder_predv2,
                                      "Autoencoder_CNN_Image_64_date-2022-02-13-14-41-10")

        #v1
        # latent_size_128 = os.path.join(model_base_folder_pred,
        #                                "Autoencoder_CNN_Image_128_date-2022-02-08-23-14-34")
        # v1
        # latent_size_128 = os.path.join(model_base_folder_pred,
        #                               "Autoencoder_CNN_Image_512_date-2022-02-08-23-14-37")
        latent_size_128 = os.path.join(model_base_folder_predv2,
                                     "Autoencoder_CNN_Image_128_date-2022-02-13-14-41-10")

        latent_size_256 = os.path.join(model_base_folder_predv2,
                                      "Autoencoder_CNN_Image_256_date-2022-02-13-14-41-19")

        latent_size_512 = os.path.join(model_base_folder_predv2,
                                      "Autoencoder_CNN_Image_512_date-2022-02-13-14-41-19")


        latent_size_1024 = os.path.join(model_base_folder_predv2,
                                       "Autoencoder_CNN_Image_1024_date-2022-02-13-14-41-11")

        latent_size_2048 = os.path.join(model_base_folder_predv2,
                                        "Autoencoder_CNN_Image_2048_date-2022-02-13-14-49-18")

        # v1
        # latent_size_2536 = os.path.join(model_base_folder_pred,
        #                                 "Autoencoder_CNN_Image_1024_date-2022-02-08-23-46-45")

        files = [latent_size_16, latent_size_32, latent_size_64,latent_size_128,latent_size_256,latent_size_512,
                 latent_size_1024,latent_size_2048]

        x_label = 'latent_size'
        x_ticks_labels = [16,32,64,128,256,512,1024,2048]
        compare_loss(files, x_label, x_ticks_labels)

    # # batch_size=256, learning_rate=0.0005
    # latent_size_64 = os.path.join(model_base_folder_pred,
    #                               "Autoencoder_CNN_Image_64_date-2022-02-08-23-06-12")
    #
    # latent_size_128 = os.path.join(model_base_folder_pred,
    #                               "Autoencoder_CNN_Image_128_date-2022-02-08-23-14-34")
    #
    # latent_size_128_2= os.path.join(model_base_folder_pred,
    #                               "Autoencoder_CNN_Image_128_date-2022-02-08-23-14-35")
    #
    # latent_size_256 = os.path.join(model_base_folder_pred,
    #                                "Autoencoder_CNN_Image_256_date-2022-02-08-23-14-37")
    #
    # latent_size_512 = os.path.join(model_base_folder_pred,
    #                                "Autoencoder_CNN_Image_512_date-2022-02-08-23-14-37")
    #
    #
    # files  = [latent_size_64,latent_size_128,latent_size_128_2,latent_size_256,latent_size_512]
    #
    # compare_loss(files)


def hp_learning_rate_plot():
    #batch_size=256, latent_space_dim=512
    # latent_space_dim=512, learning_rate=0.0005
    lr_1 = os.path.join(model_base_folder_pred,
                                      "Autoencoder_CNN_Image_512_date-2022-02-08-23-14-37")

    # latent_space_dim=512, learning_rate=0.0002
    lr_2 = os.path.join(model_base_folder_pred,
                                  "Autoencoder_CNN_Image_512_date-2022-02-08-23-17-44")

    #latent_space_dim=512, learning_rate=0.0001
    lr_3 = os.path.join(model_base_folder_pred,
                                   "Autoencoder_CNN_Image_512_date-2022-02-08-23-19-10")

    # latent_space_dim=512, learning_rate=5e-05
    lr_4 = os.path.join(model_base_folder_pred,
                        "Autoencoder_CNN_Image_512_date-2022-02-08-23-29-36")

    lr_5 = os.path.join(model_base_folder_pred,
                                 "Autoencoder_CNN_Image_64_date-2022-02-08-22-57-56")

    files  = [lr_4,lr_3,lr_2,lr_1,lr_5]

    x_label = 'learning rate'
    x_ticks_labels = ['5e-05', '1e-04', '2e-04', '4e-04', '5e-04']
    compare_loss(files, x_label, x_ticks_labels)


def fast_plot():
    #batch_size=64, learning_rate=0.0005

    m_32_128 = os.path.join(model_base_folder_predv2,
                                   "Autoencoder_CNN_Image_128_date-2022-02-13-22-28-56")

    m_32_512 = os.path.join(model_base_folder_predv2,
                            "Autoencoder_CNN_Image_512_date-2022-02-13-22-28-31")

    m_8_1024 = os.path.join(model_base_folder_predv2,
                            "Autoencoder_CNN_Image_1024_date-2022-02-13-22-30-16")

    m_16_1024 = os.path.join(model_base_folder_predv2,
                            "Autoencoder_CNN_Image_1024_date-2022-02-13-22-30-41")


    files  = [m_32_128,m_32_512,m_8_1024,m_16_1024]

    x_label = 'fast test'
    x_ticks_labels = [1, 2, 3,4]
    compare_loss(files, x_label, x_ticks_labels)

V1= False
V2 = True
hp_batch_size_plot()
#
hp_latent_size_plot()
fast_plot()
# hp_learning_rate_plot()

