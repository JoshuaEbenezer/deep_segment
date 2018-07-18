import os
import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle as pkl
import ISIC_dataset as ISIC
from metrics import dice_loss, jacc_loss, jacc_coef, dice_jacc_mean,dice_coef,dice_jacc_single,sensitivity,specificity
import models
import glob
# ../results/2017_data_full/only_texture/use_only_texture.h5

use_histeq=False
use_rgb_histeq=False
use_only_histeq=False
use_color_en = False
use_naik = False
use_hct = True
use_hsv = False
use_only_texture = False
remove_mean_imagenet=False
remove_mean_samplewise = False

loss = dice_loss
optimizer = Adam(lr=1e-4)
metrics = [jacc_coef,'accuracy',sensitivity,specificity,dice_coef]


K.set_image_dim_ordering("tf")  # Theano dimension ordering: (channels, width, height)
                                # some changes will be necessary to run with tensorflow
validation_predicted_folder = "pickled_results/ISIC_2018_predicted"                                      
validation_folder = "../datasets/2018test"    

# For multi_gpu use uncommemt and set visible devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
height,width = 128,128
n_channels = 5
model_name = "final_2018_33"
model_filename = "../saved_models/2018_final/{}.h5".format(model_name)
model = models.get_unet(height,width, loss=loss, optimizer = optimizer, metrics = metrics,channels=n_channels)

def predict_challenge(challenge_folder, challenge_predicted_folder, mask_pred_challenge=None, plot=True,validation=False):
    print(model_filename)                                
    print ('Loading model')
    model.load_weights(model_filename)

    challenge_list = ISIC.list_from_folder(challenge_folder+"_{}_{}".format(height,width)+'/image/')
    challenge_resized_folder = challenge_folder +"_{}_{}".format(height,width)
    
    if not os.path.exists(challenge_resized_folder):
        if(validation):
            ISIC.resize_images(challenge_list, input_image_folder=challenge_folder+'/image/', input_mask_folder=challenge_folder+'/mask/', 
                          output_image_folder=challenge_resized_folder+'/image/', output_mask_folder=challenge_resized_folder+'/mask/', 
                          height=height, width=width)
        else:
            ISIC.resize_images(challenge_list, input_image_folder=challenge_folder+'/image/', input_mask_folder=None, 
                          output_image_folder=challenge_resized_folder+'/image/', output_mask_folder=None, 
                          height=height, width=width)            
    challenge_resized_list =  [name.split(".")[0]+".jpg" for name in challenge_list] #challenge_list
    challenge_images = ISIC.load_images(challenge_resized_list, 
            height, width, image_folder=challenge_resized_folder+'/image/',
            mask_folder=None, remove_mean_imagenet=remove_mean_imagenet, use_hsv = use_hsv,remove_mean_samplewise=remove_mean_samplewise,use_histeq=use_histeq,use_rgb_histeq=use_rgb_histeq,use_only_histeq=use_only_histeq,use_color_en=use_color_en,use_naik=use_naik,use_hct=use_hct,use_only_texture=use_only_texture)
    if mask_pred_challenge is None:
        mask_pred_challenge = model.predict(challenge_images)
    print(mask_pred_challenge.shape)

    with open('{}.pkl'.format(os.path.join(challenge_predicted_folder,model_name)), 'wb') as f:
        pkl.dump(mask_pred_challenge, f)


def main():
    predict_challenge(challenge_folder=validation_folder, challenge_predicted_folder=validation_predicted_folder, plot=False,validation=False)
if __name__ == "__main__":
	main()