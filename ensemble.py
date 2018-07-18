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
import cv2

height,width=128,128
    
K.set_image_dim_ordering("tf")  # Theano dimension ordering: (channels, width, height)
                                # some changes will be necessary to run with tensorflow
validation_predicted_folder = "pickled_results/ISIC_2018_validation"                                      
validation_folder = "../datasets/2018validation"    

# For multi_gpu use uncommemt and set visible devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


ensemble_pkl_filenames = ["final_2018_0","final_2018_27","final_2018_29","final_2018_1","final_2018_12","final_2018_20","final_2018_11","final_2018_3","final_2018_17","final_2018_8","final_2018_26","final_2018_4","final_2018_10","final_2018_32","final_2018_23","final_2018_13","final_2018_5","final_2018_28","final_2018_31","final_2018_14","final_2018_6","final_2018_22","final_2018_18","final_2018_33","final_2018_9"]

def post_process(input_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    opening = cv2.morphologyEx(input_mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    ret,thresh = cv2.threshold(closing,127,255,0)
    im2,contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_max_area = 0
    max_index = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area>cnt_max_area:
            cnt_max_area = area
            max_index = i
    img = np.zeros(input_mask.shape,dtype=np.uint8)
    if len(contours)>=1:
        output_mask = cv2.drawContours(img, [contours[max_index]], 0,(255,255,255), -1)
    else:
        output_mask = img
    return output_mask

def img_sensitivity(y_true, y_pred):
    y_true = y_true.reshape(-1).astype(np.bool)
    y_pred = y_pred.reshape(-1).astype(np.bool)

    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + np.finfo(float).eps)

def img_specificity(y_true, y_pred):

    y_true = y_true.reshape(-1).astype(np.bool)
    y_pred = y_pred.reshape(-1).astype(np.bool)
    true_negatives = np.sum(np.round(np.clip((1-y_true) * (1-y_pred), 0, 1)))

    possible_negatives = np.sum(np.round(np.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + np.finfo(float).eps)
def img_accuracy(y_true, y_pred):
    y_true = y_true.reshape(-1).astype(np.bool)
    y_pred = y_pred.reshape(-1).astype(np.bool)

    true_negatives = np.sum(np.round(np.clip((1-y_true) * (1-y_pred), 0, 1)))
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    denom = y_true.shape[0]
    return (true_negatives+true_positives) / (denom + np.finfo(float).eps)
def join_predictions(pkl_folder, pkl_files, binary=False, threshold=0.5):
    n_pkl = float(len(pkl_files))
    print(n_pkl)
    array = None
    for fname in pkl_files:
        with open(os.path.join(pkl_folder,fname+".pkl"), "rb") as f:
            print(fname)
            tmp = pkl.load(f)
            tmp = tmp.astype(np.float)
            if binary:
                tmp = np.where(tmp>=threshold, 1, 0)
            if array is None:
                array = tmp
            else:
                array = array + tmp
                print(array.shape)
    avg_pool_images = array/n_pkl
    print(avg_pool_images.shape)
    with open('{}.pkl'.format(os.path.join(validation_predicted_folder,'ensemble')), 'wb') as f:
        pkl.dump(avg_pool_images, f)    
    return avg_pool_images
   
def predict_challenge(challenge_folder, challenge_predicted_folder, mask_pred_challenge=None, plot=True,validation=False):

    challenge_list = ISIC.list_from_folder(challenge_folder+'/image/')
    challenge_resized_folder = challenge_folder +"_{}_{}".format(height,width)

    challenge_resized_list =  [name.split(".")[0]+".jpg" for name in challenge_list] #challenge_list
    
    if (validation):
        mask_pred_challenge = np.where(mask_pred_challenge>=0.5, 1, 0)
        mask_pred_challenge = mask_pred_challenge*255
        dice=0
        jacc=0
        sen = 0
        acc = 0
        spec = 0
        mask_list = ISIC.list_masks_from_folder(challenge_folder+'/mask/')
        for i in range(len(mask_pred_challenge)):
            img = mask_pred_challenge[i,:,:,:].astype(np.uint8)
            img = post_process(img)
            orig_img_filename = os.path.join(challenge_folder+'/image/',challenge_list[i])
            orig_img_size = cv2.imread(orig_img_filename).shape[:2]
            resized_mask = cv2.resize(img,(orig_img_size[1],orig_img_size[0]),interpolation=cv2.INTER_LINEAR) #resize the predicted masks
            true_mask = ISIC.get_mask(image_name=mask_list[i], mask_folder=challenge_folder+'/mask/') # read the original mask
            
            current_dice, current_jacc = dice_jacc_single(true_mask, resized_mask, smooth = 0)   # find the jacc coeff for the resized masks
            dice = dice + current_dice
            jacc = jacc + current_jacc
            _,pred_mask = cv2.threshold(resized_mask,127,255,cv2.THRESH_BINARY)


            pred_mask = pred_mask/255
            curr_sen = img_sensitivity(true_mask,pred_mask)
            curr_acc = img_accuracy(true_mask,pred_mask)
            curr_spec = img_specificity(true_mask, pred_mask)
            print(curr_sen,curr_spec,curr_acc,current_dice,current_jacc)
            sen = sen + curr_sen
            spec = spec + curr_spec
            acc = acc + curr_acc
            # name = os.path.basename(challenge_list[i]).split('.')[0] + '_segmentation.png'
            # filename = './predicted_masks/'+name
            # cv2.imwrite(filename,resized_mask)
        dice = dice/mask_pred_challenge.shape[0]
        jacc = jacc/mask_pred_challenge.shape[0]
        print ("Original size validation dice coef      : {:.4f}".format(dice))
        print ("Original size validation jacc coef      : {:.4f}".format(jacc))
        acc = acc/len(mask_list)
        sen = sen/len(mask_list)
        spec = spec/len(mask_list)
        print ("Original size validation acc      : {:.4f}".format(acc))
        print ("Original size validation sensitivity     : {:.4f}".format(sen))
        print ("Original size validation specificity     : {:.4f}".format(spec))
    else:
        mask_pred_challenge = np.where(mask_pred_challenge>=0.5, 1, 0)
        mask_pred_challenge = mask_pred_challenge*255        
        for i in range(len(mask_pred_challenge)):
            img = mask_pred_challenge[i,:,:,:].astype(np.uint8)
            img = post_process(img)
            orig_img_filename = os.path.join(challenge_folder+'/image/',challenge_list[i])
            orig_img_size = cv2.imread(orig_img_filename).shape[:2]
            resized_mask = cv2.resize(img,(orig_img_size[1],orig_img_size[0]),interpolation=cv2.INTER_LINEAR) #resize the predicted masks
           
            _,pred_mask = cv2.threshold(resized_mask,127,255,cv2.THRESH_BINARY)
            name = os.path.basename(challenge_list[i]).split('.')[0] + '_segmentation.png'
            filename = './predicted_masks/val_submission3/'+name
            cv2.imwrite(filename,resized_mask)




print ("Start Challenge Validation"    )
val_array = join_predictions(pkl_folder = validation_predicted_folder, pkl_files=ensemble_pkl_filenames,binary=False)

predict_challenge(challenge_folder=validation_folder, challenge_predicted_folder=validation_predicted_folder,
                    mask_pred_challenge=val_array, plot=False,validation=False)
print(ensemble_pkl_filenames)