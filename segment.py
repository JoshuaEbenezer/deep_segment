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
from sklearn.model_selection import train_test_split
import cv2 

np.random.seed(3)
K.set_image_dim_ordering("tf")  # Theano dimension ordering: (channels, width, height)
                                # some changes will be necessary to run with tensorflow

# For multi_gpu use uncommemt and set visible devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Abstract: https://arxiv.org/abs/1703.04819

# Extract challenge Training / Validation / Test images as below
# Download from https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a



training_folder = "../datasets/training/image"
training_mask_folder = "../datasets/ISIC-2018_Training_Part1_GroundTruth"
training_labels_csv = "../datasets/ISIC-2018_Training_Part3_GroundTruth.csv"
training_split_yml = "../datasets/isic.yml"
validation_folder = "../2017test"    
test_folder = "../2017test"

# Place ISIC full dataset as below (optional)
isicfull_folder = "datasets/ISIC_Archive/image"
isicfull_mask_folder = "datasets/ISIC_Archive/mask"
isicfull_train_split="datasets/ISIC_Archive/train.txt"
isicfull_val_split="datasets/ISIC_Archive/val.txt"
isicfull_test_split="datasets/ISIC_Archive/test.txt"


# Folder to store predicted masks
validation_predicted_folder = "results/ISIC_2017_Predicted"                                      
test_predicted_folder = "results/ISIC-2018_Test_v2_Predicted"

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

seed = 1
height, width = 128, 128
nb_epoch = 200
model_name = "final_2018_3"

do_train = True # train network and save as model_name
do_predict = False # use model to predict and save generated masks for Validation/Test
do_ensemble = False # use previously saved predicted masks from multiple models to generate final masks
ensemble_pkl_filenames = ["model1", "model3","model5"]
model = 'unet'
batch_size = 4
loss_param = 'dice'
optimizer_param = 'adam'
monitor_metric = 'val_jacc_coef'
fc_size = 4096
mean_type = 'imagenet' # 'sample' 'samplewise'
rescale_mask = True
dataset='isic_notest' # 'isic' 'isicfull' 'isic_noval_notest' 'isic_other_split' 'isic_notest'
initial_epoch = 0 

use_hsv = False
use_histeq=False
use_rgb_histeq=False
use_only_histeq=False
use_color_en = False
use_naik = False
use_hct = True
use_only_texture = False

metrics = [jacc_coef,'accuracy',sensitivity,specificity,dice_coef]
if use_hsv:
    n_channels = 6
    print ("Using HSV")
elif use_histeq:
    n_channels = 4
    print ('Using gray hist. eq.')
elif use_rgb_histeq:
    n_channels = 4
    print ('Using BGR hist. eq.')
elif use_only_histeq:
    print ('Using only hist. eq. single channel')
    n_channels = 1
elif use_color_en:
    print ('Using color enhanced RGB')
    n_channels = 3
elif use_naik:
    print('Use Naik')
    n_channels = 4
elif use_hct:
	print('Using Heq, Naik, Texture')
	n_channels = 5
elif use_only_texture:
    print('Using only Texture')
    n_channels = 1    
else:
    print ('Using only RGB')
    n_channels = 3

print ("Using {} mean".format(mean_type))
remove_mean_imagenet=False
remove_mean_samplewise=False
remove_mean_dataset=False
if mean_type == 'imagenet':
    remove_mean_imagenet = True;
elif mean_type == 'sample':
    remove_mean_samplewise = True
elif mean_type == 'dataset':
    remove_mean_dataset = True
    train_mean = np.array([[[ 180.71656799]],[[ 151.13494873]],[[ 139.89967346]]]);
    train_std = np.array([[[1]],[[1]],[[ 1]]]); # not using std
else:
    raise Exception("Wrong mean type")
    
loss_options = {'BCE': 'binary_crossentropy', 'dice':dice_loss, 'jacc':jacc_loss, 'mse':'mean_squared_error'}
optimizer_options = {'adam': Adam(lr=1e-4),
                     'sgd': SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)}

loss = loss_options[loss_param]
optimizer = optimizer_options[optimizer_param]
model_filename = "../saved_models/2018_final/{}.h5".format(model_name)
print(model_filename)

if do_ensemble==False:
    print ('Create model'  )

    if model == 'unet':
        # model = models.Unet(height,width, loss=loss, optimizer = optimizer, metrics = metrics, fc_size = fc_size, channels=n_channels)
        model = models.get_unet(height,width, loss=loss, optimizer = optimizer, metrics = metrics,channels=n_channels)
    elif model == 'unet2':
        model = models.Unet2(height,width, loss=loss, optimizer = optimizer, metrics = metrics, fc_size = fc_size, channels=n_channels)
    elif model == 'vgg':
        model = models.VGG16(height,width, pretrained=False, freeze_pretrained = False, loss = loss, optimizer = optimizer, metrics = metrics,channels=n_channels)
    else:
        print ("Incorrect model name")

def myGenerator(train_generator, train_mask_generator, 
                remove_mean_imagenet=True, rescale_mask=True, use_hsv=False):
    while True:
        train_gen = next(train_generator)
        train_mask = next(train_mask_generator)
                
        if False: # use True to show images
            mask_true_show = np.where(train_mask>=0.5, 1, 0)
            mask_true_show = mask_true_show * 255
            mask_true_show = mask_true_show.astype(np.uint8)
            for i in range(train_gen.shape[0]):
                mask = train_mask[i].reshape((width,height))
                img=train_gen[i]
                img = img[0:3]
                img = img.astype(np.uint8)
                # img = img.transpose(1,2,0)
                f, ax = plt.subplots(1, 2)
                ax[0].imshow(img); ax[0].axis("off");
                ax[1].imshow(mask, cmap='Greys_r'); ax[1].axis("off"); plt.show()
        yield (train_gen, train_mask)

if do_train:
    if dataset == 'isicfull':
        n_samples = 2000 # per epoch
        print ("Using ISIC full dataset")
        train_list, val_list, test_list = ISIC.train_val_test_from_txt(isicfull_train_split, isicfull_val_split, isicfull_test_split)
        # folders for resized images
        base_folder = "datasets/training_{}_{}".format(height,width)
        image_folder = os.path.join(base_folder,"image")
        mask_folder = os.path.join(base_folder,"mask")
        if not os.path.exists(base_folder):
            print ("Begin resizing...")
            ISIC.resize_images(train_list+val_list+test_list, input_image_folder=isicfull_folder, 
                          input_mask_folder=isicfull_mask_folder, 
                              output_image_folder=image_folder.format(height,width), output_mask_folder=mask_folder, 
                              height=height, width=width)
            print ("Done resizing...")
            
    else:
        print ("Using ISIC 2018 dataset")
        # folders for resized images
        base_folder = "../datasets/2018training_{}_{}".format(height, width)
        image_folder = os.path.join(base_folder,"image")
        mask_folder = os.path.join(base_folder,"mask")
        rasterList = glob.glob(os.path.join(image_folder, '*.jpg'))
        # print(rasterList)
        # trainval_samples, test_samples = train_test_split(rasterList, test_size=0.1)
        train_list, test_list = train_test_split(rasterList, test_size=0.2)
        train_list,val_list = train_test_split(train_list,test_size=0.1)       


        # train_list, train_label, test_list, test_label = ISIC.train_test_from_yaml(yaml_file = training_split_yml, csv_file = training_labels_csv)
        # train_list, val_list, train_label, val_label = ISIC.train_val_split(train_list, train_label, seed = seed, val_split = 0.20)
        if not os.path.exists(base_folder):

            ISIC.resize_images(train_list+val_list+test_list, 
                          input_image_folder=training_folder, input_mask_folder=training_mask_folder, 
                          output_image_folder=image_folder, output_mask_folder=mask_folder, 
                          height=height, width=width)
        if dataset == "isic_notest": # previous validation split will be used for training
            train_list = train_list + val_list
            val_list = test_list
        elif dataset=="isic_noval_notest": # previous validation/test splits will be used for training
            monitor_metric = 'jacc_coef'
            train_list = train_list + val_list + test_list 
            val_list = test_list
        elif dataset=="isic_other_split": # different split, uses previous val/test for training
            seed = 82
            train_list1, train_list2, train_label1, train_label2 = ISIC.train_val_split(train_list, train_label, seed=seed, val_split=0.30)
            train_list = val_list+test_list+train_list1 
            val_list = train_list2
            test_list = val_list
        elif dataset=="isic_test_check":
            train_list = train_list + val_list + test_list
            test_image_folder = '../2017test_128_128/image/'
            test_mask_folder = '../2017test_128_128/mask/'
            test_list =  glob.glob(os.path.join(test_image_folder, '*.png'))
            val_list = test_list

        n_samples = len(train_list)

    print ("Loading images")
    train, train_mask = ISIC.load_images(train_list, height, width, 
                                          image_folder, mask_folder, 
                                      remove_mean_imagenet=remove_mean_imagenet, 
                                   rescale_mask=rescale_mask, use_hsv=use_hsv, remove_mean_samplewise=remove_mean_samplewise,use_histeq=use_histeq,use_rgb_histeq=use_rgb_histeq,use_only_histeq=use_only_histeq,use_color_en=use_color_en,use_naik=use_naik,use_hct=use_hct,use_only_texture = use_only_texture)
    if dataset!="isic_test_check":
        val, val_mask = ISIC.load_images(val_list, height, width, 
                                          image_folder, mask_folder,  
                                          remove_mean_imagenet=remove_mean_imagenet, 
                                   rescale_mask=rescale_mask, use_hsv=use_hsv, remove_mean_samplewise=remove_mean_samplewise,use_histeq=use_histeq,use_rgb_histeq=use_rgb_histeq,use_only_histeq=use_only_histeq,use_color_en=use_color_en,use_naik=use_naik,use_hct=use_hct,use_only_texture = use_only_texture)
        # test, test_mask = ISIC.load_images(test_list, height, width, 
        #                                   image_folder, mask_folder,
        #                                   remove_mean_imagenet=remove_mean_imagenet, 
        #                              rescale_mask=rescale_mask, use_hsv=use_hsv, remove_mean_samplewise=remove_mean_samplewise,use_histeq=use_histeq,use_rgb_histeq=use_rgb_histeq,use_only_histeq=use_only_histeq,use_color_en=use_color_en,use_naik=use_naik,use_hct=use_hct,use_only_texture = use_only_texture)
    else:
        val, val_mask = ISIC.load_images(val_list, height, width, 
                                          test_image_folder, test_mask_folder,  
                                          remove_mean_imagenet=remove_mean_imagenet, 
                                   rescale_mask=rescale_mask, use_hsv=use_hsv, remove_mean_samplewise=remove_mean_samplewise,use_histeq=use_histeq,use_rgb_histeq=use_rgb_histeq,use_only_histeq=use_only_histeq,use_color_en=use_color_en,use_naik=use_naik,use_hct=use_hct,use_only_texture = use_only_texture)


    print ("Done loading images")
    print(train.shape)
    print(val.shape)
    # print(test.shape)
    if remove_mean_dataset:
        print ("\nUsing Train Mean: {} Std: {}".format(train_mean, train_std))
        train = (train-train_mean)/train_std
        val = (val-train_mean)/train_std
        # test = (test-train_mean)/train_std

    print ("Using batch size = {}".format(batch_size))
    print ('Fit model')
    model_checkpoint = ModelCheckpoint(model_filename, monitor='val_jacc_coef', save_best_only=True, verbose=1)
    data_gen_args = dict(featurewise_center=False, 
                            samplewise_center=remove_mean_samplewise, 
                            featurewise_std_normalization=False, 
                            samplewise_std_normalization=False, 
                            zca_whitening=False, 
                            rotation_range=0, 
                            width_shift_range=0.1, 
                            height_shift_range=0.1, 
                            horizontal_flip=True, 
                            vertical_flip=True,
                            shear_range=0, 
                            zoom_range=0.2,
                            channel_shift_range=0,
                            fill_mode='reflect')
    data_gen_mask_args = data_gen_args.copy()
    data_gen_mask_args.update({'fill_mode':'nearest','samplewise_center':False})
    # data_gen_mask_args = dict(data_gen_args.items() + {'fill_mode':'nearest','samplewise_center':False}.items())
    print ("Create Data Generator" )
    train_datagen = ImageDataGenerator(**data_gen_args)
    train_mask_datagen = ImageDataGenerator(**data_gen_mask_args)
    train_generator = train_datagen.flow(train, batch_size=batch_size, seed=seed)
    train_mask_generator = train_mask_datagen.flow(train_mask, batch_size=batch_size, seed=seed)
    train_generator_f = myGenerator(train_generator, train_mask_generator, 
                                   remove_mean_imagenet=remove_mean_imagenet,
                                    rescale_mask=rescale_mask, use_hsv=use_hsv)
    # tbCallBack = TensorBoard(log_dir='./Graph', write_graph=True, write_images=True)
    if dataset=="isic_noval_notest":
        print ("Not using validation during training")
        history = model.fit_generator(
            train_generator_f,
            steps_per_epoch=n_samples/batch_size,
           epochs=nb_epoch, 
          callbacks=[model_checkpoint], initial_epoch=initial_epoch)
        model.save(model_filename)

    else:
        history = model.fit_generator(
            train_generator_f,
            steps_per_epoch=n_samples/batch_size,
            epochs=nb_epoch, validation_data=(val,val_mask), 
            callbacks=[model_checkpoint], initial_epoch=initial_epoch)

        train = None; train_mask = None # clear memory
        print ("Load best checkpoint")
        model.load_weights(model_filename) # load best saved checkpoint
        print(model_filename)

elif (do_ensemble==False):
    print ('Load model')
    model.load_weights(model_filename)
    print(model_filename)

def predict_challenge(challenge_folder, challenge_predicted_folder, mask_pred_challenge=None, plot=True,validation=False):
    print(model_filename)                                

    challenge_list = ISIC.list_from_folder(challenge_folder+'/image/')
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
            mask_folder=None, remove_mean_imagenet=remove_mean_imagenet, use_hsv = use_hsv,remove_mean_samplewise=remove_mean_samplewise,use_histeq=use_histeq,use_rgb_histeq=use_rgb_histeq,use_only_histeq=use_only_histeq,use_color_en=use_color_en,use_naik=use_naik,use_hct=use_hct,use_only_texture = use_only_texture)
    if remove_mean_dataset:
        challenge_images = (challenge_images-train_mean)/train_std
    if mask_pred_challenge is None:
        mask_pred_challenge = model.predict(challenge_images)
        mask_pred_challenge = np.where(mask_pred_challenge>=0.5, 1, 0)
        mask_pred_challenge = mask_pred_challenge * 255
        mask_pred_challenge = mask_pred_challenge.astype(np.uint8)    
        if (validation):
            dice=0
            jacc=0
            sen = 0
            acc = 0
            spec = 0
            mask_list = ISIC.list_masks_from_folder(challenge_folder+'/mask/')
            for i in range(mask_pred_challenge.shape[0]):
                img = mask_pred_challenge[i,:,:,:]
                img = post_process(img)
                orig_img_filename = os.path.join(challenge_folder+'/image/',challenge_list[i])
                orig_img_size = cv2.imread(orig_img_filename).shape[:2]
                resized_mask = cv2.resize(img,(orig_img_size[1],orig_img_size[0]),interpolation=cv2.INTER_LINEAR) #resize the predicted masks
                true_mask = ISIC.get_mask(image_name=mask_list[i], mask_folder=challenge_folder+'/mask/', rescale_mask=rescale_mask) # read the original mask
                current_dice, current_jacc = dice_jacc_single(true_mask, resized_mask, smooth = 0)   # find the jacc coeff for the resized masks
                dice = dice + current_dice
                jacc = jacc + current_jacc
                # name = os.path.basename(mask_list[i]).split('.')[0] + '.jpg'
                # filename = '../results/compare/naik/'+'naik_'+name
                # cv2.imwrite(filename,resized_mask)                
                _,pred_mask = cv2.threshold(resized_mask,127,255,cv2.THRESH_BINARY)


                pred_mask = pred_mask/255
                curr_sen = img_sensitivity(true_mask,pred_mask)
                curr_acc = img_accuracy(true_mask,pred_mask)
                curr_spec = img_specificity(true_mask, pred_mask)
                print(curr_sen,curr_spec,curr_acc,current_dice,current_jacc)
                sen = sen + curr_sen
                spec = spec + curr_spec
                acc = acc + curr_acc                
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
            print(model_filename)                                

        else:
            for i in range(mask_pred_challenge.shape[0]):
                img = mask_pred_challenge[i,:,:,:]
                img = post_process(img)
                orig_img_filename = os.path.join(challenge_folder,challenge_list[i])
                orig_img = cv2.imread(orig_img_filename)
                orig_img_size = orig_img.shape[:2]
                resized_mask = cv2.resize(img,(orig_img_size[1],orig_img_size[0]),interpolation=cv2.INTER_LINEAR) #resize the predicted masks
                _,pred_mask = cv2.threshold(resized_mask,127,255,cv2.THRESH_BINARY)
                img1_bg = cv2.bitwise_and(orig_img,orig_img,mask = pred_mask)                    
                name = os.path.basename(challenge_list[i]).split('.')[0] + '.jpg'
                filename = '../segmented2017/valid/'+name
                cv2.imwrite(filename,img1_bg)                
    else:
        if (validation):
            dice=0
            jacc=0
            sen = 0
            acc = 0
            spec = 0
            resized_masks=[]
            mask_list = ISIC.list_masks_from_folder(challenge_folder+'/mask/')
            for i in range(len(mask_pred_challenge)):
                img = mask_pred_challenge[i,:,:,:].astype(np.uint8)
                img = post_process(img)
                orig_img_filename = os.path.join(challenge_folder+'/image/',challenge_list[i])
                orig_img_size = cv2.imread(orig_img_filename).shape[:2]
                resized_mask = cv2.resize(img,(orig_img_size[1],orig_img_size[0]),interpolation=cv2.INTER_LINEAR) #resize the predicted masks
                resized_masks.append(resized_mask)                                
                true_mask = ISIC.get_mask(image_name=mask_list[i], mask_folder=challenge_folder+'/mask/', rescale_mask=rescale_mask) # read the original mask
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
            print(model_filename)                                

    # with open('{}.pkl'.format(os.path.join(challenge_predicted_folder,model_name)), 'wb') as f:
    #     pkl.dump(mask_pred_challenge, f)

    # challenge_predicted_folder = os.path.join(challenge_predicted_folder, model_name)

    # if not os.path.exists(challenge_predicted_folder):
    #     os.makedirs(challenge_predicted_folder)
    if(plot):
        print ("Start challenge prediction:")
        for i in range(len(challenge_list)):
            print ("{}: {}".format(i, challenge_list[i]))
            ISIC.show_images_full_sized(image_list = challenge_list, img_mask_pred_array = mask_pred_challenge, 
                    image_folder=challenge_folder, mask_folder=None, index = i, output_folder=challenge_predicted_folder, plot=plot)


def join_predictions(pkl_folder, pkl_files, binary=False, threshold=0.5):
    n_pkl = float(len(pkl_files))
    print(n_pkl)
    array = None
    for fname in pkl_files:
        with open(os.path.join(pkl_folder,fname+".pkl"), "rb") as f:
            print(f)
            tmp = pkl.load(f)
            tmp = np.array(tmp).astype(np.float)
            if binary:
                tmp = np.where(tmp>=threshold, 1, 0)
            if array is None:
                array = tmp
            else:
                array = array + tmp
    avg_pool_images = array/n_pkl
    avg_pool_images = avg_pool_images.astype(np.uint8)
    return avg_pool_images
    
if do_predict:
    # free memory
    train = None
    train_mask = None
    val = None
    test = None 
    
    print ("Start Challenge Validation"   )
    predict_challenge(challenge_folder=validation_folder, challenge_predicted_folder=validation_predicted_folder, plot=False,validation=False)
    # print "Start Challenge Test"    
    # predict_challenge(challenge_folder=test_folder, challenge_predicted_folder=test_predicted_folder, plot=False)
    
if do_ensemble:
    threshold = 0.5
    binary = False
    val_array = join_predictions(pkl_folder = validation_predicted_folder, pkl_files=ensemble_pkl_filenames, binary=binary, threshold=threshold)
    # test_array = join_predictions(pkl_folder = test_predicted_folder, pkl_files=ensemble_pkl_filenames, binary=binary, threshold=threshold)
    model_name="ensemble_{}".format(threshold)
    for f in ensemble_pkl_filenames:
        model_name = model_name + "_" + f
    print ("Predict Validation:")
    plot = False
    predict_challenge(challenge_folder=validation_folder, challenge_predicted_folder=validation_predicted_folder,
                        mask_pred_challenge=val_array, plot=plot,validation=True)