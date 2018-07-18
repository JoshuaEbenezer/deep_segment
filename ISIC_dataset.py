import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import yaml
from sklearn.cross_validation import train_test_split
import pandas as pd
from ldr import ldr
np.random.seed(4)

mean_imagenet = [123.68, 103.939, 116.779] # rgb

### Initialization for texture enhancement operations
C = np.array([0.15625,0.85938,-0.58203,-0.080078,-0.052979,-0.030762,0.0036621])
C_flip = np.flip(C,axis=0)

x_minus = np.zeros((7,7))
x_minus[:,3] = C_flip

x_plus = np.zeros((7,7))
x_plus[:,3] = C

y_plus = np.zeros((7,7))
y_plus[:,3] = C

y_minus = np.zeros((7,7))
y_minus[:,3] = C_flip

di = np.diag_indices(7)
rev_di = (np.array([0,1,2,3,4,5,6]),np.array([6,5,4,3,2,1,0]))
lud = np.zeros((7,7))
lud[di] = C_flip

ldd = np.zeros((7,7))
ldd[rev_di] = C

rud = np.zeros((7,7))
rud[rev_di] = C_flip        

rdd = np.zeros((7,7))
rdd[di] = C

### Initialization for LDR
U = np.zeros((255,255),dtype=np.float)
tmp_k = np.array(range(1,256))
for layer in range(1,256):
    U[:,layer-1] = np.minimum(tmp_k,256-layer) - np.maximum(tmp_k-layer,0)
alpha_ldr = 2.5

def finite_diff(img_intensity):
    out_xminus = cv2.filter2D(src=img_intensity,ddepth=-1,kernel=x_minus,anchor=(2,5))
    out_xplus = cv2.filter2D(src=img_intensity,ddepth=-1,kernel=x_plus,anchor=(2,1))
    out_yminus = cv2.filter2D(src=img_intensity,ddepth=-1,kernel=y_minus,anchor=(5,2))
    out_yplus = cv2.filter2D(src=img_intensity,ddepth=-1,kernel=y_plus,anchor=(1,2))

    out_lud = cv2.filter2D(src=img_intensity,ddepth=-1,kernel=lud,anchor=(5,5))
    out_ldd = cv2.filter2D(src=img_intensity,ddepth=-1,kernel=ldd,anchor=(5,1))
    out_rdd = cv2.filter2D(src=img_intensity,ddepth=-1,kernel=rdd,anchor=(1,1))
    out_rud = cv2.filter2D(src=img_intensity,ddepth=-1,kernel=rud,anchor=(1,5))

    output_image = ((out_xminus.astype(float))**2+(out_xplus.astype(float))**2+(out_yminus.astype(float))**2+(out_yplus.astype(float))**2+(out_lud.astype(float))**2+(out_ldd.astype(float))**2+(out_rud.astype(float))**2+(out_rdd.astype(float))**2)**0.5
    return np.expand_dims(output_image.astype(np.uint8),axis=2)

def get_labels(image_list, csv_file):
    image_list = [filename.split('.')[0] for filename in image_list]
    return pd.read_csv(csv_file,index_col=0).loc[image_list]['melanoma'].values.flatten().astype(np.uint8)

def get_mask(image_name, mask_folder, rescale_mask=True):
	# image_name = image_name.replace(".jpg","_segmentation.png")
    image_name = os.path.basename(image_name).split('.')[0] + '.jpg'
    img_mask = cv2.imread(os.path.join(mask_folder,image_name.replace(".jpg","_segmentation.png")),cv2.IMREAD_GRAYSCALE)
    if img_mask is None:
        img_mask = cv2.imread(os.path.join(mask_folder,image_name.replace(".jpg",".png")), 
                              cv2.IMREAD_GRAYSCALE)
    _,img_mask = cv2.threshold(img_mask,127,255,cv2.THRESH_BINARY)
    if rescale_mask:
        img_mask = img_mask/255.
    return img_mask
 
def get_color_image(image_name, image_folder, remove_mean_imagenet=True, use_hsv=False, remove_mean_samplewise=False,use_histeq=False,use_rgb_histeq=False,use_only_histeq=False,use_color_en=False,use_naik=False,use_hct=False,use_only_texture = False):
    if remove_mean_imagenet and remove_mean_samplewise:
        raise Exception("Can't use both sample mean and Imagenet mean")
    image_name = os.path.basename(image_name).split('.')[0] + '.jpg'
    img = cv2.imread(os.path.join(image_folder,image_name.replace(".jpg",".png")))
    if img is None:
        img = cv2.imread(os.path.join(image_folder,image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if (remove_mean_imagenet and use_only_histeq==False and use_rgb_histeq==False and use_naik==False and use_hct==False and use_only_texture==False):
        for channel in [0,1,2]:
            img[:,:,channel] -= mean_imagenet[channel]
    elif (remove_mean_samplewise and use_only_histeq==False and use_rgb_histeq==False):
        img_channel_axis = 2
        img -= np.mean(img, axis=img_channel_axis, keepdims=True)
    if use_hsv:
        img_all = np.zeros((img.shape[0],img.shape[1],6))
        img_all[:,:,0:3] = img
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_all[:,:,3:] = img_hsv
        img = img_all
    elif use_histeq:
        img_all = np.zeros((img.shape[0],img.shape[1],4))
        img_all[:,:,0:3] = img
        img_gray = (img[:,:,0]+img[:,:,1]+img[:,:,2] )/3
        img_histeq = cv2.equalizeHist(img_gray.astype(np.uint8))
        img_histeq = np.expand_dims(img_histeq,axis=2)
        img_all[:,:,3:] = img_histeq
        img = img_all
    elif use_rgb_histeq:
        img_all = np.zeros((img.shape[0],img.shape[1],4))
        
        img_gray = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),axis=2)

        img_histeq_b = np.expand_dims(cv2.equalizeHist(img[:,:,0].astype(np.uint8)),axis=2)
        img_histeq_g = np.expand_dims(cv2.equalizeHist(img[:,:,1].astype(np.uint8)),axis=2)
        img_histeq_r = np.expand_dims(cv2.equalizeHist(img[:,:,2].astype(np.uint8)),axis=2)
        img_all[:,:,0:3] = np.concatenate((img_histeq_b,img_histeq_g,img_histeq_r),axis=2)

        img_all[:,:,3:] = img_gray
        img = img_all.astype(np.float32)/255.0
    elif use_only_histeq:
        img_intensity = (img[:,:,0]+img[:,:,1]+img[:,:,2] )/3#np.expand_dims((img[:,:,0]+img[:,:,1]+img[:,:,2] )/3,axis=2)
        # img_histeq = cv2.equalizeHist(img_gray.astype(np.uint8))
        img_histeq = ldr(img_intensity.astype(np.uint8),alpha_ldr,U)  #np.expand_dims(img_histeq,axis=2)
        img = img_histeq
    elif use_color_en:
        img_all = np.zeros((img.shape[0],img.shape[1],3))
        img_intensity = np.expand_dims((img[:,:,0]+img[:,:,1]+img[:,:,2])/3,axis=2)
        img_histeq = np.expand_dims(cv2.equalizeHist(img_intensity.astype(np.uint8)),axis=2)
        img_histeq = img_histeq.astype(np.float32)
        alpha = img_histeq/img_intensity
        alpha_dash = np.zeros((img.shape[0],img.shape[1],1))
        result = np.zeros((img.shape[0],img.shape[1],3))
        mask = np.squeeze(alpha>1)
        alpha_dash = (255-img_histeq)/(255-img_intensity)
        replace = 255-alpha_dash*(255-img)
        result[mask,0:3] = replace[mask,0:3]
        neg_mask = np.squeeze(alpha<=1)
        orig = alpha*img
        result[neg_mask,0:3] = orig[neg_mask,0:3]
        img_all[:,:,0:3] = result
        img_all = img_all.astype(np.float32) 
        img = img_all
    elif use_naik:
        img_all = np.zeros((img.shape[0],img.shape[1],4))
        img_intensity = np.expand_dims((img[:,:,0]+img[:,:,1]+img[:,:,2])/3,axis=2)
        img_histeq = np.expand_dims(cv2.equalizeHist(img_intensity.astype(np.uint8)),axis=2)
        img_histeq = img_histeq.astype(np.float32)
        alpha = img_histeq/img_intensity
        alpha_dash = np.zeros((img.shape[0],img.shape[1],1))
        result = np.zeros((img.shape[0],img.shape[1],3))
        mask = np.squeeze(alpha>1)
        alpha_dash = (255-img_histeq)/(255-img_intensity)
        replace = 255-alpha_dash*(255-img)
        result[mask,0:3] = replace[mask,0:3]
        neg_mask = np.squeeze(alpha<=1)
        orig = alpha*img
        result[neg_mask,0:3] = orig[neg_mask,0:3]
        img_all[:,:,0:3] = result
        img_all[:,:,3:]=img_histeq
        img_all = img_all.astype(np.float32) 
        img = img_all
    elif use_hct:
        img_all = np.zeros((img.shape[0],img.shape[1],5))   # container
        img_intensity = np.expand_dims((img[:,:,0]+img[:,:,1]+img[:,:,2])/3,axis=2)

        # histogram equalization
        img_histeq = np.expand_dims(cv2.equalizeHist(img_intensity.astype(np.uint8)),axis=2)
        img_histeq = img_histeq.astype(np.float32)
        img_ldr = ldr(np.squeeze(img_intensity).astype(np.uint8),alpha_ldr,U)
        # Naik and Murthy color enhancement
        alpha = img_histeq/img_intensity
        alpha_dash = np.zeros((img.shape[0],img.shape[1],1))
        result = np.zeros((img.shape[0],img.shape[1],3))
        mask = np.squeeze(alpha>1)
        alpha_dash = (255-img_histeq)/(255-img_intensity)
        replace = 255-alpha_dash*(255-img)
        result[mask,0:3] = replace[mask,0:3]
        neg_mask = np.squeeze(alpha<=1)
        orig = alpha*img
        result[neg_mask,0:3] = orig[neg_mask,0:3]
        img_all[:,:,0:3] = result
        img_all[:,:,3]=np.squeeze(img_ldr)


        output_image = np.squeeze(finite_diff(img_intensity).astype(np.uint8))
        # ldr_out = ldr(output_image,alpha_ldr,U)

        img_all[:,:,4] = output_image
        img_all = img_all.astype(np.float32)
        img = img_all
    elif use_only_texture:
        img_intensity = np.expand_dims((img[:,:,0]+img[:,:,1]+img[:,:,2])/3,axis=2)

        # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # hue = hsv[:,:,0]
        # saturation = hsv[:,:,1]
        # value = hsv[:,:,2]
        # out_value = finite_diff(value)
        # out_hue = finite_diff(hue)
        # out_saturation = finite_diff(saturation)
        # output_hsv= np.concatenate((out_hue,out_saturation,out_value),axis=2)
        img = finite_diff(img_intensity)



    return img
        
def load_images(images_list, height, width, image_folder, mask_folder, remove_mean_imagenet=True, rescale_mask=True, use_hsv=False, remove_mean_samplewise=False,use_histeq=False,use_rgb_histeq=False,use_only_histeq=False,use_color_en=False,use_naik=False,use_hct=False,use_only_texture = False):
    if use_hsv:
        n_chan = 6
    elif (use_histeq==True or use_rgb_histeq==True or use_naik==True):
        n_chan = 4
    elif(use_only_histeq or use_only_texture):
        n_chan = 1
    elif use_hct:
        n_chan = 5
    else:
    	n_chan = 3
    img_array = np.zeros((len(images_list), height, width,n_chan), dtype=np.float32)
    if mask_folder:
        img_mask_array = np.zeros((len(images_list), height, width), dtype=np.float32)
    i = 0
    for image_name in images_list:
        img = get_color_image(image_name, image_folder, remove_mean_imagenet=remove_mean_imagenet,use_hsv=use_hsv,remove_mean_samplewise=remove_mean_samplewise,use_histeq=use_histeq,use_rgb_histeq=use_rgb_histeq,use_only_histeq=use_only_histeq,use_color_en=use_color_en,use_naik=use_naik,use_hct=use_hct,use_only_texture = use_only_texture)
        img_array[i] = img
        if mask_folder:
            img_mask = get_mask(image_name, mask_folder, rescale_mask)
            img_mask_array[i] =img_mask
        i = i+1
    if not mask_folder:
        return img_array
    else:
        return (img_array, img_mask_array.astype(np.uint8).reshape((img_mask_array.shape[0],img_mask_array.shape[1],img_mask_array.shape[2],1)))

def train_test_from_yaml(yaml_file, csv_file):
    with open(yaml_file,"r") as f:
        folds = yaml.load(f); 
    train_list, test_list = folds["Fold_1"]
    train_label = get_labels(train_list, csv_file=csv_file)
    test_label = get_labels(test_list, csv_file=csv_file)
    return train_list, train_label, test_list, test_label

def train_val_split(train_list, train_labels, seed, val_split = 0.20):
    train_list, val_list, train_label, val_label = train_test_split(train_list, train_labels, test_size=val_split, stratify=train_labels, random_state=seed)
    return train_list, val_list, train_label, val_label

def train_val_test_from_txt(train_txt, val_txt, test_txt):
    train_list =[]; val_list = []; test_list = [];
    with open(train_txt) as t:
        for img in t:
            img = img.strip()
            if img.endswith(".jpg"):
                train_list.append(img)
    with open(val_txt) as t:
        for img in t:
            img = img.strip()
            if img.endswith(".jpg"):
                val_list.append(img)
    with open(test_txt) as t:
        for img in t:
            img = img.strip()
            if img.endswith(".jpg"):
                test_list.append(img)
    print ("Found train: {}, val: {}, test: {}.".format(len(train_list),len(val_list),len(test_list)))
    return train_list, val_list, test_list
    
def list_from_folder(image_folder):
    image_list = []
    for image_filename in sorted(os.listdir(image_folder)):
        if image_filename.endswith(".jpg"):
            image_list.append(image_filename)
    print ("Found {} ISIC validation images.".format(len(image_list)))
    return image_list
def list_masks_from_folder(image_folder):
    image_list = []
    for image_filename in sorted(os.listdir(image_folder)):
        if image_filename.endswith(".png"):
            image_list.append(image_filename)
    print ("Found {} ISIC validation masks.".format(len(image_list)))
    return image_list

def move_images(images_list, input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, height=None, width=None, same_name=False):
    base_output_folder = output_image_folder
    base_output_mask_folder = output_mask_folder
    for k in range(len(images_list)):
        image_filename = images_list[k]
        image_name = os.path.basename(image_filename).split('.')[0]
        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)
        if input_mask_folder and not os.path.exists(output_mask_folder):
            os.makedirs(output_mask_folder)
        if height and width:
            img = cv2.imread(os.path.join(input_image_folder,image_filename))
            img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_image_folder,image_name+".png"), img)
            if input_mask_folder:
                img_mask = get_mask(image_filename, input_mask_folder, rescale_mask=False)
                img_mask = cv2.resize(img_mask, (width, height), interpolation = cv2.INTER_CUBIC)
                _,img_mask = cv2.threshold(img_mask,127,255,cv2.THRESH_BINARY)
                cv2.imwrite(os.path.join(output_mask_folder,image_name+".png"), img_mask)
        else:
            if not same_name:
                shutil.copyfile(os.path.join(input_image_folder, image_filename), os.path.join(output_image_folder,image_name+".jpg"))
            else:
                img = cv2.imread(os.path.join(input_image_folder,image_filename))
                cv2.imwrite(os.path.join(output_image_folder,image_name+".png"), img)
            
            if input_mask_folder:
                image_mask_filename = image_filename.replace(".jpg","_segmentation.png")
                shutil.copyfile(os.path.join(input_mask_folder,image_mask_filename), os.path.join(output_mask_folder,image_name+".png"))
            
def resize_images(images_list, input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, height, width):
    return move_images(images_list, input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, height, width)

def get_mask_full_sized(mask_pred, original_shape, output_folder = None, image_name = None):
    image_name = os.path.basename(image_name).split('.')[0] + '.jpg'
    mask_pred = cv2.resize(mask_pred, (original_shape[1], original_shape[0])) # resize to original mask size
    _,mask_pred = cv2.threshold(mask_pred,127,255,cv2.THRESH_BINARY)
    if output_folder and image_name:
        cv2.imwrite(os.path.join(output_folder,image_name.split('.')[0]+"_segmentation.png"), mask_pred)
    return mask_pred

def show_images_full_sized(image_list, img_mask_pred_array, image_folder, mask_folder, index, output_folder=None, plot=True):
    image_name = image_list[index]
    img = get_color_image(image_name, image_folder, remove_mean_imagenet=False).astype(np.uint8)
    if mask_folder:
        mask_true = get_mask(image_name, mask_folder, rescale_mask=False)
    mask_pred = get_mask_full_sized(img_mask_pred_array[index][0], img.shape, output_folder=output_folder, image_name = image_name)
    if mask_folder:
        if plot:
            f, ax = plt.subplots(1, 3)
            ax[0].imshow(img); ax[0].axis("off");
            ax[1].imshow(mask_true, cmap='Greys_r');  ax[1].axis("off"); 
            ax[2].imshow(mask_pred, cmap='Greys_r'); ax[2].axis("off"); plt.show()
        return img, mask_true, mask_pred
    else:
        if plot:
            f, ax = plt.subplots(1, 2)
            ax[0].imshow(img); ax[0].axis("off");
            ax[1].imshow(mask_pred, cmap='Greys_r'); ax[1].axis("off"); plt.show()
        return img, mask_pred
