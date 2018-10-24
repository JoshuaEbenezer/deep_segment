# deep_segment
Segmentation of skin cancer images for submission to ISIC 2018. Base built on RECOD Titans' implementation.

Code is based on https://github.com/learningtitans/isbi2017-part1

## Code

**IMPORTANT: Please note all default paths in segment.py, pickle_results.py, and ensemble.py**

Please create folders *'pickled_results'* and *'predicted_masks'* within this directory. Change the relevant paths inside **pickle_result.py** and **ensemble.py** otherwise.

A folder by name *'saved_models'* must exist outside this directory.

Data must be stored as 128x218 images in the base training folder specified inside **segment.py**

**MAIN** file is **segment.py**

After training with **segment.py**, saved weights are stored in the specified path (please go through **segment.py** code) Default is *'../saved_models/2018_final'*

After training a model, you can either evaluate it using 'do_predict=True' in **segment.py** or use **'pickle_results.py'** to do it separately

The pickled results are stored in the folder *"pickle_results"*

After storing the pickled results, use **ensemble.py** to combine them all and evaluate them in their original sizes (if masks are available) or/and write the predicted, resized masks out to the folder *"predicted_masks"*

**ISIC_dataset.py**, **models.py**, and **metrics.py** are used by the main file

**post_process.py** contains the function for post-processing, but is not used by the main file since the function is also written inside segment.py

Change the numpy random seed to create a different training-validation split and different random weights

## Documentation and paper

Arxiv link: https://arxiv.org/abs/1807.04893v1
Preprint of longer version - journal_paper_preprint.pdf available on the repo

## Author
Joshua Ebenezer

## Date created
July 9th, 2018
