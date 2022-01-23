#Import all packages

from nd2reader import ND2Reader
import matplotlib.pyplot as plt
from skimage import data
import napari
from skimage.data import astronaut
import cv2
import numpy as np
from tifffile import imwrite
from nd2reader import ND2Reader
import numpy as np
import glob
import csv
import os



#All the functions you need for cropping and cleaning rois from segmented cell images

from read_roi import read_roi_file
from read_roi import read_roi_zip

def extract_roi_coordinates(roizipfile):
    coordinates_list=[] #[(left,top,right,bottom)] row and column coordinates for the rectangular roi
    roi = read_roi_zip(roizipfile); #read the roi contaning zip file
    a=list(roi.items()); #extract the items (individual rois) in the roi file
    l=len(a); #find the number of ROIs in the zip file
    for i in range(0,l):
        top=a[i][1]['top']
        left=a[i][1]['left']
        bottom=a[i][1]['top']+a[i][1]['height']
        right=a[i][1]['left']+a[i][1]['width']
        coordinates_list.append([(left,top,right,bottom)])

    return coordinates_list


def crop_using_roi_tuple(coordinates_tuple,image_array):

    return image_array[coordinates_tuple[1]:coordinates_tuple[3], coordinates_tuple[0]:coordinates_tuple[2]]



import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate

def cell_size_extractor(cleaned_image):
    cell_size=[];
    dummy=cleaned_image;
    dummy[dummy>0]=1;
    
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(dummy,kernel,iterations = 1)
    label_img = label(erosion)
    regions = regionprops(label_img)
    props = regionprops_table(label_img, properties=('centroid',
                                                 'orientation',
                                                 'major_axis_length',
                                                 'minor_axis_length','area'))
    
    for props in regions:
        
        volume=4/3*np.pi*props.major_axis_length*props.minor_axis_length*props.minor_axis_length
        cell_size.append([props.major_axis_length,props.minor_axis_length,props.area,volume])
    
    return cell_size


#to remove edge cells/half cells. algo: find intensity values of edges, create a set of those values and set those values to zero all over the image,
#since the edge values will be the values throughout the edge cells (since this is a binarized image), we will be only left with the mom and daughter cell
#of interest

import numpy as np
def clean_roi_borders(image_array):
    return_array=image_array
    shape=np.shape(image_array)
    nrows=shape[0];
    ncols=shape[1];
    left_edge=image_array[0:nrows-1,1];
    right_edge=image_array[0:nrows-1,ncols-1];
    top_edge=image_array[1,0:ncols-1];
    bottom_edge=image_array[nrows-1,0:ncols-1];
    all_edge_values=np.concatenate((left_edge,right_edge,top_edge,bottom_edge));
    unique_edge_values=np.unique(all_edge_values);
    l=len(unique_edge_values);

    for i in range(1,l): #starting from 1 to avoid zeros

        return_array[return_array==unique_edge_values[i]]=0;
    return return_array





from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np


def pre_clean(image_array):
    image_array = image_array.astype('uint8') # or c.astype(np.byte)
    all_labels = measure.label(image_array)
    blobs_labels = measure.label(image_array, background=0)
    return blobs_labels

#mu=107.916
#sigma=5.832

def noisify(np_image,mu,sigma):
    img=np_image
    for x in np.nditer(img, op_flags=['readwrite']):
        if x==0:
            x[...]=round(np.random.normal(mu, sigma, 1)[0]) 
    return img
