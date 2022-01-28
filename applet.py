
from PyQt5.QtWidgets import QMainWindow, QApplication, QLineEdit, QComboBox, QDialogButtonBox, QLabel, QDialog, QPushButton, QFileDialog
from PyQt5 import uic
import sys
import all_funcs
from math import ceil
import os
import csv
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
from all_funcs import extract_roi_coordinates
from all_funcs import cell_size_extractor
from all_funcs import clean_roi_borders
from all_funcs import pre_clean
from all_funcs import noisify
from all_funcs import crop_using_roi_tuple


class UI(QDialog):

    def __init__(self):
        super(UI,self).__init__()
        #Load the ui file
        uic.loadUi("/Users/Saransh/Documents/Git/BakersPy/qt5/bakers_py_1.0.ui",self)
        self.setWindowTitle("BakersPy 1.0!")

        #Define our widgets

        #labels
        self.label_logo = self.findChild(QLabel,"logo")
        self.label_roi = self.findChild(QLabel,"ROIs_label")
        self.label_mask = self.findChild(QLabel,"mask_label")
        self.label_nd2 = self.findChild(QLabel,"nd2_label")
        self.label_c1 = self.findChild(QLabel,"Channel1_label")
        self.label_c2 = self.findChild(QLabel,"Channel2_label")
        self.label_c3 = self.findChild(QLabel,"Channel3_label")
        self.label_nfov = self.findChild(QLabel,"num_fovs_label")
        self.label_zslice = self.findChild(QLabel,"zslices_label")
        self.label_zspace = self.findChild(QLabel,"zspacing_label")

        #Lineedits
        self.edit_mask = self.findChild(QLineEdit,"mask_path")
        self.edit_roi = self.findChild(QLineEdit,"roi_path")
        self.edit_nd2 = self.findChild(QLineEdit,"nd2_path")
        self.edit_output_path = self.findChild(QLineEdit,"outputpath")


        #meta data
        self.edit_numfovs = self.findChild(QLineEdit,"num_fovs")
        self.edit_numslices = self.findChild(QLineEdit,"num_zslices")
        self.edit_zspace = self.findChild(QLineEdit,"z_space")


        #buttons
        self.button_run = self.findChild(QPushButton, "Run_Button")
        self.button_browse_nd2 = self.findChild(QPushButton, "browsend2")
        self.button_browse_mask = self.findChild(QPushButton, "browsemask")
        self.button_browse_roi = self.findChild(QPushButton, "browseroi")
        self.button_browse_output = self.findChild(QPushButton, "browseoutput")


        self.combo_c1 = self.findChild(QComboBox, "channel1_combo_box")
        self.combo_c2 = self.findChild(QComboBox, "channel2_combo_box")
        self.combo_c3 = self.findChild(QComboBox, "channel3_combo_box")


        # Click the buttons


        self.button_browse_nd2.clicked.connect(self.selectFile_nd2)
        self.button_browse_mask.clicked.connect(self.selectDir_mask)
        self.button_browse_roi.clicked.connect(self.selectDir_roi)
        self.button_browse_output.clicked.connect(self.selectDir_out)
        self.button_run.clicked.connect(self.process_files)


        #Show the App
        self.show()


    def selectFile_nd2(self):
        #print(QFileDialog.getOpenFileName()[0])
        #print(QFileDialog.getOpenFileName()[1])
        self.edit_nd2.setText(QFileDialog.getOpenFileName()[0])

    def selectDir_mask(self):
        self.edit_mask.setText(QFileDialog.getExistingDirectory())

    def selectDir_roi(self):
        self.edit_roi.setText(QFileDialog.getExistingDirectory())

    def selectDir_out(self):
        self.edit_output_path.setText(QFileDialog.getExistingDirectory())



    def process_files(self):




        #extract all the file Paths
        mask_path = self.edit_mask.text()
        roi_path = self.edit_roi.text()
        nd2_path = self.edit_nd2.text()
        output_path = self.edit_output_path.text()

        #nd2 file metadata
        num_fovs = float(self.edit_numfovs.text())
        num_slices = float(self.edit_numslices.text())
        zspace = float(self.edit_zspace.text())

        #extract channel names
        channel1 =self.combo_c1.currentText()
        channel2 =self.combo_c2.currentText()
        channel3 =self.combo_c3.currentText()


        #create subfolders to upload Results if not already created
        try:
            os.mkdir(output_path+'/'+channel1) #organelle 1
            os.mkdir(output_path+'/'+channel2) #organelle 2
            os.mkdir(output_path+'/'+channel3) #cells
            os.mkdir(output_path+'/'+channel1+'_masked')
            os.mkdir(output_path+'/'+channel2+'_masked')
        except Exception as e:
            print(e)

        cell_index = 1; #initialize cell index

        fov_start_index = int((2*num_fovs*num_slices)+ceil(num_slices/2))
        fov_end_index = int((3*num_fovs*num_slices)-(num_slices-ceil(num_slices/2)))
        fov_increment = int(num_slices)

        with open(output_path+'cell_size_state.csv', 'w') as csvfile:
            fieldnames = ['cellsize', 'state'] #state describes wheter the cell is in a budded state (2) or an unbudded state (1)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            with ND2Reader(nd2_path) as images:
                images.iter_axes = 'cvz';
                maskindex=1; #initialize mask/fov index

                for i in range(fov_start_index , fov_end_index , fov_increment):
                    #iterating through all central z slices/fovs

                    image=images[i];
                    image_array=np.array(image);

                    try:

                        if maskindex<11:

                            #extract roi's
                            roi_tuples= extract_roi_coordinates(roi_path+'/fov'+str(maskindex)+'.zip');
                            maskimage=cv2.imread(mask_path+'/cell_mask000'+str(maskindex-1)+'.tif',cv2.IMREAD_UNCHANGED)
                            maskarray=np.array(maskimage)
                            maskarray.astype(int)
                            l=len(roi_tuples);

                            #iterate through each roi in the fov
                            for j in range(0,l):

                                roi_coordinates=roi_tuples[j][0]; #get roi coordinates
                                cropped_image=crop_using_roi_tuple(roi_coordinates,maskarray) # crop the roi from the fov cell mask
                                pre_cleaned_img=pre_clean(np.array(cropped_image)) #binarize and label connected-regions before cleaning borders
                                cleaned_img=clean_roi_borders(pre_cleaned_img) #clean the borders of the ROI image so that only one cell remains in the centre
                                cleaned_img[cleaned_img>0]=1; #binarize the cleaned image
                                cleaned_img=cleaned_img.astype('uint8') #set the datatype of the numpy array to uint8 for further operations


                            #Channel 1 extraction - conversion of channel1 and cell zstack to a horizontal stack that can be analyzed later

                                start_Index=i-int((2*num_fovs*num_slices)+ceil(num_slices/2)-1); #channel 1 images start at index 1
                                cell_img=cleaned_img
                                c1_img=crop_using_roi_tuple(roi_coordinates,np.array(images[start_Index])); #intialize channel 1 hstack with the first image
                                for k in range(start_Index+1,start_Index+int(num_slices)): #add zslices horizontally
                                    curr_img=crop_using_roi_tuple(roi_coordinates,np.array(images[k]));
                                    c1_img=np.hstack((c1_img,curr_img))
                                    cell_img=np.hstack((cell_img,cleaned_img))
                                kernel = np.ones((3,3),np.uint8)
                                cell_img = cv2.dilate(cell_img,kernel,iterations = 1) #to connect mother and daughter cells

                                masked_hstack = np.multiply(cell_img,c1_img) #create a cell-masked image of channel 1


                                if np.shape(cell_size_extractor(cleaned_img))[0]==1: #unbudded cells

                                    imwrite(output_path+'/'+channel1+'/'+str(cell_index)+'.tif', c1_img)
                                    imwrite(output_path+'/'+channel3+'/'+str(cell_index)+'_cell.tif',cell_img)
                                    imwrite(output_path+'/'+channel1+'_masked'+'/'+str(cell_index)+'.tif',masked_hstack)
                                    writer.writerow({'cellsize':cell_size_extractor(cleaned_img)[0][3]*0.000091125,'state':1}) #cell volume added to the csv file
                                    cell_index+=1

                                elif np.shape(cell_size_extractor(cleaned_img))[0]==2: #budded cells
                                    imwrite(output_path+'/'+channel1+'/'+str(cell_index)+'.tif', c1_img)
                                    imwrite(output_path+'/'+channel3+'/'+str(cell_index)+'_cell.tif',cell_img)
                                    imwrite(output_path+'/'+channel1+'_masked'+'/'+str(cell_index)+'.tif',masked_hstack)
                                    writer.writerow({'cellsize':(cell_size_extractor(cleaned_img)[0][3]+cell_size_extractor(cleaned_img)[1][3])*0.000091125,'state':2}) #add mother and daughter cell volumes to the csv file
                                    cell_index+=1


                            #Channel 2 extraction - conversion of channel1 and cell zstack to a horizontal stack that can be analyzed late

                                start_Index=i-int((2*num_fovs*num_slices)+ceil(num_slices/2)-(num_fovs*num_slices)-1); #channel 2 start index
                                cell_img=cleaned_img
                                c2_img=crop_using_roi_tuple(roi_coordinates,np.array(images[start_Index])); #intialize channel 1 hstack with the first image

                                for k in range(start_Index+1,start_Index+int(num_slices)): #add zslices horizontally

                                    curr_img=crop_using_roi_tuple(roi_coordinates,np.array(images[k]));
                                    c2_img=np.hstack((c2_img,curr_img))
                                    cell_img=np.hstack((cell_img,cleaned_img))
                                kernel = np.ones((3,3),np.uint8)
                                cell_img = cv2.dilate(cell_img,kernel,iterations = 1) #to connect mother and daughter cells

                                masked_hstack = np.multiply(cell_img,c2_img) #create a cell-masked image of channel 1


                                if np.shape(cell_size_extractor(cleaned_img))[0]==1: #unbudded cells

                                    imwrite(output_path+'/'+channel2+'/'+str(cell_index)+'.tif', c2_img)
                                    imwrite(output_path+'/'+channel2+'_masked'+'/'+str(cell_index)+'.tif',masked_hstack)
                                    cell_index+=1




                                elif np.shape(cell_size_extractor(cleaned_img))[0]==2: #budded cells
                                    imwrite(output_path+'/'+channel2+'/'+str(cell_index)+'.tif', c2_img)
                                    imwrite(output_path+'/'+channel2+'_masked'+'/'+str(cell_index)+'.tif',masked_hstack)
                                    cell_index+=1


                        elif maskindex>10 & maskindex<101:

                            #extract roi's
                            roi_tuples= extract_roi_coordinates(roi_path+'/fov'+str(maskindex)+'.zip');
                            maskimage=cv2.imread(mask_path+'/cell_mask00'+str(maskindex-1)+'.tif',cv2.IMREAD_UNCHANGED)
                            maskarray=np.array(maskimage)
                            maskarray.astype(int)
                            l=len(roi_tuples);

                            #iterate through each roi in the fov
                            for j in range(0,l):

                                roi_coordinates=roi_tuples[j][0]; #get roi coordinates
                                cropped_image=crop_using_roi_tuple(roi_coordinates,maskarray) # crop the roi from the fov cell mask
                                pre_cleaned_img=pre_clean(np.array(cropped_image)) #binarize and label connected-regions before cleaning borders
                                cleaned_img=clean_roi_borders(pre_cleaned_img) #clean the borders of the ROI image so that only one cell remains in the centre
                                cleaned_img[cleaned_img>0]=1; #binarize the cleaned image
                                cleaned_img=cleaned_img.astype('uint8') #set the datatype of the numpy array to uint8 for further operations


                            #Channel 1 extraction - conversion of channel1 and cell zstack to a horizontal stack that can be analyzed later

                                start_Index=i-int((2*num_fovs*num_slices)+ceil(num_slices/2)-1); #channel 1 images start at index 1
                                cell_img=cleaned_img
                                c1_img=crop_using_roi_tuple(roi_coordinates,np.array(images[start_Index])); #intialize channel 1 hstack with the first image
                                for k in range(start_Index+1,start_Index+int(num_slices)): #add zslices horizontally
                                    curr_img=crop_using_roi_tuple(roi_coordinates,np.array(images[k]));
                                    c1_img=np.hstack((c1_img,curr_img))
                                    cell_img=np.hstack((cell_img,cleaned_img))
                                kernel = np.ones((3,3),np.uint8)
                                cell_img = cv2.dilate(cell_img,kernel,iterations = 1) #to connect mother and daughter cells

                                masked_hstack = np.multiply(cell_img,c1_img) #create a cell-masked image of channel 1


                                if np.shape(cell_size_extractor(cleaned_img))[0]==1: #unbudded cells

                                    imwrite(output_path+'/'+channel1+'/'+str(cell_index)+'.tif', c1_img)
                                    imwrite(output_path+'/'+channel3+'/'+str(cell_index)+'_cell.tif',cell_img)
                                    imwrite(output_path+'/'+channel1+'_masked'+'/'+str(cell_index)+'.tif',masked_hstack)
                                    writer.writerow({'cellsize':cell_size_extractor(cleaned_img)[0][3]*0.000091125,'state':1}) #cell volume added to the csv file
                                    cell_index+=1




                                elif np.shape(cell_size_extractor(cleaned_img))[0]==2: #budded cells
                                    imwrite(output_path+'/'+channel1+'/'+str(cell_index)+'.tif', c1_img)
                                    imwrite(output_path+'/'+channel3+'/'+str(cell_index)+'_cell.tif',cell_img)
                                    imwrite(output_path+'/'+channel1+'_masked'+'/'+str(cell_index)+'.tif',masked_hstack)
                                    writer.writerow({'cellsize':(cell_size_extractor(cleaned_img)[0][3]+cell_size_extractor(cleaned_img)[1][3])*0.000091125,'state':2}) #add mother and daughter cell volumes to the csv file
                                    cell_index+=1


                            #Channel 2 extraction - conversion of channel1 and cell zstack to a horizontal stack that can be analyzed late

                                start_Index=i-int((2*num_fovs*num_slices)+ceil(num_slices/2)-(num_fovs*num_slices)-1); #channel 2 start index
                                cell_img=cleaned_img
                                c2_img=crop_using_roi_tuple(roi_coordinates,np.array(images[start_Index])); #intialize channel 1 hstack with the first image

                                for k in range(start_Index+1,start_Index+int(num_slices)): #add zslices horizontally

                                    curr_img=crop_using_roi_tuple(roi_coordinates,np.array(images[k]));
                                    c2_img=np.hstack((c2_img,curr_img))
                                    cell_img=np.hstack((cell_img,cleaned_img))
                                kernel = np.ones((3,3),np.uint8)
                                cell_img = cv2.dilate(cell_img,kernel,iterations = 1) #to connect mother and daughter cells

                                masked_hstack = np.multiply(cell_img,c2_img) #create a cell-masked image of channel 1


                                if np.shape(cell_size_extractor(cleaned_img))[0]==1: #unbudded cells

                                    imwrite(output_path+'/'+channel2+'/'+str(cell_index)+'.tif', c2_img)
                                    imwrite(output_path+'/'+channel2+'_masked'+'/'+str(cell_index)+'.tif',masked_hstack)
                                    cell_index+=1


                                elif np.shape(cell_size_extractor(cleaned_img))[0]==2: #budded cells
                                    imwrite(output_path+'/'+channel2+'/'+str(cell_index)+'.tif', c2_img)
                                    imwrite(output_path+'/'+channel2+'_masked'+'/'+str(cell_index)+'.tif',masked_hstack)
                                    cell_index+=1

                    except Exception as e:
                        print(e)
                    maskindex+=1;



#Initialize the App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec()
