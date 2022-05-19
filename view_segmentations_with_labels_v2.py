
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
from skimage.measure import label, regionprops, regionprops_table
import math
import pickle

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

        #Line edits
        self.edit_mask = self.findChild(QLineEdit,"mask_path")
        self.edit_roi = self.findChild(QLineEdit,"roi_path")
        self.edit_nd2 = self.findChild(QLineEdit,"nd2_path")
        self.edit_output_path = self.findChild(QLineEdit,"outputpath")
        self.edit_mean1 = self.findChild(QLineEdit,"mean_1")
        self.edit_mean2 = self.findChild(QLineEdit,"mean_2")
        self.edit_std1 = self.findChild(QLineEdit,"std_1")
        self.edit_std2 = self.findChild(QLineEdit,"std_2")

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

        #extract mu and sigma for noising
        mu1 = float(self.edit_mean1.text())
        sigma1 = float(self.edit_std1.text())

        mu2 = float(self.edit_mean2.text())
        sigma2 = float(self.edit_std2.text())


        #create subfolders to upload Results if not already created
        #try:
        os.mkdir(output_path+'/'+channel1) #organelle 1
        os.mkdir(output_path+'/'+channel2) #organelle 2
        os.mkdir(output_path+'/'+channel3) #cells
        os.mkdir(output_path+'/'+channel1+'_masked')
        os.mkdir(output_path+'/'+channel2+'_masked')
        os.mkdir(output_path+'/'+channel1+'_noised')
        os.mkdir(output_path+'/'+channel2+'_noised')
        #except Exception as e:
            #print(e)

        cell_index = 1; #initialize cell index

        fov_start_index = int((2*num_fovs*num_slices)+ceil(num_slices/2))
        fov_end_index = int((3*num_fovs*num_slices)-(num_slices-ceil(num_slices/2)))
        fov_increment = int(num_slices)

        with open(output_path+'cell_size_state.csv', 'w') as csvfile:
            fieldnames = ['major_d','minor_d','major_m','minor_m','state'] #state describes wheter the cell is in a budded state (2) or an unbudded state (1)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            with ND2Reader(nd2_path) as images:
                images.iter_axes = 'cvz';
                maskindex = 1; #initialize mask/fov index
                cell_size=[]
                for i in range(fov_start_index , fov_end_index , fov_increment):
                    #iterating through all central z slices/fovs

                    image = images[i];
                    image_array = np.array(image);

                    try:

                        if maskindex<11:
                            roi_tuples = extract_roi_coordinates(roi_path+'/fov'+str(maskindex)+'.zip');
                            maskimage = cv2.imread(mask_path+'/cell_masks000'+str(maskindex-1)+'.tif',cv2.IMREAD_UNCHANGED)
                            cell_phase = cv2.imread('/Users/Saransh/Documents/2022.03.03/Galactose/TIFs_Phase/'+str(maskindex)+'.tif',-1)
                            phase_array = np.array(cell_phase)
                            maskarray = np.array(maskimage)
                            maskarray.astype(int)
                            l = len(roi_tuples);


                            #iterate through each roi in the fov
                            for j in range(0,l):

                                #########

                                #extract roi's
                                maskimage = cv2.imread(mask_path+'/cell_masks000'+str(maskindex-1)+'.tif',cv2.IMREAD_UNCHANGED)
                                cell_phase = cv2.imread('/Users/Saransh/Documents/2022.03.03/Galactose/TIFs_Phase/'+str(maskindex)+'.tif',-1)
                                phase_array = np.array(cell_phase)
                                maskarray = np.array(maskimage)
                                maskarray.astype(int)
                                roi_coordinates = roi_tuples[j][0]; #get roi coordinates
                                cropped_image = crop_using_roi_tuple(roi_coordinates,maskarray) # crop the roi from the fov cell mask
                                #pre_cleaned_img = pre_clean(np.array(cropped_image)) #binarize and label connected-regions before cleaning borders
                                cleaned_img = clean_roi_borders(cropped_image) #clean the borders of the ROI image so that only one cell remains in the centre



                                #cleaned_img[cleaned_img>0] = 1; #binarize the cleaned image
                                #cleaned_img = cleaned_img.astype('uint8') #set the datatype of the numpy array to uint8 for further operations


                                cropped_cell_img = crop_using_roi_tuple(roi_coordinates,phase_array)
                                #cropped_cell_img = cropped_cell_img.astype(np.uint8)





                                fig, (ax1,ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
                                regions = regionprops(cleaned_img)
                                ax1.imshow(cropped_cell_img,cmap='gray')
                                #ax1.imshow(cropped_cell_img,vmin=np.amin(cropped_cell_img), vmax=np.amax(cropped_cell_img))########

                                ax2.imshow(cleaned_img) #########
                                current_size=[]
                                for props in regions:
                                    y0, x0 = props.centroid
                                    orientation = props.orientation
                                    x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
                                    y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
                                    x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
                                    y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

                                    ax2.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                                    ax2.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                                    ax2.plot(x0, y0, '.g', markersize=15)

                                    minr, minc, maxr, maxc = props.bbox
                                    bx = (minc, maxc, maxc, minc, minc)
                                    by = (minr, minr, maxr, maxr, minr)
                                    ax2.plot(bx, by, '-b', linewidth=2.5)
                                    current_size.append([props.major_axis_length,props.minor_axis_length])
                                    ax2.set_title(str(np.amax(current_size)))



                                if np.shape(cell_size_extractor(cleaned_img))[0]==1: #unbudded cells
                                    writer.writerow({'major_m':np.amax(current_size)})
                                    #writer.writerow({'major_m':cell_size_extractor(cleaned_img)[0][0]*0.09,'minor_m':cell_size_extractor(cleaned_img)[0][1]*0.09,'state':1}) #cell volume added to the csv file
                                    plt.savefig(output_path+'/'+channel3+'/'+str(cell_index)+'_cell.tif')
                                    imwrite(output_path+'/'+channel3+'/'+str(cell_index)+'_cleaned_cell.tif',cropped_cell_img,photometric='minisblack')
                                    cell_size.append(current_size)
                                    cell_index+=1

                                elif np.shape(cell_size_extractor(cleaned_img))[0]==2: #budded cell
                                    writer.writerow({'major_m':np.amax(current_size)})
                                    #writer.writerow({'major_m':cell_size_extractor(cleaned_img)[0][0]*0.09,'minor_m':cell_size_extractor(cleaned_img)[0][1]*0.09,'major_d':cell_size_extractor(cleaned_img)[1][0]*0.09,'minor_d':cell_size_extractor(cleaned_img)[1][1]*0.09,'state':2}) #cell volume added to the csv file
                                    plt.savefig(output_path+'/'+channel3+'/'+str(cell_index)+'_cell.tif')
                                    imwrite(output_path+'/'+channel3+'/'+str(cell_index)+'_cleaned_cell.tif',cropped_cell_img,photometric='minisblack')
                                    cell_size.append(current_size)
                                    cell_index+=1




                        elif maskindex>10 & maskindex<101:
                            roi_tuples = extract_roi_coordinates(roi_path+'/fov'+str(maskindex)+'.zip');
                            maskimage = cv2.imread(mask_path+'/cell_masks00'+str(maskindex-1)+'.tif',cv2.IMREAD_UNCHANGED)
                            cell_phase =  cv2.imread('/Users/Saransh/Documents/2022.03.03/Galactose/TIFs_Phase/'+str(maskindex)+'.tif',-1)
                            phase_array = np.array(cell_phase)
                            maskarray = np.array(maskimage)
                            maskarray.astype(int)
                            l = len(roi_tuples);



                            #iterate through each roi in the fov
                            for j in range(0,l):
                                maskimage = cv2.imread(mask_path+'/cell_masks00'+str(maskindex-1)+'.tif',cv2.IMREAD_UNCHANGED)
                                cell_phase =  cv2.imread('/Users/Saransh/Documents/2022.03.03/Galactose/TIFs_Phase/'+str(maskindex)+'.tif',-1)
                                phase_array = np.array(cell_phase)
                                maskarray = np.array(maskimage)
                                maskarray.astype(int)
                                roi_coordinates = roi_tuples[j][0]; #get roi coordinates
                                cropped_image = crop_using_roi_tuple(roi_coordinates,maskarray) # crop the roi from the fov cell mask
                                #pre_cleaned_img = pre_clean(np.array(cropped_image)) #binarize and label connected-regions before cleaning borders
                                cleaned_img = clean_roi_borders(cropped_image) #clean the borders of the ROI image so that only one cell remains in the centre
                                #cleaned_img = (cropped_image)


                                #cleaned_img[cleaned_img>0] = 1; #binarize the cleaned image
                                #cleaned_img = cleaned_img.astype('uint8') #set the datatype of the numpy array to uint8 for further operations


                                cropped_cell_img = crop_using_roi_tuple(roi_coordinates,phase_array)
                                #cropped_cell_img = cropped_cell_img.astype(np.uint8)






                                fig, (ax1,ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
                                regions = regionprops(cleaned_img)
                                ax1.imshow(cropped_cell_img,cmap='gray')
                                #ax1.imshow(cropped_cell_img,vmin=np.amin(cropped_cell_img), vmax=np.amax(cropped_cell_img))########
                                ax2.imshow(cleaned_img) #########
                                current_size=[]
                                for props in regions:
                                    y0, x0 = props.centroid
                                    orientation = props.orientation
                                    x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
                                    y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
                                    x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
                                    y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

                                    ax2.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                                    ax2.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                                    ax2.plot(x0, y0, '.g', markersize=15)

                                    minr, minc, maxr, maxc = props.bbox
                                    bx = (minc, maxc, maxc, minc, minc)
                                    by = (minr, minr, maxr, maxr, minr)
                                    ax2.plot(bx, by, '-b', linewidth=2.5)
                                    current_size.append([props.major_axis_length,props.minor_axis_length])
                                    ax2.set_title(str(np.amax(current_size)))




                                if np.shape(cell_size_extractor(cleaned_img))[0]==1: #unbudded cells
                                    writer.writerow({'major_m':np.amax(current_size)})
                                    #writer.writerow({'major_m':cell_size_extractor(cleaned_img)[0][0]*0.09,'minor_m':cell_size_extractor(cleaned_img)[0][1]*0.09,'state':1}) #cell volume added to the csv file
                                    plt.savefig(output_path+'/'+channel3+'/'+str(cell_index)+'_cell.tif')
                                    imwrite(output_path+'/'+channel3+'/'+str(cell_index)+'_cleaned_cell.tif',cropped_cell_img,photometric='minisblack')
                                    cell_size.append(current_size)
                                    cell_index+=1

                                elif np.shape(cell_size_extractor(cleaned_img))[0]==2: #budded cell
                                    writer.writerow({'major_m':np.amax(current_size)})
                                    #writer.writerow({'major_m':cell_size_extractor(cleaned_img)[0][0]*0.09,'minor_m':cell_size_extractor(cleaned_img)[0][1]*0.09,'major_d':cell_size_extractor(cleaned_img)[1][0]*0.09,'minor_d':cell_size_extractor(cleaned_img)[1][1]*0.09,'state':2}) #cell volume added to the csv file
                                    plt.savefig(output_path+'/'+channel3+'/'+str(cell_index)+'_cell.tif')
                                    imwrite(output_path+'/'+channel3+'/'+str(cell_index)+'_cleaned_cell.tif',cropped_cell_img,photometric='minisblack')
                                    cell_size.append(current_size)
                                    cell_index+=1


                        f = open('/Users/Saransh/Documents/2022.03.03/size_gal.pckl', 'wb')
                        pickle.dump(cell_size, f)
                        f.close()

                    except Exception as e:
                        print(e)
                    maskindex+=1;




#Initialize the App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec()
