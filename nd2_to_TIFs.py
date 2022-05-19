#Extract cell channel images from an nd2 file and write it TIFs
import numpy as np
from pims import ND2_Reader
from tifffile import imsave

def nd2_to_TIFs(file,output_path):
    with ND2_Reader(file) as frames:
        frames.iter_axes = 'm'  # iterates through FOVs
        frames.bundle_axes = 'yx'  # stitching the whole image in x and y dimension
        frames.default_coords['c'] = 2  # setting channel to the cell channel
        frames.default_coords['z'] = 16 #picking the central z slice
        i=1;
        print((frames))
        for frame in frames[:]:
            imsave(output_path+str(i)+'.tif', frame, photometric='minisblack')
            i+=1
            print(i)
    frames.close()


nd2_to_TIFs('/Users/Saransh/Documents/2022.03.03/nd2/20220303_gal_mito_er+mito_cyto_mitovac_OD0.426.nd2','/Users/Saransh/Documents/2022.03.03/Galactose/TIFs_Phase/')
