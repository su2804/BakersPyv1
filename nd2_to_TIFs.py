#Extract cell channel images from an nd2 file and write it TIFs
import nd2
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


nd2_to_TIFs('/Users/Saransh/Documents/OneDrive - UC San Diego/20211211/20211211_raf+aa+succ_mito_er+mito_cyto_OD0.265_3k2m8k2m001.nd2','/Users/Saransh/Documents/OneDrive - UC San Diego/20211209/TIFs_Raf/')
