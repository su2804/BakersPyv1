from PIL import Image
def crop_using_roi_tuple(coordinates_tuple,image):
    im=Image.open(image);
    im1 = im.crop(coordinates_tuple);
    return im1

    
