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
        
