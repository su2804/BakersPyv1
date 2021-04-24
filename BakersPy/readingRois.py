#Read roi files from ImageJ
#Extracts coordinates for rectangular ROIs from .zip IJ files
from read_roi import read_roi_file
from read_roi import read_roi_zip
roi = read_roi_zip('/Users/Saransh/Desktop/formichaelray/formichael/RoiSet.zip')
#print(roi.items())
a=list(roi.items())
print(a[0][1])

print(a[0][1]['left'])
