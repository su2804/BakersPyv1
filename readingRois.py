
#Read roi files from ImageJ
#sys.path.insert(0,"/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/read_roi/_read_roi.py")
from read_roi import read_roi_file
from read_roi import read_roi_zip
roi = read_roi_zip('/Users/Saransh/Desktop/formichaelray/formichael/RoiSet.zip')
#print(roi.items())
a=list(roi.items())
print(a[0][1])

print(a[0][1]['left'])
