/*
 * Cytosol segmentation - Otsu thresholding plus hole filling
 */

#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File suffix", value = ".tif") suffix

// See also Process_Folder.py for a version of this code
// in the Python scripting language.

processFolder(input);

// function to scan folders/subfolders/files to find files with correct suffix


function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, output, list[i]);
	}
}


function processFile(input, output, file) {
	
	open(input+"/"+file);
	
	
	run("Bandpass Filter...", "filter_large=5 filter_small=2 suppress=None tolerance=5 autoscale saturate");
	setAutoThreshold("Otsu");
	setOption("BlackBackground", true);
	run("Convert to Mask");
	saveAs("Tiff", toString(output)+"/"+file);		
   
	close();

	print("Processing: " + input + File.separator + file);
	print("Saving to: " + output);
}
