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

//works as expected on FiJi 1

function processFile(input, output, file) {
	//uses already segmented er images
	open(input+"/"+file);
	
	
	run("Invert LUT");
	run("Close-");			//fills in gaps between nuclear membrane
	run("Fill Holes");
	run("Remove Outliers...", "radius=10 threshold=50 which=Bright");  // remove non-nuclei regions
	saveAs("Tiff", toString(output)+"/IJ_script_output_nuclei_segmentation/"+file);		
   
	close();

	print("Processing: " + input + File.separator + file);
	print("Saving to: " + output);
}
