macro "D_C_separation"{
// Standardization parameters
meanR		=	10.394; 	// mean radius
sdR			=	3.128;		// sd radius
meanGV		=	129.070;	// mean GV
sdGV		=	4.653;		// sd GV
meanDAZ		=	259.125; 	// mean distAZ
sdDAZ		=	118.528;	// sd distAZ
meanSD		=	5.852;		// mean sd of middle slice
sdSD		=	1.712;		// sd sd of middle slice

// SVM Weights
R_weight	=	-1.687;
GV_weight	=	1.646;
DAZ_weight	=	-0.758;
SD_weight	=	1.214;

// Intercept
theta		=	4.362;

// Empty arrays to store vesicle-wise features 
arrayR		=	newArray();
arrayGV		=	newArray();
arrayDistAZ	=	newArray();
arraySD		=	newArray();

// Create Dialog
Dialog.create("Pictures");
Dialog.addMessage("Please specify filenames");
Dialog.addString("StackSegmented:", "StackSegmented.tif", 35);
Dialog.addString("StackDuplicateScale:", "StackDuplicateScale.tif", 35);

// Standardization parameters
Dialog.addMessage("Standardization parameters");
Dialog.addNumber("mean r",		meanR);		
Dialog.addNumber("sd r",		sdR);
Dialog.addNumber("mean gv",		meanGV);	
Dialog.addNumber("sd gv",		sdGV);
Dialog.addNumber("mean distAZ",		meanDAZ);	
Dialog.addNumber("sd distAZ",		sdDAZ);
Dialog.addNumber("mean 2DSD",		meanSD);
Dialog.addNumber("sd 2DSD",		sdSD);

// SVM weights and intercept
Dialog.addMessage("SVM weights");
Dialog.addNumber("r",			R_weight);
Dialog.addNumber("gv",			GV_weight);
Dialog.addNumber("distAZ",		DAZ_weight);
Dialog.addNumber("2DSD",		SD_weight);
Dialog.addNumber("intercept",		theta);

Dialog.show();

segmented	=	Dialog.getString();
scaled		=	Dialog.getString();

// Standardization parameters. 
meanR		=	Dialog.getNumber();
sdR		=	Dialog.getNumber();
meanGV		=	Dialog.getNumber();
sdGV		=	Dialog.getNumber();
meanDAZ		=	Dialog.getNumber();
sdDAZ		=	Dialog.getNumber();
meanSD		=	Dialog.getNumber();
sdSD		=	Dialog.getNumber();

// Weights
R_weight	=	Dialog.getNumber();
GV_weight	=	Dialog.getNumber();
DAZ_weight	=	Dialog.getNumber();
SD_weight	=	Dialog.getNumber();

// Intercept
theta		=	Dialog.getNumber();


// Save parameters of the scaled stack to transfer them to newly generated images 
selectImage(scaled);
getDimensions(w, h, channels, s, frames); // get w, h, s (= width, height, nSlices) of the whole image
getVoxelSize(width, height, depth, unit); // width, height, depth returns scale of a single voxel

run("3D Manager");
selectImage(segmented);
Ext.Manager3D_AddImage();
Ext.Manager3D_Count(nb);    // Count vesicles

setTool("point");
selectWindow(scaled);
numROIs	=	0;

while(numROIs < 1) {
    title	=	"Wait For Selection";  
    msg		=	"Please mark the center of the active zone in " + scaled + " by clicking. Confirm with OK";
    waitForUser(title, msg);
    selectWindow(scaled);
    
    type	=	selectionType();    // Check type of selection made. Type 10 (= Point or multipoint selection) is required)
    
    if (type == 10){
        roiManager("Add");
        numROIs	= roiManager("count");
    }
    
    if(numROIs != 1) {
		msg	= "One point selection is required.";
		waitForUser(title, msg);
	}    
}

run("Measure");
roiManager("reset")

// Get x, y, z coordinates of point-selection
selectWindow("Results");
azX	=	getResult("X"); 
azY	=	getResult("Y"); 
azZ	=	getResult("Slice");

// print AZ-coordinates
print("AZ coordinates (x, y, z): " + azX + ", " + azY + ", " + azZ);
	
selectWindow("Results");
run("Close");
setTool("rectangle");    // Set tool to rectangle to make accidental selections unlikely

// Duplicate StackDuplicateScale to StackDuplicateBlur, which is blurred
selectImage(scaled);
run("Duplicate...", "title=StackDuplicateBlur.tif duplicate");
run("Gaussian Blur 3D...", "x=2 y=2 z=2");		

// result window settings 
run("Set Measurements...", "area mean standard min stack redirect=None decimal=3");

for(vesicle = 0; vesicle < nb; vesicle++){
	selectWindow(segmented);									// Select StackSegmented to derive object properties from it. 

	Ext.Manager3D_Quantif3D(vesicle, "Mean", color);			// Save color of segmented vesicle
	Ext.Manager3D_Bounding3D(vesicle, x0, x1, y0, y1, z0, z1); 	// Output the 6 limits of vesicle.

	setSlice(z0 + ((z1 - z0) / 2));								// Set slice to center vesicle of interest
	setThreshold(color, color);									// Set threshold such that only vesicle i with grey value "color" is included
	run("Create Selection");									// Convert threshold to selection
	resetThreshold();								
	roiManager("Add");											// add selection to 2D roiManager			
}

selectWindow("StackDuplicateBlur.tif");

for(vesicle = 0; vesicle < nb; vesicle++){
	roiManager("Select", vesicle);
	run("Measure");										
	sd		=	getResult("StdDev", vesicle);			// Save standard-deviation of selection
	arraySD	=	Array.concat(arraySD, sd);				// Make array of standard deviation of each vesicles-selection						
}

selectWindow(segmented);
run("Select None");
setSlice(1);
selectWindow("StackDuplicateBlur.tif");
run("Close");

// Create arrays of R, GV, DistAZ
selectImage(scaled);

for (vesicle = 0; vesicle < nb; vesicle++){
    Ext.Manager3D_Measure3D(vesicle, "DCMean", r);
    Ext.Manager3D_Quantif3D(vesicle, "Mean", gv);
    Ext.Manager3D_Centroid3D(vesicle, cx, cy, cz);	   // Compute vesicle-centroid
    
    distAZ = sqrt((pow((azX-cx), 2))
    			+ (pow((azY-cy), 2)) 
    			+ (pow((azZ-cz), 2))); //pow(base, exponent); sqrt(n) = square root
		    
    arrayR		=	Array.concat(arrayR, r);
    arrayGV		=	Array.concat(arrayGV, gv);
    arrayDistAZ	=	Array.concat(arrayDistAZ, distAZ);
}
//

// Check if darkest vesicle is too bright. If so, darken all vesicles. 
Array.getStatistics(arrayGV, minGV, maxGV, _, _);

if(minGV > 120){
    offset = minGV - 120;
    
    	for (i = 0; i < nb; i++){
	    	arrayGV[i] = arrayGV[i] - offset;    // Darkest vesicle is now 120
	    }
    print("GV offset: -" + offset);
} else print("no min GV offset needed");
//

// Standardize all arrays. Function at the end of macro.
arrayR_sd		=	standardization(arrayR, meanR, sdR);
arrayGV_sd		=	standardization(arrayGV, meanGV, sdGV); // with darkening of gv > 125
arrayDistAZ_sd		=	standardization(arrayDistAZ, meanDAZ, sdDAZ);
arraySD_sd		=	standardization(arraySD, meanSD, sdSD);
//


// Multiply with SVMs weights. Function at the end of macro.
arrayR_weight		=	weighting(arrayR_sd, R_weight);
arrayGV_weight		=	weighting(arrayGV_sd,  GV_weight);
arrayDistAZ_weight	=	weighting(arrayDistAZ_sd, DAZ_weight);
arraySD_weight		=	weighting(arraySD_sd, SD_weight);
//

// Sum weighted features for each vesicle. Add intercept. outputarray = (nb, 0)
sum_per_vesicle = newArray();

for (i = 0; i < nb; i++){
	sum	=	((arrayR_weight[i] + arrayGV_weight[i] + arrayDistAZ_weight[i] + arraySD_weight[i]) + theta);    // theta = intercept
	sum_per_vesicle = Array.concat(sum_per_vesicle, sum);
}

//define variable for dense core vesicle (dcv) and clear core vesicle (ccv) counting
ccv	=	0;
dcv	=	0;

// Create array with labels. CCV = 1; DCV = 0
labelArray = newArray();

for (i = 0; i < nb; i++){
    if (sum_per_vesicle[i] > 0){
    	labelArray = Array.concat(labelArray, 1);
    	ccv += 1;
    }
    else {
    	labelArray = Array.concat(labelArray, 0);
    	dcv += 1;
    }
}
//


newImage("label", "16-bit Black", w, h, s);	//New Image with same Width, Heigth and n Slices as StackSegmented is created
run("Properties...", "channels=1 slices="+s+" frames=1 unit="+unit+" pixel_width="+width+" pixel_height="+height+" voxel_depth="+depth);

// Color vesicles according to labelArray
for (i = 0; i < nb; i++){
    Ext.Manager3D_Select(i);
    Ext.Manager3D_FillStack(labelArray[i] * 21845 + 21845,     // 21845 ~ 2^16/3
    						labelArray[i] * 21845 + 21845, 
    						labelArray[i] * 21845 + 21845);
	}
run("3-3-2 RGB");	//LUT: CCV are now magenta, DCV green		

// Create composite-Image
selectWindow(scaled);
run("Merge Channels...", "c1=label c2=&scaled create keep");   
rename("CompositeLabel");

//print total number of DCV and CCV:
print("Total number of Clear Core Vesicles: " 
	+ ccv + "\nTotal number of Dense Core Vesicles: " + dcv);
	
// Print arrays
print("obj, " + "R_[nm], " + "gv_[8_Bit], " + "dist_[nm], " + "2DSD, " + "label");

for(i = 0; i < nb; i++){
    print((i + 1) + ", " + arrayR[i] + ", " + arrayGV[i] + ", " 
   		 + arrayDistAZ[i] + ", " + arraySD[i] + ", " + labelArray[i]);
}

// Close ROI Manager (2D)
selectWindow("ROI Manager");
run("Close");

// Close ROI Manager (3D)
Ext.Manager3D_Close();

// Close results-window
selectWindow("Results");
run("Close");

function standardization(featurearray, mean, sd){
	temparray = newArray();
	for (i = 0; i < nb; i ++){
		i_sd		=	(featurearray[i]-mean)/sd;
		temparray	=	Array.concat(temparray, i_sd);
	}
	return temparray;
}

function weighting(featurearray, weight){
    weighted_array = newArray();
    for (i = 0; i < nb; i++){
	    weighted		=	(featurearray[i] * weight);
	    weighted_array	=	Array.concat(weighted_array, weighted);
    }
    return weighted_array;
}

}









