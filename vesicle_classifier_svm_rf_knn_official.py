#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:25:22 2018

@author: Maria Theiss
"""

import sys 
import numpy as np
import pandas as pd
import copy
import csv
import string
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC                                      
from sklearn.neighbors import KNeighborsClassifier


def main():
    filenames = []  # List with filenames 

###############################################################################    
#
#    # Filenames and output-csv name can be added here, 
#    # if input from command-line is not available
#    filenames = [
#                'example/path/to/filename1.csv',
#                'example/path/to/anotherfile.csv'
#                ]
#    csv_name = "classifier_results.csv"  
#

###############################################################################    
 
    # if-condition applies if argvs are not entered within this script. 
    # They can thus be entered from the command-line
    if len(filenames) == 0:        
        csv_name, filenames = fromCommandLine(filenames)
    
    assert len(filenames) > 0, 'Please add files to process'
    
    # Process input
    x_train_std_list, x_test_std_list, y_train_list, y_test_list, names,  standardization_parameters, svm_parameters = prepareData(filenames)
    
    
    svm_output_list, knn_output_list, forest_output_list, majority_output_list = classifiers(x_train_std_list, 
                                                                                             x_test_std_list, 
                                                                                             y_train_list, 
                                                                                             y_test_list)

    if csv_name != "null":
        createCsv(csv_name, svm_output_list, forest_output_list, 
                  knn_output_list, majority_output_list, standardization_parameters, svm_parameters, names)

    return svm_output_list, knn_output_list, forest_output_list, majority_output_list, names


def fromCommandLine(filenames):
    py3 = sys.version_info[0] > 2   # Check python version to select apropriate code blocks
    csv_name = sys.argv[1] # name of output csv-file

    # add .csv if not existing. Necessary for Windows
    if  csv_name != "null" and csv_name[-4:] != ".csv":
        csv_name = csv_name + ".csv"

    # Alert user when files are overwritten
    file_exists = glob.glob(csv_name)
    
    while len(file_exists) > 0 and csv_name != "null":  # Enter loop when filename exists and is not named "null"
        
        if py3:
            response = input("Output csv-file already exists. Do you want to overwrite? y: continue. n: rename. ")
        else: 
            response = raw_input("Output csv-file already exists. Do you want to overwrite? y: continue. n: rename. ")
            
        if response == "y":
            break
        
        if response == "n":
            
            if py3:
                csv_name = input("Please enter a new filename. ")
            else:
                csv_name = raw_input("Please enter a new filename. ")

            if csv_name == "null":
                break    
            
            if  csv_name[-4:] != ".csv":       
                csv_name = csv_name + ".csv"               
                        
            file_exists = glob.glob(csv_name)   # Check again if file exists
            
        else: 
            print("Invalid input.")
        
    # Raise KeyError if invalid character (like "/" indicating namespaces) are found in first argv.    
    invalidChars = set(string.punctuation.replace("_", "")) # set "_" as valid character
    invalidChars = set(string.punctuation.replace(".", "")) # set "." as valid character
    
    if any(char in invalidChars for char in csv_name):
        raise KeyError("""First argv contains at least one invalid character.
                         The first argv must contain the name of the output csv-file 
                         or 'null' if no csv-output is desired.""")
                    
    # create list of filepaths 
    for filename in sys.argv[2: ]:
        filenames.append(filename)
            
    return csv_name, filenames


def prepareData(filenames):  
    """Input file-paths. Control and format data. Call functions to shuffle data,
    split data into trainings- and testdata, standardize data. 
    """
    print("Preprocessing data... ")
    
    features_per_tomogram = []  # list with tomogram-wise feature-arrays 
    labels_per_tomogram = []    # list with tomogram-wise label-arrays
    
    names = copy.deepcopy(filenames)   # shortened filenames are saved in list "names"

      
    for filename in filenames: 
        # Shorten filenames if possible
        start = filename.rfind("/")
        stop = filename.rfind(".csv")
        
        if (start != -1 and stop != -1):
            names[filenames.index(filename)] = filename[start+1:stop]   
    
        # read in csv
        temp = pd.read_csv(filename)  
     
        #assert np.shape(temp) == (len(temp), 5), "Data does not have 5 columns."
        assert temp.iloc[:, 4].dtype == 'O', "Column 5 must contain labels."
        assert all(temp.iloc[:, 4] != 'D '), "Delete whitespaces from labelvector."

        # Offset gv if darkest vesicle is brighter than 120
        if int(np.min(temp.iloc[:, 1])) > 120:
            gv_offset = np.min(temp.iloc[:, 1]) - 120
            temp.iloc[:, 1] = temp.iloc[:, 1] - gv_offset
            print("gv [8 bit] of ", names[filenames.index(filename)], 
                  " is offset by: ", np.around(-gv_offset, 1))    
    
        temp = temp[temp.iloc[:, 4] != 'E']      # Exclude data with label "E" (= Error) 
        label = temp.iloc[:, 4].values           # Set column 5 as label 
        label = np.where(label == 'D', -1, 1)    # Set DCV-label as -1, others as 1 
        features = temp.iloc[:, :4].values       # Set columns 1 - 4 as features 
        
        features_per_tomogram.append(features)   # Save feature-arrays in a list with length = n featurearrays
        labels_per_tomogram.append(label)        # Labelarrays likewise 

    for i in range(0, len(filenames)):
        assert len(labels_per_tomogram[i]) == len(features_per_tomogram[i]), "n labels != n samples."  
        
    assert len(features_per_tomogram) == len(filenames), "n files != n filenames"
    assert len(labels_per_tomogram) == len(filenames)
    
    width = features_per_tomogram[0].shape[1]    # Save width of features_per_tomogram (= n features)
    
    standardization_parameters, svm_parameters = getSvmParameters(features_per_tomogram, 
                                                               labels_per_tomogram, width)
    
    x_train_std_list, x_test_std_list, y_train_list, y_test_list = trainTestCombinations(features_per_tomogram, 
                                                                                         labels_per_tomogram, 
                                                                                         width)

    return x_train_std_list, x_test_std_list, y_train_list, y_test_list, names, standardization_parameters, svm_parameters


def getSvmParameters(features_per_tomogram, labels_per_tomogram, width):
    """ Input: 
    features_per_tomogram: List of all tomograms containing their respective features. 
    labels_per_tomogram: List of all tomograms containing their respective labels. 
    width: Number of features.
    Output: 
    standardization_parameters: numpy array (2, n features) containing mean and std of pooled features.
    svm_parameters: numpy array (n weights+intercept, ) containing containing svm weights and intercept 
    of pooled features.
    """

    # create lists containing features or labels of all tomograms 
    all_as_training = []
    all_as_label = []
    
    for f in range(len(features_per_tomogram)):        
        all_as_training = np.empty((0, width))  
        all_as_label = np.empty((0, 0))
        
        for array in features_per_tomogram:
           all_as_training = np.append(all_as_training, array, axis = 0)
        
        for label in labels_per_tomogram: 
            all_as_label = np.append(all_as_label, label)
       
    # calculate mean and std of all tomograms.
    standardization_parameters = np.append([np.mean(all_as_training, axis = 0)], [np.std(all_as_training, axis = 0)], axis = 0) 
    
    # standardize features for svm
    all_as_training, _ = standardization(all_as_training, all_as_training)
    
    # call svm to return weights and intercept. Combine them to one array
    _, _, _, _, _, _, weight_array, intercept = svm(all_as_training, all_as_training, all_as_label, all_as_label)
    svm_parameters = np.append(weight_array[0], intercept[0])
    
    return standardization_parameters, svm_parameters


def trainTestCombinations(features_per_tomogram, labels_per_tomogram, width):   
    """ Input: 
    features_per_tomogram: List of all tomograms containing their respective features. 
    labels_per_tomogram: List of all tomograms containing their respective labels. 
    width: Number of features.
    Create four lists: x_train_std_list is a list containing all test-tomograms as arrays. 
    x_test_std_list is a list containing all training-tomograms as arrays. All features 
    are thereby standardized. List - indices of trainings- and test-arrays are corresponding. 
    Likewise with labels (y_train_list and y_test_list).
    """
    print("Computing leave-one-out cross-validation combinations...")
    
    x_train_std_list = [] # List with feature-arrays for training (standardized)
    x_test_std_list = []  # List with feature-arrays for testing (standardized)
    y_train_list = []     # List with label-arrays for training
    y_test_list = []      # List with label-arrays for testing
    


    # Compute leave one out combinations
    for f in range(len(features_per_tomogram)):        
        x_train_summary = np.empty((0, width))   # Create empty array with same width as features_per_tomogram. 
        y_train_summary = np.empty((0, 0))       # Create empty array for labels
                    
        # copy features_per_tomogram and labels_per_tomogram 
        x_train = copy.deepcopy(features_per_tomogram)
        y_train = copy.deepcopy(labels_per_tomogram)
        
        # exclude file f from trainingsdata. It is used for testing.
        del x_train[f]       
        del y_train[f]
        
        # List with training-tomograms is rewritten as one contigous array
        for array in x_train:
            x_train_summary = np.append(x_train_summary, array, axis = 0)
        
        for labelarray in y_train:
            y_train_summary = np.append(y_train_summary, labelarray)
        
        # Call function for standardization
        x_train_std, x_test_std = standardization(x_train_summary, 
                                                  features_per_tomogram[f])
        
        # Create list of all training-combinations
        x_train_std_list.append(x_train_std)
        x_test_std_list.append(x_test_std)
        y_train_list.append(y_train_summary)
        y_test_list.append(labels_per_tomogram[f])
        
    return x_train_std_list, x_test_std_list, y_train_list, y_test_list



def standardization(x_train, x_test): 
    """Standardize data.
    """ 
    sc = StandardScaler()
    
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    
    return x_train_std, x_test_std


def classifiers(x_train_std_list, x_test_std_list, y_train_list, y_test_list):
    """Call all classification-functions.
    """
    print("Calling classifiers...")
    svm_output_list = []
    knn_output_list = []
    forest_output_list = []
    majority_output_list = []
    


    for i in range(len(x_train_std_list)):
        # svm
        svm_accuracy_list, svm_misclassified_list, svm_dc_precision, svm_dc_recall, svm_dc_f_score, svm_y_pred, _, _ = svm(x_train_std_list[i], 
                                                                                                                  x_test_std_list[i], 
                                                                                                                  y_train_list[i], 
                                                                                                                  y_test_list[i])            
        svm_output = [svm_accuracy_list, svm_misclassified_list, 
                      svm_dc_precision, svm_dc_recall, svm_dc_f_score]


        # knn
        knn_accuracy_list, knn_misclassified_list, knn_dc_precision, knn_dc_recall, knn_dc_f_score, knn_y_pred = knn(x_train_std_list[i], 
                                                                                                                  x_test_std_list[i], 
                                                                                                                  y_train_list[i], 
                                                                                                                  y_test_list[i])                 
        knn_output = [knn_accuracy_list, knn_misclassified_list, 
                      knn_dc_precision, knn_dc_recall, knn_dc_f_score]
    
        # Random forest
        forest_accuracy_list, forest_misclassified_list, forest_dc_precision, forest_dc_recall, forest_dc_f_score, forest_y_pred = randomForest(x_train_std_list[i], 
                                                                                                                                             x_test_std_list[i], 
                                                                                                                                             y_train_list[i],
                                                                                                                                             y_test_list[i])         
        forest_output = [forest_accuracy_list, forest_misclassified_list, 
                         forest_dc_precision, forest_dc_recall, forest_dc_f_score]
        
        # Majority prediction
        majority_accuracy_list, majority_misclassified_list, majority_dc_precision, majority_dc_recall, majority_dc_f_score, majority_pred = majorityPrediction(svm_y_pred, 
                                                                                                                                                             knn_y_pred, 
                                                                                                                                                             forest_y_pred, 
                                                                                                                                                             y_test_list[i])
        majority_output = [majority_accuracy_list, majority_misclassified_list, 
                           majority_dc_precision, majority_dc_recall, majority_dc_f_score]
        
        # Summerize classifier-results in one list for each classifier.
        svm_output_list.append(svm_output)
        knn_output_list.append(knn_output)
        forest_output_list.append(forest_output)
        majority_output_list.append(majority_output)
        
    return svm_output_list, knn_output_list, forest_output_list, majority_output_list


def svm(x_train_standard, x_test_standard, y_train, y_test):   
    """Apply the svm and call the evaluation function.
    """
    svm = SVC(kernel = "linear", random_state = 0, gamma = 1, C = 1)  
    
    svm.fit(x_train_standard, y_train)
    y_pred = svm.predict(x_test_standard)
 
    accuracy_list, misclassified_list, dc_precision, dc_recall, dc_f_score = evaluateClassifier(y_test, y_pred)
    
    
    weight_array = svm.fit(x_train_standard, y_train).coef_
    intercept = svm.fit(x_train_standard, y_train).intercept_

    return accuracy_list, misclassified_list, dc_precision, dc_recall, dc_f_score, y_pred, weight_array, intercept


def knn(x_train_standard, x_test_standard, y_train, y_test):
    """Apply knn and call the evaluation function.
    """
    knn = KNeighborsClassifier(n_neighbors = 10, p = 2, metric = 'minkowski')
   
    knn.fit(x_train_standard, y_train)
    y_pred = knn.predict(x_test_standard)
   
    accuracy_list, misclassified_list, dc_precision, dc_recall, dc_f_score = evaluateClassifier(y_test, y_pred)

    return accuracy_list, misclassified_list, dc_precision, dc_recall, dc_f_score, y_pred


def randomForest(x_train_standard, x_test_standard, y_train, y_test):
    """Apply randomForest and call the evaluation function.
    """    
    forest = RandomForestClassifier(criterion='entropy', n_estimators = 10, random_state = 0, n_jobs = 2) 
    
    forest.fit(x_train_standard, y_train)
    y_pred = forest.predict(x_test_standard)    

    accuracy_list, misclassified_list, dc_precision, dc_recall, dc_f_score = evaluateClassifier(y_test, y_pred)
    
    return accuracy_list, misclassified_list, dc_precision, dc_recall, dc_f_score, y_pred


def majorityPrediction(svm_y_pred, knn_y_pred, forest_y_pred, y_test):
    """Create a majority-prediction of all three classifiers.
    """ 
    assert len(svm_y_pred) == len(knn_y_pred) == len(forest_y_pred), "predictions not same the length for all classifiers"
   
    majority_pred = np.zeros(shape = (len(svm_y_pred)))
    
    for i in range(len(svm_y_pred)):
        
        if (svm_y_pred[i] + knn_y_pred[i] + forest_y_pred[i]) >= 0:
            majority_pred[i] = 1    # with an even number of classifier bias for CCV         
        else: 
            majority_pred[i] = -1
              
    accuracy_list, misclassified_list, dc_precision, dc_recall, dc_f_score = evaluateClassifier(y_test, majority_pred)
            
    return accuracy_list, misclassified_list, dc_precision, dc_recall, dc_f_score, majority_pred


def evaluateClassifier(y_test, y_pred):
    """Calculate accuracy, number of misclassified samples. 
    Calculates precision, recall and F-score for DCV
    """      
    dcv_tp = 0    # True positive DCV
    dcv_fp = 0    # False positive DCV
    dcv_fn = 0    # False negative DCV 
    
    accuracy = accuracy_score(y_test, y_pred)
    misclassified = (y_test != y_pred).sum()

    # For-loop counts tp, fp and fn DCV
    for value in range(len(y_test)):
        
        if y_pred[value] == -1:
            
           if y_test[value] == -1:
                dcv_tp += 1               
           else:
                dcv_fp += 1               
        else:
            if y_test[value] == -1:
                dcv_fn += 1
                
    # Calculate precision, recall and F-score of DCV
    # DCV precision
    if (dcv_tp + dcv_fp) > 0:    # Avoid dividing by 0
        dc_precision = dcv_tp / (dcv_tp + dcv_fp)    
    else: dc_precision = ("NA")
          
    # DCV recall
    if (dcv_tp + dcv_fn) > 0:
        dc_recall = dcv_tp / (dcv_tp + dcv_fn)        
    else: dc_recall = ("NA")
    
    # DCV F-score
    if (dcv_tp + dcv_fp) > 0 and (dcv_tp + dcv_fn) > 0:
        dc_f_score = 2 * (dc_precision * dc_recall) / (dc_precision + dc_recall)    
    else: dc_f_score = "NA"
    
    return accuracy, misclassified, dc_precision, dc_recall, dc_f_score


def createCsv(csv_name, svm_output_list, forest_output_list, 
              knn_output_list, majority_output_list, standardization_parameters, svm_parameters, names):
    """Create a .csv file with classification results"
    """
    
    print("Writing CSV...")
    
    with open(csv_name, 'w+') as myfile: # open(filename, mode)
        csvwriter = csv.writer(myfile, quoting = csv.QUOTE_ALL)            
        csvwriter.writerow(["Test_Tomogram", "Accuracy", "misclassified", 
                            "DCV_precision", "DCV_recall", "DCF_F-Score"])   
            
         # svm           
        csvwriter.writerow(["svm: gamma = 1, C = 1"])      
        for i in range(len(names)):
            csvwriter.writerow(tomogram for tomogram in ([names[i]] + svm_output_list[i]))
        csvwriter.writerow([""])

        # random forest
        csvwriter.writerow(["random forest:, n trees = 10, entropy"])  
        for i in range(len(names)):
            csvwriter.writerow(tomogram for tomogram in ([names[i]] + forest_output_list[i]))
        csvwriter.writerow([""])
      
        # knn
        csvwriter.writerow(["knn: neighbours = 10, p = 2"])
        for i in range(len(names)):
            csvwriter.writerow(tomogram for tomogram in ([names[i]] + knn_output_list[i]))
        csvwriter.writerow([""])
         
        # majority
        csvwriter.writerow(["majority prediction"])
        for i in range(len(names)):
            csvwriter.writerow(tomogram for tomogram in ([names[i]] + majority_output_list[i]))
        csvwriter.writerow([""])
    
        # svm parameters
        csvwriter.writerow(["", "radius", "gv", "dist", "GVSD", "intercept"])
        csvwriter.writerow(parameter for parameter in (["svm_weights"] + list(svm_parameters)))
        csvwriter.writerow(mean for mean in (["mean"] + list(standardization_parameters[0])))
        csvwriter.writerow(std for std in (["std"] + list(standardization_parameters[1])))

            

if __name__ == '__main__':   
    svm_output_list, knn_output_list, forest_output_list, majority_output_list, names = main() 