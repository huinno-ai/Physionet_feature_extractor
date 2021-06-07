#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.
from feature.Feature_Extractor import Feature_Extractor
from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
twelve_lead_model_filename = '12_lead_model.sav'
six_lead_model_filename = '6_lead_model.sav'
three_lead_model_filename = '3_lead_model.sav'
two_lead_model_filename = '2_lead_model.sav'

################################################################################
#
# Training function
#
################################################################################
classmap = pd.read_csv('class_label.csv',sep='\t')
# classmap.set_index('Abbreviation')
classdict = classmap[['Abbreviation','SNOMED CT Code']]
classdict = classdict.set_index("SNOMED CT Code").T
classdict = classdict.to_dict(orient='list')

def labelToStr(arr):
    stra =''
    for ar in arr:

        # print(ar , classdict[int(ar)])
        if ar == '':
            continue
        if int(ar) in classdict:
            stra = stra + str(classdict[int(ar)]) +' , '
        else:
            stra = stra + str(ar) + ' , '
    stra = stra[0:-2]
    return stra


def summer_history(np_summer,np_return,np_counter):
    for i in range (np_summer.shape[0]):
        if isinstance(np_return,int) == True:
            return np_summer,np_counter
        if np.isnan(np_return[i]) == True:
            # np_summer[i] = np_summer[i] + np_return[i]
            pass
        else:
            np_summer[i] = np_summer[i] + np_return[i]
            np_counter[i] += 1
    return np_summer,np_counter

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')

    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
       # print(classes)
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)
    print(num_classes)
    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    data = np.zeros((num_recordings, 14), dtype=np.float32) # 14 features: one feature for each lead, one feature for age, and one feature for sex
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes
    counter3 = 0
    counter2 = 0
    counter4=0
    list_labelvalid = []
    ij = np.zeros((27,27),dtype=int)
    counter = np.zeros(27,dtype=int)
    # ij = [[0 for x in range(num_classes)] for y in range(num_classes)]
    print(num_recordings)
    print(recording_files[180])
    np_counter = np.zeros(256)
    np_summer =  np.zeros(256)

    noreturn = 0
    for i in range(0,num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        # print(recording_files)

        # Get age, sex and root mean square of the leads.
        age, sex, rms ,rate = get_features(header, recording, twelve_leads)
        current_labels = get_labels(header)

        # 1.rate 500 의 기준 feature 만 extraction
        # 2. 길이가 10초이상의 기준 recoding 만 extraction, 단 5000이넘는경우 0~5000 기준으로 feature 생성
        if rate == 500 and len(recording[1]) >= 5000:
            strlabel = labelToStr(current_labels)
            np_return = Feature_Extractor(recording[:,:5000],rate,strlabel,age,sex,False,header_files[i])
            if np_return is None or np_return is 0:
                noreturn += 1
                print( 'Feature Extraction Failed and failed count(cannot extract enough R peaks) : ',noreturn)
                pass
            else :
                np_summer,np_counter = summer_history(np_summer,np_return,np_counter)

            idx = [0,1,5,6,11]

            fs = 500

        flag = 0
    # feature 의 평균 값을 계산하기위해 모든 data 의 feature 의 합 / count 를 계산
    # todo : class 별로 평균값을 구하는 부분 추가해야 함
    np.savetxt('feat_summer.txt',np_summer)
    np.savetxt('feat_counter.txt',np_counter)

    # Define parameters for random forest classifier.
    n_estimators = 3     # Number of trees in the forest.
    max_leaf_nodes = 100 # Maximum number of leaf nodes in each tree.
    random_state = 0     # Random state; set for reproducibility.

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
    joblib.dump(d, filename, protocol=0)

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    return load_model(filename)

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    return load_model(filename)

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    return load_model(filename)

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    return load_model(filename)

# Generic function for loading a model.
def load_model(filename):
    return joblib.load(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording):
    classes = model['classes']
    leads = model['leads']
    imputer = model['imputer']
    classifier = model['classifier']

    # Load features.
    num_leads = len(leads)
    data = np.zeros(num_leads+2, dtype=np.float32)
    age, sex, rms = get_features(header, recording, leads)
    data[0:num_leads] = rms
    data[num_leads] = age
    data[num_leads+1] = sex

    # Impute missing data.
    features = data.reshape(1, -1)
    features = imputer.transform(features)

    # Predict labels and probabilities.
    labels = classifier.predict(features)
    labels = np.asarray(labels, dtype=np.int)[0]

    probabilities = classifier.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    return classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads, dtype=np.float32)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))
    rate = get_rate(header)
    return age, sex, rms , rate
