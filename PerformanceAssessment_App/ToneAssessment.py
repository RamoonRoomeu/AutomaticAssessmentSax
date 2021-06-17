import essentia.standard as es
from scipy import signal
from scipy.signal import find_peaks
from madmom.features.onsets import SpectralOnsetProcessor
import numpy as np
import essentia.standard as es
import pickle
from pickle import load
import os

#Essentia energy computation
def NRG_computation(x):
    '''
    Calculates the energy of an input audio signal
    :param x: audio signal
    :return: the energy array and the hopsize used
    '''
    windowSize = 1024*2
    hopSize = 512*2
    NRG = [];
    for frame in es.FrameGenerator(x, frameSize = windowSize, hopSize = hopSize, startFromZero = True):
        NRG.append(es.Energy()(frame))
    NRG = np.array(NRG)
    NRG = NRG / np.max(NRG)
    return NRG, hopSize

def attack_detection(x, M):
    '''
    Detects the start and end of the attack of a note. In order to use it to calculate the release, just flip the imput audio
    :param x: the audio array
    :param M: threashold value. M=3 for attack and M=1.5 for release
    :return: the start and end of the attack in samples location
    '''
    NRG, hopSize = NRG_computation(x)
    #Upscale the NRG computation
    up_scale = 10
    up_NRG = signal.resample(NRG, len(NRG)*up_scale)

    
    #Find the indexes that match the threshold
    first_min = min(up_NRG[0:int(len(up_NRG)*0.15)])
    th = np.arange(first_min, 1, 0.05)
    th_index = np.zeros_like(th)
    th_found = np.zeros_like(th)
    j=0
    for i in range(int(len(up_NRG)/2)) :
        if j<len(th):
            if abs(up_NRG[i]-th[j])<0.025  and th_found[j] == 0:
                th_index[j] = i
                th_found[j] = 1
                j = j+1
    
    #Find each value of t
    t = []
    for i in range(len(th_index)):
        if th_index[i] != 0:
            t.append(int(th_index[i]))


    start_found = False
    end_found = False
    #Calculate w
    if len(t)>1:
        w = np.zeros(len(t)-1)
        for i in range(len(t)-1):
            w[i] = t[i+1]-t[i]

        #Calculate w_av*M
        w_av = np.mean(w)
        th_val = w_av*M
        # Loop to find the start and end values
        for i in range(len(w)):
            if w[i]<th_val and start_found == False:
                att_start = t[i]
                start_found = True
            if w[i]>th_val and end_found == False and start_found == True:
                att_end = t[i]
                end_found = True
    if start_found == False:
        att_start = int(len(up_NRG)*0.01)
    if end_found == False:
        att_end = att_start+int(len(up_NRG)*0.05)

    att_start_sample = int(att_start/up_scale)*hopSize
    att_end_sample = int(att_end/up_scale)*hopSize

    return att_start_sample, att_end_sample


# Onset detection using Spectral Onset Processor
def onset_SOP(audiofile):
    '''
    Calculates the onsets of each note of an audiofile
    :param audiofile: The audiofile you want to ewxtract the onsets from
    :return: The onset array in seconds
    '''
    # Detection function
    fps = 200
    sodf = SpectralOnsetProcessor('superflux', diff_frames=20)
    sodf.processors[-1]  # doctest: +ELLIPSIS
    det_function = sodf(audiofile, fps=fps)
    det_function_norm = det_function / (max(det_function))

    # Dynamic threashold
    C_t = 0.99
    H = 100
    delta = 0.1

    din_th = np.zeros(len(det_function_norm))
    for m in range(H, len(det_function_norm)):
        din_th[m] = C_t * np.median(det_function_norm[m - H:m + H]) + delta

    # Peak detection
    peaks, _ = find_peaks(det_function_norm, distance=fps / 10, height=din_th)
    onset_array = peaks / fps

    return onset_array

def sound_island(x):
    '''
    Detects if there is sound/energy in the audio signal
    :param x: the audui signal
    :return: the split decision function, 0 if there is no enery, 1 if there is energy
    '''
    NRG, hopSize = NRG_computation(x)
    #Applying energy threshold to decide wave split boundaries
    split_decision_func = np.zeros_like(NRG)
    split_decision_func[NRG > 0.005] = 1
    #Setting segment boundaries
    #Inserting a zero at the beginning since we will decide the transitions using a diff function
    split_decision_func = np.insert(split_decision_func, 0, 0)
    diff_split_decision = np.diff(split_decision_func)
    #Start indexes: transition from 0 to 1
    start_indexes = np.nonzero(diff_split_decision > 0)[0] * hopSize
    #Stop indexes: transition from 1 to 0
    stop_indexes = np.nonzero(diff_split_decision < 0)[0] * hopSize
    return (split_decision_func, start_indexes, stop_indexes)

# --------Code for the assessment-------------------------------------------------------------

#Create a full directory path for each file of the input data
def get_file_paths(path):
    sound_files = os.listdir(path)
    segments_dir = []
    for i in range(len(sound_files)):
        filepath = os.path.join(path, sound_files[i])
        segments_dir.append(filepath)
    return segments_dir

#Load txt file
def load_txt(path, filename, n_delete_end):
    path = os.path.join(path, filename)
    file = open(path+'.txt', "r")
    content = file.read()
    content = content.split(",")
    file.close()
    content = content[:n_delete_end]
    return content

#Extract the features for the audio
def get_features(audiofile, scaler, start, stop):
    features, features_frames = es.MusicExtractor(lowlevelSilentFrames='drop',
                                                              lowlevelFrameSize = 2048,
                                                              lowlevelHopSize = 1024,
                                                              lowlevelStats = ['mean', 'stdev'],
                                                              startTime = start,
                                                              endTime = stop)(audiofile)
    scalar_lowlevel_descriptors = [descriptor for descriptor in features.descriptorNames() if 'lowlevel' in descriptor and isinstance(features[descriptor], float)]
    selected_features = [features[descriptor] for descriptor in scalar_lowlevel_descriptors]
    X = str(selected_features)[1:-1]
    X = np.array(X.split(','))
    X = X.reshape(1, -1)
    X = X.astype(np.float64)
    X_scaled = scaler.transform(X)
    return X_scaled

#Select the wright features form all the extracted ones
def select_features(descriptor, feature_names, extracted_features):
    # Obtain the selected features for this descriptor/klass
    selected_features = load_txt(path='selected_features', filename=descriptor, n_delete_end=-1)

    # Find the indexes of the complete feature set that coincide
    feature_idx = []
    for i in range(len(feature_names)):
        for j in range(len(selected_features)):
            if feature_names[i] == selected_features[j]:
                feature_idx.append(i)

    # Save only the features that match de previously extracted indexes
    np_features = np.transpose(np.array(extracted_features))
    sel_features_array = np.transpose(np_features[feature_idx])

    return sel_features_array


# Calculate a score from the predictions of each note
def compute_score(predictions):
    predictions_sig = logit(predictions)
    scores = np.mean(predictions_sig, axis=0)

    return scores


def logit(x):
    y = np.log(x / (1 - x + 4e-5) + 1e-4) * 0.05 + 0.5
    return y


def AssessTone(audiofile):

    #Load the audio
    fs = 44100
    x = es.MonoLoader(filename=audiofile)()

    # Obtain all the feature names
    feature_names = load_txt(path='extracted_features', filename='feature_names', n_delete_end=-7)

    # load the scalers from the trainig data
    Goodsound_scaler = load(open('scalers/Good-sound_scaler.pkl', 'rb'))
    Badattack_scaler = load(open('scalers/Bad-attack_scaler.pkl', 'rb'))

    # Get the model paths
    models_dir = 'models/selected_features_SVM'
    descriptors = ['Good-sound', 'Bad-attack', 'Bad-dynamics', 'Bad-pich', 'Bad-timber', 'Bad-richness']
    model_paths = get_file_paths(models_dir)

    print("Start NRG computation")
    # Compute de decision function for the enregy threashold
    split_decision_func, NRG_start_indexes, NRG_stop_indexes = sound_island(x)

    # Upsample it to the same mumber of samples as the audio
    is_there_NRG = np.around(signal.resample(split_decision_func, len(x)))
    print("NRG computed")
    # Extract samples where the onsets happen
    print("Start Onsets Computation")
    onset_array = onset_SOP(audiofile) * fs
    onset_array = np.append(onset_array, NRG_stop_indexes[-1])
    print("Onsets computed")
    # Separate each note to a different file
    note_count = 0
    onset_shift = 2700

    # Generate the predictions array
    seg_predictions = np.zeros((1, len(descriptors)))
    predictions = []
    start_array = []
    stop_array = []
    attack_start_array = []
    attack_end_array = []
    release_start_array = []
    release_end_array = []
    first_prediction = True
    # For each segent, extract features and apply each one of the models
    for i in range(len(onset_array) - 1):
        start = int(onset_array[i]) - onset_shift
        if start < 0:
            start = 0
        stop = int(onset_array[i + 1]) - onset_shift

        # only use segments that have enregy (no between silences)
        if np.mean(is_there_NRG[start:stop]) > 0.35:
            # only keep segments larger than 1/4 second
            if stop - start > fs / 4:
                # Segment the whole audio between the onsets
                x_seg = x[start:stop]
                start_array.append(start)
                stop_array.append(stop)

                # Attack detection
                attack_start, attack_end = attack_detection(x_seg, M=5)

                # Release detection (attack but flipped)
                x_flip = np.array(np.flip(x_seg))
                release_start_flip, release_end_flip = attack_detection(x_flip, M=1.5)
                release_start = len(x_seg) - release_end_flip
                release_end = len(x_seg) - release_start_flip

                # Condition if the attack is too short
                if attack_end - attack_start < fs / 10:
                    attack_end = attack_start + int(len(x_seg) * 0.05)

                # Check that the attack end comes after the release start and is long enough for evaluation
                attack_release_diff = release_start - attack_end
                if attack_release_diff > fs / 8:

                    # Calculate the exact sample for the whole audio, not only the segment
                    attack_start_array.append(attack_start + onset_array[i])
                    attack_end_array.append(attack_end + onset_array[i])
                    release_start_array.append(release_start + onset_array[i])
                    release_end_array.append(release_end + onset_array[i])

                    # If we want to extract the features from the attack or not, select the right segments and scaler
                    startTime_att = (attack_start + onset_array[i]) / fs
                    endTime_att = (attack_end + onset_array[i]) / fs
                    scaler_att = Badattack_scaler

                    startTime = (attack_end + onset_array[i]) / fs
                    endTime = (release_start + onset_array[i]) / fs
                    scaler = Goodsound_scaler

                    # Extract the features
                    extracted_features = get_features(audiofile, scaler, startTime, endTime)
                    extracted_features_att = get_features(audiofile, scaler_att, startTime_att, endTime_att)

                    for j in range(len(descriptors)):
                        if descriptors[j] == 'Bad-attack':
                            selected_features = select_features(descriptors[j], feature_names, extracted_features_att)
                        else:
                            selected_features = select_features(descriptors[j], feature_names, extracted_features)
                        # Obtain model name
                        model_name = os.path.join(models_dir, descriptors[j])
                        loaded_model = pickle.load(open(model_name, 'rb'))
                        seg_predictions[0, j] = loaded_model.predict_proba(selected_features)[:, 0]
                    if first_prediction == True:
                        predictions = seg_predictions
                        first_prediction = False
                    else:
                        predictions = np.vstack((predictions, seg_predictions))

                    note_count = note_count + 1

    scores = compute_score(predictions)
    #print(scores)
    return scores