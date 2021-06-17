import essentia.standard as es
import numpy as np
from scipy import signal
import madmom
from scipy.signal import find_peaks


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
    sodf = madmom.features.onsets.SpectralOnsetProcessor('superflux', diff_frames=20)
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