import os,sys, math
import numpy as np
import xml, xmltodict
import essentia
import essentia.standard as ess
essentia.log.warningActive=False


'''
============================================
    CONVERSION OF XML FILE TO DICTIONARY
============================================

'''

def ConvertXML2Annotation(infilename):

    '''
    Converts the XML file that has the relevant information of the scale exercise. The XML file must be created with MuseScore
    and have proper formatting specific to this exercise. The guidelines for exercise preparation using MuseScore will be provided
    Parameters:
    -------
    infilename: str
        input XML file which has the exercise annotations

    Returns:
    -------
    DATA : dict
        the annotation information (timestamps, key and scale information for each segment of analysis)
    '''

    NUM_MEASURES_PER_PART = 4

    # LOAD XML FILE
    with open(infilename) as fd:
        doc = xmltodict.parse(fd.read())

    DATA = dict()

    # PARSE TITLE INFO
    DATA["title"] = doc['score-partwise']['work']['work-title']

    #PARSE METER INFO
    meter_nom = doc['score-partwise']['part']['measure'][0]['attributes']['time']['beats']
    meter_denom = doc['score-partwise']['part']['measure'][0]['attributes']['time']['beat-type']
    DATA["meter"] = meter_nom + '/' + meter_denom

    # PARSE TEMPO (BPM) INFO
    BPM = doc['score-partwise']['part']['measure'][0]['direction'][0]['direction-type']['metronome']['per-minute']
    DATA["bpm"] = BPM

    BPM = int(BPM)
    # duration of a measure / 4 beats (in seconds)
    measureDur = int(meter_denom) * 60 / BPM
    # number of measures in the exercise
    num_measures = len(doc['score-partwise']['part']['measure'])
    # number of separate parts in the exercise
    num_parts = int(num_measures / NUM_MEASURES_PER_PART)

    # PARSE PARTS INFO
    parts = []
    for i in range(num_parts):
        part = dict()
        part['name'] = 'Part' + str(i + 1)
        ### note that there are 4 empty beats in the beginning of the exercise.
        ### thus, startTime and endTime are computed accordingly (in seconds)
        if i == 0:
            part['startTime'] = measureDur
            scale = doc['score-partwise']['part']['measure'][0]['direction'][1]['direction-type']['words']

        else:
            part['startTime'] = NUM_MEASURES_PER_PART * (i) * measureDur + measureDur
            scale = doc['score-partwise']['part']['measure'][(i) * NUM_MEASURES_PER_PART]['direction'][0]['direction-type']['words']
        part['endTime'] = part[
                              'startTime'] + NUM_MEASURES_PER_PART * measureDur  ## 4 bars of length for each part
        key = scale.split(' ')[0]
        scaletype = scale.split(' ')[1]
        part['scaleType'] = key + ':' + scaletype

        parts.append(part)

    DATA["parts"] =  parts

    return DATA


'''
=====================================
        FEATURE EXTRACTION
=====================================
'''


def FeatureExtraction(Filename, Annotations, PARAMS):
    '''
    INPUT:
    ------
    Filename : (.mp3)
            input audio file
    AnnotationFile : ("exercise_annotations".json)
            Data from the annotation

    OUTPUT:
    ------
    PARTS_DATA: (dict)
            Dictionary that contains the relevant information/data of the audio segments separately.
    '''

    ### ASSIGN Spectral Analysis Parameters

    NUM_BINS = 12  # number of chroma bins per octave, NUM_BINS = number of pitch-classes per octave

    SAMPLE_RATE = PARAMS.fs  # fs = Sample Rate

    # LOAD the audio
    Audio = ess.MonoLoader(filename=Filename, sampleRate=SAMPLE_RATE)()
    Audio = ess.DCRemoval()(Audio)  # preprocessing - DC Offset Removal
    Audio = ess.EqualLoudness()(Audio)  # preprocessing - Equal Loudness Filter

    # WINDOWING PARAMETERS (first converting from msec to number of samples)
    windowSize = round(SAMPLE_RATE * PARAMS.windowSize / 1000)
    windowSize = int(windowSize / 2) * 2  # assuring window size is even
    hopSize = round(SAMPLE_RATE * PARAMS.hopSize / 1000)
    hopSize = int(hopSize / 2) * 2  # assuring hopSize is even

    # CREATE dictionary to store Part (audio segment of analysis) Data
    PARTS_DATA = dict()

    for part in Annotations['parts']:

        # CREATE data structure for each analysis part
        PartData = InitiateDataStructure(part['name'])
        # FETCH relevant information from annotations data (as ground truth)
        PartData['startTime'] = part['startTime']
        PartData['endTime'] = part['endTime']
        PartData['key'] = part['scaleType'].split(':')[0]
        PartData['scaleType'] = part['scaleType'].split(':')[1]

        # SLICE Audio Segments of analysis
        AudioSegment = sliceAudiotoParts(Audio, PartData['endTime'], PartData['startTime'], SAMPLE_RATE)

        HPCP_ARRAY = []
        ### FRAME-BASED Feature Extraction
        for frame in ess.FrameGenerator(AudioSegment, frameSize=windowSize, hopSize=hopSize, startFromZero=True):
            frame = ess.Windowing(size=windowSize, type=PARAMS.windowFunction)(frame)
            # EXTRACT frame-based spectrum
            mX = ess.Spectrum(size=windowSize)(frame)
            mX[mX < np.finfo(float).eps] = np.finfo(float).eps
            # EXTRACT frequency and magnitude information of spectral harmonic peaks
            frequency, magnitude = ess.SpectralPeaks()(mX)

            # ESTIMATE tuning frequency of the audio segment
            f_tuning, f_Cents = ess.TuningFrequency()(frequency, magnitude)
            # COMPUTE Reference frequency of the reference bin in HPCP vectors (w.r.t 'f_tuning')
            f_reference = computeReferenceFrequency(PartData['key'], f_tuning)

            # EXTRACT Harmonic Pitch-Class Profiles (HPCP)
            HPCP_Framebased = ess.HPCP(normalized='unitSum',
                                       referenceFrequency=f_reference,
                                       size=NUM_BINS,
                                       windowSize=12 / NUM_BINS)(frequency, magnitude)

            # Post-Processing - KEEP ONLY the HPCP bin with max value, set other to 0 (zero).
            HPCP_OnlyMax = np.zeros_like(HPCP_Framebased)
            HPCP_OnlyMax[np.argmax(HPCP_Framebased)] = np.max(HPCP_Framebased)
            HPCP_OnlyMax.tolist()
            HPCP_ARRAY.append(HPCP_OnlyMax)

        # Last condition of HPCP_ARRAY[i] < 0.7 is to check if the amplitudes of pitch-classes
        # are distributed more evenly within the chroma vectors (implying ambiguity in pitch perception),
        # FOR detecting & removing noisy regions.
        # !!! NOTE that the threshold value of 0.7 is determined experimentally, not computationally.
        for i in range(1, len(HPCP_ARRAY) - 1):
            if np.argmax(HPCP_ARRAY[i]) != np.argmax(HPCP_ARRAY[i - 1]) and np.argmax(HPCP_ARRAY[i]) != np.argmax(
                    HPCP_ARRAY[i + 1]) or np.max(HPCP_ARRAY[i]) < 0.7:
                HPCP_ARRAY[i] = np.zeros(NUM_BINS)

        PartData['HPCP_Framebased'] = HPCP_ARRAY

        ### SUMMARIZATION of Frame-Based Features over Audio Segments of Analysis
        for j in range(NUM_BINS):
            HPCP_temp = []
            for i in range(len(HPCP_ARRAY)):
                HPCP_temp.append(HPCP_ARRAY[i][j])
            PartData['HPCP_Mean'].append(np.mean(HPCP_temp))
            PartData['HPCP_Std'].append(np.std(HPCP_temp))

        PARTS_DATA[part['name']] = PartData

    return PARTS_DATA

'''
=====================================
     FEATURE EXTRACTION UTILITIES
=====================================
'''

def InitiateDataStructure(FileName):
    '''
    Forming the data structure for file
    INPUT :
    ----------
    Filename : (str)
        File name and path info

    OUTPUT :
    -------
    FileData : (dict)
        Dictionary containing all info and data for the file
    '''
    FileData=dict()
    FileData['Name']=FileName.split('.')[0]
    FileData['key']=[]
    FileData['startTime']=[]; FileData['endTime']=[]
    FileData['HPCP_Framebased']=[]; FileData['HPCP_Mean']=[]; FileData['HPCP_Std']=[]

    return FileData

def sliceAudiotoParts(audio,endTime,startTime,fs):
    '''
    Slicing the audio signal into parts according to annotated timestamps
    INPUT
    -----
    audio : (array)
        Input Audio Signal
    startTime : (float)
        start time of the audio segment of analysis (in seconds)
    endTime : (float)
        end time of the audio segment of analysis (in seconds)
    fs : (int)
        Sample Rate
    '''

    endtime=float(endTime)
    starttime=float(startTime)
    audio_slice = audio[int(starttime*fs):int(endtime*fs)]
    return audio_slice

def computeReferenceFrequency(tonic,tuningfreq):
    '''
    Computation of the reference frequency of HPCP vectors
    INPUT :
    -------
    tonic: (string)
        tonic of the audio segment
    tuningfreq : (float)
        tuning frequency of the audio segment, extracted using Essentia Algorithm
    OUTPUT :
    -------
    (float)
        returns the reference frequency of the audio segment (that corresponds to the key / tonic)
    '''

    keys = {'A':0,'Bb':1,'B':2,'C':3,'C#':4,'D':5,'Eb':6,'E':7,'F':8,'F#':9,'G':10,'Ab':11}
    for key in keys:
        if key == tonic:
            #compute reference frequency according to the key (tonic) of the audio file
            #divide the results by 2 the get one octave lower pitch as the reference freq
            return (tuningfreq * (2**((keys[key]*100)/1200)) /2)


'''
=====================================
         CHORD-SCALE DETECTION
=====================================
'''

def ScaleTypeEstimation(ChromaVector, ScaleTemplates):
    '''
    INPUT:
    ------
    ChromaVector : (array)
            Chroma Vector for estimating the scale type
    ScaleTemplates : (list)
            Scale type container that hast the binary templates of scale types in customized order

    OUTPUT:
    ------
    maxLikelihood: (tuple) ('SCALETEYPE', LIKELIHOOD SCALETYPE)
    Data of Likeliest Scale type

    List_ScaleLikelihoods: (list) [('SCALETYPE',LIKELIHOOD_SCALETYPE)]
    '''
    NUM_SCALETYPES = len(ScaleTemplates)
    ScaleLikelihoods = []

    List_ScaleLikelihoods = []
    ### Compute Likelihoods of each Scale type within ScaleTemplates.
    for i in range(NUM_SCALETYPES):
        Likelihood_ScaleType = ScaleLikelihoodEstimation(ChromaVector,ScaleTemplates[i][1])
        ScaleLikelihoods.append(Likelihood_ScaleType)

    LikelihoodsNORM = np.sum(ScaleLikelihoods)

    for i in range(NUM_SCALETYPES):
        ScaleLikelihoods[i] = ScaleLikelihoods[i]/LikelihoodsNORM
        List_ScaleLikelihoods.append((ScaleTemplates[i][0],ScaleLikelihoods[i]))

    LikeliestScale = [(ScaleTemplates[np.argmax(ScaleLikelihoods)],np.max(ScaleLikelihoods))]

    return(LikeliestScale,List_ScaleLikelihoods)

def ScaleLikelihoodEstimation(ChromaVector,scaleArray):
    '''
        Template based Scale type likelihood estimation.
        Computation is done by summation of the amplitudes of
        chroma bins

        INPUT:
        ------
        ChromaVector : (list)
            The chroma vector of analysis
        scaleArray : (array)
            The template of the scale-type for computing the scale estimation
        OUTPUT
        ------

        scale_likelihood : (float)
            the likelihood value of the chord-scale type

    '''

    NumNotesScale = np.sum(scaleArray)
    ChromaScaled = np.multiply(ChromaVector,scaleArray)
    scale_likelihood = np.sum(ChromaScaled) / NumNotesScale

    return(scale_likelihood)

def SegmentAnalysis(ExercisePart, ScaleTemplates):
    '''
    Segment-by-Segment Analysis of Student Performances

    INPUT :
    ------
    ExercisePart : (dict)
            data structure for that has relevant data of the audio segment of analysis (exercise part)

    ScaleTemplates : (list)
            Scale type container that hast the binary templates of scale types in customized order

    OUTPUT :
    ------
    LikeliestScale : (float)
            The estimated scale from the last accumulated chroma vector (the overall chroma histogram)

    LikelihoodsVector: (list)
            The list of scale type estimations

    '''

    PART_HPCP = ExercisePart['HPCP_Framebased']
    HPCPVector = np.zeros_like(PART_HPCP)
    LikelihoodsVector = []

    ###APPLY Scale Type Estimation on accummulated frame-based HPCP vectors (with the same HopSize)

    for k in range(len(PART_HPCP)):
        HPCPVector = HPCPVector + PART_HPCP[k] #accumulation of chroma vectors
        LikeliestScale, LikelihoodsArray = ScaleTypeEstimation(HPCPVector, ScaleTemplates)
        ### we will visualize LikelihoodsArray on frame level
        FramebasedLikelihoods = []
        for j in range(len(LikelihoodsArray)):
            FramebasedLikelihoods.append(LikelihoodsArray[j][1])
        FramebasedLikelihoods = np.array(FramebasedLikelihoods).reshape(1,-1)
        FramebasedLikelihoods = FramebasedLikelihoods[0]
        LikelihoodsVector.append(FramebasedLikelihoods)

    return LikeliestScale[0], LikelihoodsVector

'''
======================================
   PERFORMANCE ASSESSMENT & GRADING
======================================
'''


def ComputeInScaleRate(ChromaVector, ScaleArray):
    '''
    SCORE 1 : In-Scale Rate:
        The ratio of in-scale degrees in the student performance over all the degrees in chromatic scale
    INPUT
    ------
    ChromaVector : (list)
        The chroma vector of analysis
    ScaleArray : (array)
        The template of the scale-type for computing the scale estimation

    '''

    return np.sum(np.multiply(ChromaVector,ScaleArray)/np.sum(ChromaVector))

def ComputeCosineSimilarity(v1,v2):
    '''
    SCORE 2 : Scale Correctness:
        Compute Cosine Similarity of Binary Templates of 'Expected Scale Type' to 'Estimated Scale Type'
        using following equation:
        (v1 dot v2)/{||v1||*||v2||)

        This metric is used to measure the angular distance between the templates of the 'GROUND_TRUTH'
        and 'STUDENT_PERFORMANCE' scales.
    INPUT
    -----
    v1 : (array)
        Binary Template of Expected Chord-scale type
    v2 : (array)
        Binary Template of Student Chord-scale type
    OUTPUT
    -----
    CosineSimilarity_Scaled : (float)
        'Scale Correctness measure'
    '''
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    CosineSimilarity = (sumxy/math.sqrt(sumxx*sumyy))
        # NORMALIZE/SCALE between the maximum and minimum possible scores given the set of templates
        # Minimum Possible Score = 0.29, Maximum Possible Score = 1 (Full Match)
    CosineSimilarity_Scaled = (CosineSimilarity - 0.29) / 0.71

    return CosineSimilarity_Scaled

def ComputeScaleCompleteness(ChromaVector, ScaleArray):
    '''
    SCORE 3 : Scale Completeness:
        Ratio of number of played pitch-classes over the number of pitch-classes in the Expected (Target) Scale type

    INPUT:
    ------
    ChromaVector : (list)
        The chroma vector of analysis
    ScaleArray : (array)
        The template of the scale-type for computing the scale estimation
    OUTPUT:
    ------
    (float)
        'ScaleCompleteness'
    '''

    return (np.count_nonzero(np.multiply(ChromaVector,ScaleArray)))/np.count_nonzero(ScaleArray)

def PerformanceAssessment(StudentData, LikeliestScale, ScaleTemplates):

    '''
    The rule-based functions for student performance assessment

    INPUT :
    -----
    StudentData : (dict)

    likeliestScale :

    ScaleTemplates : (list)
            Scale type container that hast the binary templates of scale types in customized order

    '''

    ExpectedScale = StudentData['scaleType']

    for i in range(len(ScaleTemplates)):

        if ExpectedScale in ScaleTemplates[i][0]:
            scaleArrayExpected = ScaleTemplates[i][1]
        if LikeliestScale[0][0] in ScaleTemplates[i][0]:
            scaleArrayStudent = ScaleTemplates[i][1]

    chromaVector = StudentData['HPCP_Mean']
    stdchromaVector = StudentData['HPCP_Std']

    # IN-SCALE RATE
    inScaleRate = ComputeInScaleRate(chromaVector,scaleArrayExpected)
    # SCALE COMPLETENESS
    scaleCompleteness = ComputeScaleCompleteness(chromaVector, scaleArrayExpected)
    # number of played (or 'non-zero') Pitch-Classes
    NONZERO_PITCHES = np.count_nonzero(chromaVector)

    # SCALE CORRECTNESS
    '''
        CONDITION :
        1) IF the number of played notes are 2 less than that of Expected Scale:
            ScaleCorrectness = N/A (or '0' for computation) ==> Too few numbers to imply a Chord-Scale
        2) ELSE IF the number of played notes are more than previous case AND In-Scale Rate > 95%:
            ScaleCorrectness = 100%. Consider the performance as if the student has intended the correct chord-scale
            type but plays it incompletely. This relatively optimistic heuristic is due to the imperfection of
            HPCP Vectors
        3) ELSE :
            Compute ScaleCorrectness using Cosine Similarity
    '''
    if NONZERO_PITCHES < np.sum(scaleArrayExpected)-2 :
        return(inScaleRate, 0 , scaleCompleteness)

    elif NONZERO_PITCHES < np.sum(scaleArrayExpected) and inScaleRate > 95 :
        return(inScaleRate, 100 , scaleCompleteness)

    elif NONZERO_PITCHES < np.sum(scaleArrayExpected) and inScaleRate < 95 :
        scalechoicecorrectness = ComputeCosineSimilarity(scaleArrayExpected,scaleArrayStudent)
        return(inScaleRate, scalechoicecorrectness, scaleCompleteness)

    else:
        scalechoicecorrectness = ComputeCosineSimilarity(scaleArrayExpected,scaleArrayStudent)
        return inScaleRate, scalechoicecorrectness, scaleCompleteness


def MapScores2Four(grades):
    '''
    INPUT:
    --------
    grades : Grades class object
        overall grades of the student recording for providing an automatic assessment
        score between 1-4

    OUTPUT
    --------
    overall_score : int

        1 - Very Bad (out of context)
        2 - Bad (major mistakes)
        3 - Good (minor mistakes, tensions / out-scale notes present)
        4 - Very Good (very few or no mistakes / correct and complete scale choices)

    '''
    grades_SUM = grades.scale_correctness

    if grades_SUM < 0.5:
        overall_score = 1
    elif grades_SUM >= 0.5 and grades_SUM < 0.75:
        overall_score = 2
    elif grades_SUM >= 0.75 and grades_SUM < 1:
        overall_score = 3
    elif grades_SUM == 1:
        overall_score = 4

    return overall_score

'''
===============================================
                    EXTRAS
===============================================
'''

def ScaleDictionary():
    '''
    Dictionary that contains the predefined set of Chord-Scale Types and their binary templates.
    TODO - Better Structure!

    '''

    ScaleTemplates = []

    ScaleTemplates.append(['ionian',[1,0,1,0,1,1,0,1,0,1,0,1]])
    ScaleTemplates.append(['dorian',[1,0,1,1,0,1,0,1,0,1,1,0]])
    ScaleTemplates.append(['phrygian',[1,1,0,1,0,1,0,1,1,0,1,0]])
    ScaleTemplates.append(['lydian',[1,0,1,0,1,0,1,1,0,1,0,1]])
    ScaleTemplates.append(['mixolydian',[1,0,1,0,1,1,0,1,0,1,1,0]])
    ScaleTemplates.append(['aeolian',[1,0,1,1,0,1,0,1,1,0,1,0]])
    ScaleTemplates.append(['locrian',[1,1,0,1,0,1,1,0,1,0,1,0]])
    ScaleTemplates.append(['lydianb7',[1,0,1,0,1,0,1,1,0,1,1,0]])
    ScaleTemplates.append(['altered',[1,1,0,1,1,0,1,0,1,0,1,0]])
    ScaleTemplates.append(['melmin',[1,0,1,1,0,1,0,1,0,1,0,1]])
    ScaleTemplates.append(['hminor',[1,0,1,1,0,1,0,1,1,0,0,1]])
    ScaleTemplates.append(['wholetone',[1,0,1,0,1,0,1,0,1,0,1,0]])
    ScaleTemplates.append(['hwdiminished',[1,1,0,1,1,0,1,1,0,1,1,0]])

    return ScaleTemplates
