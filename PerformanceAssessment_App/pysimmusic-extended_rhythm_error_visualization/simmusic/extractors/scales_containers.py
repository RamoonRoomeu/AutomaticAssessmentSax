import os, sys, warnings,math
import xml, xmltodict
import numpy as np
from simmusic.extractors.scales_utilities import *

class Annotation:
    """Object for annotation file.
        This object comprises the function to read XML and parse the relevant
        data to 'dictionary' format
           Attributes
           ----------
           data_dict : dictionary
               Dictionary for annotation data
           ReadAnnotationFile : function
               Function to read XML and directly parse relevant information to proper
               format for processing
           """
    def __init__(self, data_dict = {}):
        self. data_dict = data_dict

    def __repr__(self):
        return 'Annotation File (data_dict={})'.format(self.data_dict)

    def read_annotation_file(self, XMLfile):
        self.data_dict = ConvertXML2Annotation(XMLfile)



class Recording(object):
    """Object for representing a recording.
        This object comprises features and related properties (transcription/segments etc)
       Attributes
       ----------
       rec_type : str
           Type of recording, either 'reference' or a 'student'
       mode : str
           Mode of recording; 'play-along', 'dialouge' or 'improvisation'
       annotation_data : dict
       
       segments : list
           List of :class:`simmusic.Segment` objects
       scale_dictionary : dictionary   
       
       overall_grades : class Grades
       
       """

    def __init__(self,
                 rec_type = 'student',
                 mode = 'improvisation',
                 bpm = None,
                 meter = None,
                 scale_dictionary = None,
                 overall_grades = None,
                 final_score = None):
        self.rec_type = rec_type
        self.mode = mode
        self.bpm = bpm
        self.meter = meter
        self.scale_dictionary = scale_dictionary
        self.segments = []
        self.overall_grades = overall_grades
        self.final_score = final_score

    def __repr__(self):
        return 'Recording (type={}, mode={},\
                segments={}, bpm = {}, meter = {},\
                overall_grades = {})'\
                .format(self.rec_type, self.mode,\
                '<%s of length %d>' % (type(self.segments).\
                __name__, len(self.segments)), self.bpm, self.meter, self.overall_grades, self.final_score)

    def evaluate_performance(self, audioFileName, AnnotationFile):
        '''
        This function performs feature extraction and performance assessment automatically 
        for each audio segment of analysis. The feature data and grades are assigned to 
        related attributes of Recording object
        
        PARAMETERS
        ----------
        audioFileName = audio file (mp3, wav, ...)
            Audio file of student recording
        AnnotationFile : dict
            Annotation data
        '''
        
        #PARSE global parameters to Recording Object (bpm, meter)
        self.bpm = AnnotationFile['bpm']
        self.meter = AnnotationFile['meter']
        #ANALYSIS Parameters
        parameters = AnalysisParams(200, 100, 'hann', 2048, 44100)
        #INITIATE Scales Dictionary
        ScaleTemplates = ScaleDictionary()
        self.scale_dictionary = ScaleTemplates
        #FEATURE Extraction
        FeaturesDictionary = FeatureExtraction(audioFileName, AnnotationFile, parameters)
        #CONTAINERS for overall grades calculation
        inscalerateS = []; scalecorrectnesS = []; scalecompletenesS = [];
        
        self.segments = []
        for i in range(len(FeaturesDictionary)):
            segment_features = FeaturesDictionary['Part'+str(i+1)]
            #ANALYZE audio segments and ESTIMATE chord-scale type
            likeliest_scaleType, likelihoods_vector = SegmentAnalysis(segment_features,ScaleTemplates)           
            
            #CREATE AudioSegment object for storing feature data           
            segment = AudioSegment()
            #PARSE Feature Data to AudioSegment Object
            segment.title = segment_features['Name']
            segment.startTime = segment_features['startTime']
            segment.endTime = segment_features['endTime']
            segment.key = segment_features['key']
            segment.expected_scaleType = segment_features['scaleType']
            segment.estimated_scaleType = likeliest_scaleType[0][0]
            segment.HPCP_framebased = segment_features['HPCP_Framebased']
            segment.HPCP_mean = segment_features['HPCP_Mean']
            segment.HPCP_std = segment_features['HPCP_Std']
            
            # PERFORMANCE assessment & evaluation
            scores = PerformanceAssessment(segment_features,likeliest_scaleType,ScaleTemplates)
            #CREATE Grades object for storing grades data
            segment_grades = Grades()
            #PARSE scores to Grades object
            segment_grades.inscale_rate = scores[0]; inscalerateS.append(scores[0])
            segment_grades.scale_correctness = scores[1]; scalecorrectnesS.append(scores[1])
            segment_grades.scale_completeness = scores[2]; scalecompletenesS.append(scores[2])
            segment.grades = segment_grades
            #APPEND to 'segment' attribute of Recording object
            self.segments.append(segment)
         
        #CREATE Grades object for storing overall grades data
        overallGrades = Grades()
        #CALCULATE & PARSE overall grades data to Recording Object
        overallGrades.inscale_rate = np.sum(inscalerateS) / len(FeaturesDictionary)
        overallGrades.scale_correctness = np.sum(scalecorrectnesS) / len(FeaturesDictionary)
        overallGrades.scale_completeness = np.sum(scalecompletenesS) / len(FeaturesDictionary)        
        self.overall_grades = overallGrades
        
        #MAP overall Grades between (1-4) for providing Final Score for the app
        finalscore = MapScores2Four(overallGrades)
        self.final_score = finalscore
        
class AudioSegment(object):
    '''
    Object for representing an audio segment of analysis
    Attributes
       ----------
       featureData : dictionary
           Dictionary that contains the feature Data
       grades : nd.array
           Assessment scores (grades) of the audio segment
    
    '''
    
    def __init__(self,
                 title = None,
                 startTime = None,
                 endTime = None,
                 key = None,
                 expected_scaleType = None,
                 estimated_scaleType = None,                
                 HPCP_framebased = None, 
                 HPCP_mean = None, 
                 HPCP_std = None,                 
                 grades = None):
        self.title = title
        self.startTime = startTime
        self.endTime = endTime
        self.key = key
        self.expected_scaleType = expected_scaleType
        self.estimated_scaleType = estimated_scaleType
        self.HPCP_framebased = HPCP_framebased
        self.HPCP_mean = HPCP_mean
        self.HPCP_std = HPCP_std
        self.grades = grades
        
class Grades(object):
    '''
    Object for representing the Grades Class Object
    '''

    def __init__(self,
                 inscale_rate = None,
                 scale_correctness = None,
                 scale_completeness = None):
        '''
        inscale_rate: float
        scale_correctness: float
        scale_completeness: float 
        '''
        self.inscale_rate = inscale_rate
        self.scale_correctness = scale_correctness
        self.scale_completeness = scale_completeness

    def __repr__(self):
        return 'Grades (inscale_rate={}, scale_correctness={}, scale_completeness={})'.\
                                            format(self.inscale_rate, self.scale_correctness,
                                                   self.scale_completeness)                
        

class AnalysisParams:
    '''
    Analysis Parameters Class Object
    '''

    def __init__(self, windowSize, hopSize, windowFunction, fftN, fs):
        '''
        windowSize: milliseconds,
        hopSize: milliseconds,
        windowFunction: str ('blackman','hanning',...)
        fftN: int
        '''
        self.windowSize = windowSize
        self.hopSize = hopSize
        self.windowFunction = windowFunction
        self.fftN = fftN
        self.fs = fs

    def __repr__(self):
        return 'AnalysisParams (windowSize={}, hopSize={}, windowFunction={}, fftN={}, fs={} )'.\
                                            format(self.windowSize, self.hopSize,
                                                   self.windowFunction, self.fftN, self.fs)
                                                              
        
