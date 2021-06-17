import simmusic.extractors.scales_containers as sc
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'scales')

annotationFileName = os.path.join(data_dir,'exercise.xml')
audioFileName = os.path.join(data_dir, 'SamplePerformance.wav')

# INITIATE Annotation Class object
annotation = sc.Annotation()

# READ annotation data from music XML file, which is extracted using MuseScore.
annotation.read_annotation_file(annotationFileName)

# IMPORTANT NOTE : Please read  'readme.txt' in 'demo_scales' folder for Chord-Scale Exercise
# creation steps for extracting XML file in proper format

# INITIATE Recording Class object
recording = sc.Recording()

# EVALUATE Student Performance
recording.evaluate_performance(audioFileName,annotation.data_dict)

# ACCESS attributes of recording object

print('Final Scores (1-4) : ' + str(recording.final_score), "\n")

print('Some Evaluation Dimensions for Feedback (0-100) \n')
print('In-scale Rate : ' , str(recording.overall_grades.inscale_rate), "\n")
print('Scale Correctness : ' , str(recording.overall_grades.scale_correctness), "\n")
print('Scale Completeness : ' , str(recording.overall_grades.scale_completeness), "\n")

# ACCESS feature data and assessment scores for each Part of the exercise separately

# Choose part of exercise
PART = 3

print('Exercise ', recording.segments[PART].title, "\n")
print('Grades (part)', recording.segments[PART].grades, "\n")

print('Summarized Features (mean)', recording.segments[PART].HPCP_mean, "\n")
#print('Summarized Features (std)', recording.segments[PART].HPCP_std, "\n")
#print('Frame-based Features', recording.segments[PART].HPCP_framebased, "\n")

print('Estimated Chord-Scale Type : ', recording.segments[PART].estimated_scaleType, "\n")
