# test
from simmusic.extractors.solfege_grade_estimator_xml import SolfegeGradeEstimator
import os
import tempfile
import shutil
import numpy as np

# paths
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

input_xml_filename_ref = os.path.join(data_dir, "solfege_musicxml", "exercise-43-1.musicxml")
input_audio_filename_stu = os.path.join(data_dir, "solfege_audio", "724.wav")

# create temporary directory
output_file_path = tempfile.mkdtemp()
visualization_output_file_path = os.path.join(output_file_path,
                                                "syllable_assessment_visualization.png")

# parameters
tempo = 88.0
initial_sil_beats = 4.0
overall_beats = 16.0

quarter_time = 60.0 / tempo
segment_start = 60.0 / tempo * initial_sil_beats - 0.5
segment_duration = 60 / tempo * overall_beats - 0.5

grader = SolfegeGradeEstimator(output_file_path=output_file_path,
                              quarter_time=quarter_time,
                              segment_start=segment_start,
                              segment_duration=segment_duration)

grader.reference_feature_extraction(filename_input_ref_xml=input_xml_filename_ref)

grader.student_feature_extraction(filename_input_stu_audio=input_audio_filename_stu)

grader.syllable_sequences_alignment()

grader.onset_assessment()

grader.pitch_assessment()

grader.fill_the_rest_scores()

print("onset assessment score REF audio {} and STUDENT audio {} is {}".
      format(input_xml_filename_ref, input_audio_filename_stu, grader.score_overall_onset))

print("pitch assessment score REF audio {} and STUDENT audio {} is {}".
      format(input_xml_filename_ref, input_audio_filename_stu, np.mean(grader.score_syllable_pitch)))

grader.score_plot_no_reference(visualization_output_path=visualization_output_file_path)
print("Visualisation saved to %s" % visualization_output_file_path)

# # remove temporary directory
shutil.rmtree(output_file_path)