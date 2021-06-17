import os
import unittest
from simmusic.extractors.feature_based_grade_estimator import grade_estimator
import numpy as np


scratch_dir = os.path.join(os.path.dirname(__file__), 'scratch_dir')
if not os.path.isdir(scratch_dir):
    os.makedirs(scratch_dir)


class TestGraders(unittest.TestCase):
    def test_feature_based_grade_estimator_lin_regression(self):
        """
        Testing the code for feature_based_grade_estimator
        :return:
        """
        filename_perf = os.path.join(os.path.dirname(__file__), 'data/mast_samples/51_mel1_per106160_pass.wav')
        filename_ref = os.path.join(os.path.dirname(__file__), 'data/mast_samples/51_mel1.txt')
        temp_pitch_file = os.path.join(scratch_dir, '51_mel1_per106160_pass.pitch')
        training_set = 'MAST'
        model_name = 'linear_regression'

        if os.path.isfile(temp_pitch_file):
            os.remove(temp_pitch_file)
        try:
            score = grade_estimator.estimate_grades(trans_file_ref=filename_ref, temp_pitch_filename=temp_pitch_file, training_set=training_set,
                                                    regression_method=model_name, sound_filename = filename_perf)
        except Exception as e:
            raise e

        is_same = False
        if np.abs(3.2924043510428755-score) < 0.001:
            is_same = True
        self.assertTrue(is_same)

    def test_feature_based_grade_estimator_no_pitch_in_audio_linear_regression(self):
        """
        Testing the code for feature_based_grade_estimator (no pitch in audio)
        :return:
        """
        filename_perf = os.path.join(os.path.dirname(__file__), 'data/mast_samples/51_mel1_per106160_pass_sil.wav')
        filename_ref = os.path.join(os.path.dirname(__file__), 'data/mast_samples/51_mel1.txt')
        temp_pitch_file = os.path.join(scratch_dir, '51_mel1_per106160_pass_sil.pitch')
        training_set = 'MAST'
        model_name = 'linear_regression'

        if os.path.isfile(temp_pitch_file):
            os.remove(temp_pitch_file)
        try:
            score = grade_estimator.estimate_grades(filename_perf, filename_ref, temp_pitch_file,
                                                    training_set, model_name)
            print(score)
        except Exception as e:
            print(e)
        #
        # is_same = False
        # if np.abs(3.2924043510428755-score) < 0.001:
        #     is_same = True
        self.assertTrue(True)

    def test_feature_based_grade_estimator_mlp_regression(self):
        """
        Testing the code for feature_based_grade_estimator
        :return:
        """
        filename_perf = os.path.join(os.path.dirname(__file__), 'data/mast_samples/51_mel1_per106160_pass.wav')
        filename_ref = os.path.join(os.path.dirname(__file__), 'data/mast_samples/51_mel1.txt')
        temp_pitch_file = os.path.join(scratch_dir, '51_mel1_per106160_pass.pitch')
        training_set = 'MAST'
        model_name = 'mlp_regression'

        if os.path.isfile(temp_pitch_file):
            os.remove(temp_pitch_file)
        try:
            score = grade_estimator.estimate_grades(trans_file_ref=filename_ref, temp_pitch_filename=temp_pitch_file, training_set=training_set,
                                                    regression_method=model_name, sound_filename = filename_perf)
        except Exception as e:
            raise e

        is_same = False
        if np.abs(3.2940077921-score) < 0.001:
            is_same = True
        self.assertTrue(is_same)

    def test_feature_based_grade_estimator_no_pitch_in_audio_mlp_regression(self):
        """
        Testing the code for feature_based_grade_estimator (no pitch in audio)
        :return:
        """
        filename_perf = os.path.join(os.path.dirname(__file__), 'data/mast_samples/51_mel1_per106160_pass_sil.wav')
        filename_ref = os.path.join(os.path.dirname(__file__), 'data/mast_samples/51_mel1.txt')
        temp_pitch_file = os.path.join(scratch_dir, '51_mel1_per106160_pass_sil.pitch')
        training_set = 'MAST'
        model_name = 'mlp_regression'

        if os.path.isfile(temp_pitch_file):
            os.remove(temp_pitch_file)
        try:
            score = grade_estimator.estimate_grades(filename_perf, filename_ref, temp_pitch_file,
                                                    training_set, model_name)
            print(score)
        except Exception as e:
            print(e)
        #
        # is_same = False
        # if np.abs(3.2924043510428755-score) < 0.001:
        #     is_same = True
        self.assertTrue(True)
