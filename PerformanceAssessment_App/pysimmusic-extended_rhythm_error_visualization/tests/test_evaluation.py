import os
import unittest
import test_utilities as utils
from simmusic import evaluation


exts = {'ref_pitch': '.pitch', 'user_pitch': '.pitch_user', 'transcription': '.trans', 'feedback': '.fdbk',
        'log': '.log', 'params': '.params'}

scratch_dir = os.path.join(os.path.dirname(__file__), 'scratch_dir')
if not os.path.isdir(scratch_dir):
    os.makedirs(scratch_dir)


class TestEvaluation(unittest.TestCase):
    def test_eval_flatnote_playalong_rule_based_1_case1(self):
        """
        Testing the code for scoring flat notes played as play-along mode (first rule based approach)
        :return:
        """
        filename = os.path.join(os.path.dirname(__file__), 'data/riyaz_psds/0AL0nEmjVR/0AL0nEmjVR')
        output_file = os.path.join(scratch_dir, '0AL0nEmjVR' + exts['feedback'])
        scoring_type = 0

        if os.path.isfile(output_file):
            os.remove(output_file)

        evaluation.evaluate_flatnote_playalong_rule_based(filename + exts['user_pitch'],
                                                          filename + exts['transcription'],
                                                          output_file,
                                                          scoring_logic=scoring_type,
                                                          output_format='text')
        is_same = False
        if os.path.isfile(output_file) and os.path.isfile(filename + exts['feedback']):
            is_same = utils.compare_text_feedback_files(output_file, filename + exts['feedback'])
        self.assertTrue(is_same)

    def test_eval_flatnote_playalong_rule_based_2_case1(self):
        """
        Testing the code for scoring flat notes played as play-along mode (second rule based approach)
        :return:
        """
        filename = os.path.join(os.path.dirname(__file__), 'data/riyaz_psds/zzzx4FqSC5/zzzx4FqSC5')
        output_file = os.path.join(scratch_dir, 'zzzx4FqSC5' + exts['feedback'])
        scoring_type = 1

        if os.path.isfile(output_file):
            os.remove(output_file)

        evaluation.evaluate_flatnote_playalong_rule_based(filename + exts['user_pitch'],
                                                          filename + exts['transcription'],
                                                          output_file,
                                                          scoring_logic=scoring_type,
                                                          output_format='text')
        is_same = False
        if os.path.isfile(output_file) and os.path.isfile(filename + exts['feedback']):
            is_same = utils.compare_text_feedback_files(output_file, filename + exts['feedback'])
        self.assertTrue(is_same)

    def test_eval_flatnote_dialogue_rule_based_case1(self):
        """
        Testing the code for scoring flat notes played as play-along mode (second rule based approach)
        :return:
        """
        filename = os.path.join(os.path.dirname(__file__), 'data/other/0AL0nEmjVR/0AL0nEmjVR')
        output_file = os.path.join(scratch_dir, '0AL0nEmjVR_dialogue' + exts['feedback'])
        scoring_type = 1

        if os.path.isfile(output_file):
            os.remove(output_file)

        try:
            evaluation.evaluate_flatnote_dialogue_rule_based(filename + exts['user_pitch'],
                                                             filename + exts['transcription'],
                                                             output_file,
                                                             scoring_logic=scoring_type,
                                                             output_format='text')
        except Exception as e:
            print(e)
        is_same = False
        if os.path.isfile(output_file) and os.path.isfile(filename + exts['feedback']):
            is_same = utils.compare_text_feedback_files(output_file, filename + exts['feedback'])
        self.assertTrue(is_same)

    def test_eval_phrase_dialogue_rule_based_case1(self):
        """
        Testing the code for scoring phrases played as dialogue mode (rule based approach)
        :return:
        """
        filename = os.path.join(os.path.dirname(__file__), 'data/other/yaman_MDN_NRS/yaman_MDN_NRS')
        output_file = os.path.join(scratch_dir, 'yaman_MDN_NRS' + exts['feedback'])

        if os.path.isfile(output_file):
            os.remove(output_file)
        try:
            evaluation.evaluate_phrase_dialogue_rule_based(filename + exts['ref_pitch'],
                                                           filename + exts['user_pitch'],
                                                           output_file)
        except Exception as e:
            print(e)
        is_same = False
        if os.path.isfile(output_file) and os.path.isfile(filename + exts['feedback']):
            is_same = utils.compare_text_feedback_files(output_file, filename + exts['feedback'])
        self.assertTrue(is_same)


if __name__ == '__main__':
    unittest.main()
