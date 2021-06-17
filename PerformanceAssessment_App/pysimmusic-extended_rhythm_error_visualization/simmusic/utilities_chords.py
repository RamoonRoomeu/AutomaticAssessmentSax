"""Utility function for chords subsystem
"""

from pychord_tools.low_level_features import UidExtractor, AudioPathExtractor
import json

def load_file_list(listFileName, pysim_chords_test_data_dir):
    result = []
    with open(listFileName) as list_file:
        for line in list_file:
            result.append(line.rstrip().replace('%PYSIM_CHORDS_TEST_DATA_DIR%', pysim_chords_test_data_dir))
    return result


class SimUidAndAudioPathExtractor(UidExtractor, AudioPathExtractor):
    def __init__(self, pysim_chords_test_data_dir):
        self.pysim_chords_test_data_dir = pysim_chords_test_data_dir

    def uid(self, annotation_file_name):
        print('annotation_file_name: ', annotation_file_name)
        with open(annotation_file_name) as json_file:
            data = json.load(json_file)
            audioPath = str(
                data['sandbox']['path']).replace('%PYSIM_CHORDS_TEST_DATA_DIR%', self.pysim_chords_test_data_dir)
            return audioPath

    def audio_path_name(self, uid):
        print('uid: ', uid)
        return super().audio_path_name(uid)

