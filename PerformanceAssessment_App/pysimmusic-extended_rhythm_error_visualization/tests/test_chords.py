import unittest
import json
import os
import pychord_tools.common_utils as common_utils
import simmusic.extractors.chords as mc_grade
from simmusic.utilities_chords import SimUidAndAudioPathExtractor, load_file_list

pysim_chords_test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/chords'))

def readFromJSON(jsonFileName, uidAndAudioPathExtractor = SimUidAndAudioPathExtractor(pysim_chords_test_data_dir)):
    with open(jsonFileName) as json_file:
        data = json.load(json_file)
    duration = float(data['duration'])
    metreNumerator = int(data['metre'].split('/')[0])
    allBeats = []
    allChords = []
    common_utils.process_parts(
        metreNumerator, data, allBeats, allChords, 'chords')
    duration = min(duration,  2 * allBeats[-1] - allBeats[-2])
    path, uid = uidAndAudioPathExtractor.uidAndAudioPathName(jsonFileName)
    return path, uid, allBeats, duration

class TestGraders(unittest.TestCase):
    def test_generic_interface4(self):
        berkleeAnnoPathname = os.path.join(pysim_chords_test_data_dir, 'annotations/berklee_demo.json')
        berkleeSoundPathname, uid, beats, duration = readFromJSON(berkleeAnnoPathname)
        grade, chords_scores, png_bytes = mc_grade.estimate_grade(
            beats,
            berkleeSoundPathname,
            berkleeSoundPathname,
            duration)
        self.assertEqual(4, grade)

    def do_berklee_grade_test(self, fileName, expectedGrade):
        berkleeAnnoPathname = os.path.join(pysim_chords_test_data_dir, 'annotations/berklee_demo.json')
        berkleeSoundPathname, uid, beats, duration = readFromJSON(berkleeAnnoPathname)

        gradeAnnoPathname = os.path.join(pysim_chords_test_data_dir, fileName)
        gradeSoundPathname, uid_g, beats_g, duration_g = readFromJSON(gradeAnnoPathname)

        grade, chords_scores, png_bytes = mc_grade.estimate_grade(
            beats,
            berkleeSoundPathname,
            gradeSoundPathname,
            duration_g,
            student_beats_array=beats_g)
        self.assertEqual(expectedGrade, grade)

    def test_generic_interface3(self):
        self.do_berklee_grade_test('annotations/1416430482.json', 3)

    def test_generic_interface2(self):
        self.do_berklee_grade_test('annotations/1370305373.json', 2)

    def test_generic_interface1(self):
        self.do_berklee_grade_test('annotations/1455539867.json', 1)

    def test_bpm_interface(self):
        sevaDemoPathname = os.path.join(pysim_chords_test_data_dir, 'audio/seva_demo.mp3')
        grade, chords_scores, png_bytes = mc_grade.estimate_grade_bpm(
            60, 4, 32,
            sevaDemoPathname,
            sevaDemoPathname)
        self.assertEqual(4, grade)
