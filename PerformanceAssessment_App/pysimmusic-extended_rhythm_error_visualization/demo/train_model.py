"""
Code for prparing the model for chord estimation.
Ready-made model already included in the repository
(simmusic/chord_tools/data/ind_pdf.pkl)
"""

from pychord_tools.low_level_features import AnnotatedBeatChromaEstimator
from simmusic.utilities_chords import SimUidAndAudioPathExtractor, load_file_list

from pychord_tools.low_level_features import SmoothedStartingBeatChromaEstimator
from pychord_tools.third_party import NNLSChromaEstimator
from pychord_tools.models import IndependentPDFModel
import os
import json


pysim_chords_test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests/data/chords'))

extractor = SimUidAndAudioPathExtractor(pysim_chords_test_data_dir)
chromaEstimator = AnnotatedBeatChromaEstimator(
    chroma_estimator = NNLSChromaEstimator(audio_path_extractor=extractor),
    segment_chroma_estimator = SmoothedStartingBeatChromaEstimator(smoothing_time = 0.6),
    uid_extractor=extractor)
segments = chromaEstimator.load_chromas_for_annotation_file_list(load_file_list(
    os.path.join(pysim_chords_test_data_dir, 'annotations/correct.txt'),
    pysim_chords_test_data_dir))

m = IndependentPDFModel({'maj':['I', 'III', 'V'], 'min':['I', 'IIIb', 'V']}, {'maj': '', 'min': ':min'})
m.fit(segments)
m.save_model(os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../simmusic/extractors/chord_models/ind_pdf.pkl')))
