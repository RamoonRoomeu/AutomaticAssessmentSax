import os
import pickle
import pandas as pd
from simmusic.feature_extraction import extract_melody_timing_features, extract_pitch_melodia
from simmusic import constants
import numpy as np


def estimate_grades(trans_file_ref, temp_pitch_filename, training_set, regression_method, sound_filename=None):
    """Extractor for estimating grades in the musiccritic project

    This is the main extractor used for estimating grades for student performances. This is the extractor used in the
    initial versions of the MusicCritic project. The grading is based on learning from annotations. There are different
    models based on learnings on different datsets and different learning algorithms. Depending on the context choose
    the appropriate method. However, since incremently the performance is improved, the method added latter are more
    accurate / sophisticated (e.g. 'mlp_regression').

    Parameters
    ----------
    sound_filename : str
        Name of the performance audio file of a student
    trans_file_ref : str
        Name of the transcription file of a reference lesson
    temp_pitch_filename : str
        File path of the pitch file that will be created on the feature extraction step
    training_set : str
        Name of the training set used, options are ('Riyaz_MC' or 'MAST')
    regression_method : str
        Name of the regression model ('linear_regression' or 'mlp_regression')

    Returns
    -------
    grades : float
        grades obtained

    Notes
    -----
    The system primarily is tuned to the MAST dataset. So for normal situations prefer the model learned on the MAST.
    Also mlp_regression performs better than the linear_regression, so that's better choice in general.

        """

    # Important parameters
    resize_length = 200 # This is decent for the MAST dataset, in the future select it more intelligently.
    auto_pitch_shift = True
    min_seg_dur = 0.1
    filter_corner_seg = True

    # first step is to extract pitch
    if sound_filename:
        extract_pitch_melodia(sound_filename, temp_pitch_filename)
        no_extraction = False
    else:
        no_extraction = True

    # extract features based on pitch
    features_for_file = extract_melody_timing_features(temp_pitch_filename, trans_file_ref,
                                                       auto_pitch_shift=auto_pitch_shift,
                                                       resize_length=resize_length,
                                                       ref_freq=constants.REF55HZ,
                                                       min_segment_duration=min_seg_dur,
                                                       filter_corner_segments=filter_corner_seg,
                                                       no_extraction=no_extraction)

    # Creating a panda object from it
    features_for_file = pd.DataFrame([features_for_file], index=[0])

    # selecting the set of features used to generate the model
    if training_set.lower() in constants.feature_names:
        features_selected = constants.feature_names[training_set.lower()][regression_method]
    else:
        raise ValueError('Please specify a valid dataset name.')

    # load normalizing factors
    if training_set.lower() in constants.model_directories:
        model_dir = os.path.join(os.path.dirname(__file__),
                                 constants.model_directories[training_set.lower()][regression_method])
    else:
        raise ValueError('Please specify a valid dataset name.')
    norm_facs = pd.read_pickle(os.path.join(model_dir, 'norm_factors.pkl'))

    # performing normalization using the factors
    features_for_file = (features_for_file[features_selected] -
                         norm_facs[features_selected].loc['mean']) / norm_facs[features_selected].loc['std']

    # loading the linear regression model for grading
    model = pickle.load(open(os.path.join(model_dir,
                                          constants.model_name_map[training_set.lower()][regression_method]), 'rb'))

    # doing the prediction
    prediction = model.predict(features_for_file)

    # restricting the output of the regression model to min and max rating used in the dataset
    score = prediction[0]

    # fetching the grading levels for this dataset
    if training_set.lower() in constants.grade_levels:
        grade_levels = constants.grade_levels[training_set.lower()]
    else:
        raise ValueError('Please specify a valid dataset name.')

    if score < np.min(grade_levels):
        score = np.min(grade_levels)
    if score > np.max(grade_levels):
        score = np.max(grade_levels)

    # returning the score
    return score
