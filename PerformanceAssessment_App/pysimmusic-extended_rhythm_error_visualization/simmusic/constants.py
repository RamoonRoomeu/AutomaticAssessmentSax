"""This file defines all the constants to be used in this library for different purposes.
"""

import numpy as np

# Math constants
INF = np.finfo(np.float).max
LOG2 = 0.693147180559945
EPS = np.finfo(np.float).min
FLT_MAX = INF
FLT_MIN = -1 * FLT_MAX

# Pitch related constants
CENTS_PER_OCTAVE = 1200.0
SVAR_THRESHOLD_CENTS_TIGHT = 20
SVAR_THRESHOLD_CENTS_LOOSE = 50
SVAR_THRESHOLD_CENTS_MEDIUM = 35
INVALID_PITCH_CENTS = -100000
PITCH_HZ_ZERO = 0
PITCH_MIN_VOCAL = 60
REF55HZ = 55.0

# Evaluation related constants
INVALID_EVAL_CODE = -1
INVALID_SCORE = -2
INVALID_INDEX = -3
MIN_TIME_DIFF_ALLOWED = 0.5  # (seconds). This is the time that is allowed between the closest sample in a feature time
# stamp and the note boundary. In Riyaz this was needed because some low quality phones did
# hang leaving a good amount of chunk in time without any pitch values

# Other constants
RECORDING_TYPES = ['reference',
                   'imitation']
RECORDING_MODES = ['play-along',
                   'dialogue',
                   'improvisation']

feature_names = {
    'riyaz_mc': {
        'linear_regression': ['diff_100_Inf', 'diff_20_50', 'diff_50_100', 'dtw_cost', 'mean_diff_ext', 'std_diff',
                              'va_ratio']},
    'mast': {
        'linear_regression': ['diff_100_Inf', 'diff_20_50', 'diff_50_100', 'dtw_cost', 'mean_diff', 'ratio_note_dur',
                              'mean_note_diff', 'max_note_diff', 'derivative_mean_notes', 'mean_diff_ext',
                              'mean_path_dev', 'path_len_to_ref_len', 'std_diff', 'va_ratio'],
        'mlp_regression': ['diff_100_Inf', 'diff_20_50', 'diff_50_100', 'dtw_cost', 'mean_diff', 'ratio_note_dur',
                           'mean_note_diff', 'max_note_diff', 'derivative_mean_notes', 'mean_diff_ext',
                           'mean_path_dev', 'path_len_to_ref_len', 'std_diff', 'va_ratio']}

}

model_directories = {
    'mast': {
        'linear_regression': 'MAST_model/linear_regression',
        'mlp_regression': 'MAST_model/mlp_regression'},
    'riyaz_mc': 'Riyaz_MC_model'}

model_name_map = {
    'mast': {
        'linear_regression': 'lin_reg_model.pkl',
        'mlp_regression': 'mlp_reg_model.pkl'},
    'riyaz_mc': 'lin_reg_model.pkl'}

grade_levels = {'mast': [1, 2, 3, 4], 'riyaz_mc': [1, 2, 3, 4]}

dict_solfege_name = {"C": "DO",
                     "D": "RE",
                     "E": "MI",
                     "F": "FA",
                     "G": "SOL",
                     "A": "LA",
                     "B": "SI"}
