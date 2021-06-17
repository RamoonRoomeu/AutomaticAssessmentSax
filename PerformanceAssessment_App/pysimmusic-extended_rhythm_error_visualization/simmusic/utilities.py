"""Utility function for different purposes
"""

import os
import numpy as np
import soundfile as sf
import resampy
from simmusic import constants


# ==============================================================================
# PITCH RELATED UTILS
# ==============================================================================


def hz2pitch(hertz, reference=440):
    """
    Converts from Hz to midi notes
    :param hertz: frequency in Hz
    :param reference: reference pitch in Hz
    :return: midi note
    """
    if not np.all(hertz):  # avoid division by zero
        return 0
    return 12 * np.log2(hertz / float(reference)) + 69


# C0 = semitone 0, C#0 = semitone 1, etc...
def pitch2hz(semitone, reference=440):
    """
    Convert midi note to frequency in Hz
    :param semitone:
    :param reference:
    :return:
    """
    return 2 ** ((semitone - 69) / 12.) * reference


def midi2note(midi_note, show_octave=True):
    """
    Convert midi note to musical note
    :param midi_note:
    :param show_octave: return number of octave
    :return:
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    total_notes = []
    for octave in range(11):
        for note in notes:
            if show_octave:
                total_notes.append(note + str(octave))
            else:
                total_notes.append(note)
    return total_notes[midi_note]


def to_db(vector, mul=10):
    """Transform vector to decibels avoiding -Inf results"""
    vector[vector == 0] = np.spacing(1)  # avoid -Inf results with spacing()
    return mul * np.log10(vector)


def window_bins(mili_seconds, sample_rate):
    """Returns the size in bins for a window of the desired milliseconds"""
    return sample_rate * mili_seconds // 1000


def minimum_window_size(pitch, window_factor=4, rounding=False, fs=44100):
    """
    Returns the minimum frame size needed to detect the given pitch
    @param pitch: lowest note to be detected (Hz)
    @param window_factor: window's main lobe bin's number (hann & hamming = 4, blackman-harris=9)
    @param rounding: round to the next power of 2
    @param fs: sampling rate
    @return:
    """
    period = 1. / pitch  # seconds
    window_size_seconds = window_factor * period  # minimum window size in seconds
    window_size_samples = window_size_seconds * float(fs)  # minimum window size in samples
    if rounding is True:
        window_size_samples = next_pow(window_size_samples, 2)
    return window_size_samples


def next_pow(x, power=2):
    """
    Compute next power of x
    :param x: int
    :param power: int
    :return: power power of x
    """
    return pow(power, np.ceil(np.log(x) / np.log(power)))


NOTE_NAMES = {'c': 0, 'b#': 0, 'do': 0, 'si#': 0,
              'c#': 1, 'db': 1, 'do#': 1, 'reb': 1,
              'd': 2, 're': 2,
              'd#': 3, 'eb': 3, 're#': 3, 'mib': 3,
              'e': 4, 'fb': 4, 'mi': 4, 'fab': 4,
              'f': 5, 'e#': 5, 'fa': 5, 'mi#': 5,
              'f#': 6, 'gb': 5, 'fa#': 6, 'solb': 5,
              'g': 7, 'sol': 7,
              'g#': 8, 'ab': 8, 'sol#': 8, 'lab': 8,
              'a': 9, 'la': 9,
              'a#': 10, 'bb': 10, 'la#': 10, 'sib': 10,
              'b': 11, 'cb': 11, 'si': 11, 'dob': 11}


def note2pitch(note_name, octave, octave_reference=4):
    """
    Returns a pitch in semitones given a note and its octave.
    @param note_name: the note name. For example: G#, C, Bb or Sol#, Do, Sib
    @param octave: the octave of the note
    @param octave_reference: in which octave do you consider middle C is?
    @return:
    """
    semitone = NOTE_NAMES[note_name.lower()]
    semitone += 12 * (octave + 4 - octave_reference)
    return semitone


def note2hz(note_name, octave, reference=440, octave_reference=4):
    """
    Returns a pitch in Hz given a note and its octave.
    @param note_name: the note name. For example: G#, C, Bb or Sol#, Do, Sib
    @param octave: the octave of the note
    @param reference: the tuning reference in hertz
    @param octave_reference: in which octave do you consider middle C is?
    @return: Hertz
    """
    semitone = note2pitch(note_name, octave, octave_reference)
    hz = pitch2hz(semitone, reference)
    return hz


def hz2cents_array(array, tonic, threshold, invalid_val):
    # TODO: documentation on this method
    """
    :param array:
    :param tonic:
    :param threshold:
    :param invalid_val:
    :return:
    """
    ind_valid = np.where(array > threshold)[0]
    output = invalid_val * np.ones(array.size)
    output[ind_valid] = 1200 * np.log2(array[ind_valid] / float(tonic))
    return output


def hz2cents(pitch_hz, tonic, threshold=constants.PITCH_MIN_VOCAL, invalid_val=constants.INVALID_PITCH_CENTS):
    # TODO: documentation on this method
    """
    :param pitch_hz:
    :param tonic:
    :param threshold:
    :param invalid_val:
    :return:
    """
    if pitch_hz > threshold:
        return 1200 * np.log2(pitch_hz / float(tonic))
    else:
        return invalid_val


# ==============================================================================
# TIME RELATED UTILS
# ==============================================================================


def samples2seconds(nsamples, fs=44100.):
    """
    Convert number of sample to time in seconds
    :param nsamples:
    :param fs:
    :return:
    """
    return nsamples / float(fs)


def seconds2samples(seconds, fs=44100, rounding=False):
    """
    Convert time in seconds to number of samples
    :param seconds:
    :param fs:
    :param rounding:
    :return:
    """
    samples = seconds * fs
    if rounding:
        samples = np.round(samples)
    return samples


def seconds2frames(time, fs=44100, hop_size=512):
    """
    Convert time in seconds to frame number
    :param time:
    :param fs:
    :param hop_size:
    :return:
    """
    return np.round(time * fs / float(hop_size))


def frames2seconds(frames, fs=44100, hop_size=512):
    """
    Convert number of frame to time in seconds
    :param frames:
    :param fs:
    :param hop_size:
    :return:
    """
    return frames * hop_size / float(fs)


# ==============================================================================
# OTHER UTILS
# ==============================================================================


def find_yin_minima(array, min_ind, max_ind, threshold):
    # TODO: add documentation
    min_val = constants.FLT_MAX
    min_off = -1
    first_valley = False

    for ii in np.arange(min_ind + 1, max_ind):
        if array[ii] < threshold and not first_valley:
            first_valley = True
        if array[ii] >= threshold and first_valley:
            break
        if first_valley and array[ii] < min_val:
            min_val = array[ii]
            min_off = ii
    return min_val, min_off


def parabolic_interpolation(left_val, middle_val, right_val, current_bin):
    # TODO: add documentation
    delta_x = 0.5 * ((left_val - right_val) / (left_val - 2 * middle_val + right_val))
    result_bin = float(current_bin) + delta_x
    return result_bin


def find_interval_index(intervals, val):
    # TODO: add documentation
    for ii, intVal in enumerate(intervals[:-1]):
        if intVal <= val < intervals[ii + 1]:
            return ii
    return None


def compute_nearest_index(array, val, ind_start, ind_end):
    """compute nearest index
    Parameters
    ----------
    array: input array
    val: value (for which the minimum index is to be computed)
    ind_start: start index of the array
    ind_end: end index of the array
    :return:
    """
    return np.argmin(np.abs(np.array(array[ind_start: ind_end]) - val)) + ind_start


#
# def groupIndices(indexes):
#     """
#     This function groups indexes. This is often needed to produce segments given indices.
#     """
#     segments = []
#     segStart = indexes[0]
#     N = len(indexes)
#     for ii in range(len(indexes)):
#         if ii == N-1:
#             segments.append([segStart, indexes[ii]])
#             return np.array(segments)
#         if indexes[ii]+1 != indexes[ii+1]:
#             segments.append([segStart, indexes[ii]])
#             segStart = indexes[ii+1]
#
#     return np.array(segments)

def group_indices(indexes):
    """
    This function groups indexes. This is often needed to produce segments given indices.
    #ASSUMPTION: given indices are in sorted order!
    # highly optimized version of the same code done with loops
    """

    diff_inds = np.where(np.abs(np.diff(indexes)) > 1)[0]
    diff_points = np.concatenate(([-1], diff_inds, [len(indexes) - 1]))
    length = diff_points.size
    pairs = np.hstack((diff_points[:-1].reshape(length - 1, 1) + 1, diff_points[1:].reshape(length - 1, 1)))
    # pairs = zip(diff_points[::]+1, diff_points[1::])
    segments = indexes[pairs]
    return np.array(segments)


def compute_hopsize(time_stamps):
    """
    This function computes hop size given time stamps of a feature

    Parameters
    ----------
    time_stamps : array_like (np.ndarray)

    Returns
    -------
    hop size : float
        Hop size (average) of the time stamps
    """

    diff = np.diff(time_stamps)
    return np.mean(diff)


def center_of_mass_1d(x):
    """
    This function computes center of mass for a 1 dimensional data.

    Parameters
    ----------
    x : array (N,)
        Input array

    Returns
    -------
    com : float
        Center of mass of an one dimensional array

    """
    inds = np.arange(x.size)
    return np.sum(x*inds)/np.sum(x)

# ==============================================================================
# AUDIO RELATED UTILS
# ==============================================================================


def resample_kaldi(input_audio_filename,
                   output_audio_file_path,
                   output_filename,
                   resampling_fs):
    """
    Resampy audio to resampling_fs and copy the file to the output directory for kaldi audio format
    :param input_audio_filename: input audio full filename
    :param output_audio_file_path: output file path
    :param output_filename: output filename
    :param resampling_fs: resampling rate
    :return: bool
    """
    output_audio_full_path = os.path.join(output_audio_file_path, output_filename)

    # create the output path if not exist
    if not os.path.exists(output_audio_file_path):
        os.makedirs(output_audio_file_path)

    data, fs = sf.read(input_audio_filename)
    if len(data.shape) > 1:
        data = data[:, 0]
    if fs != resampling_fs:
        data = resampy.resample(data, fs, resampling_fs)
    sf.write(output_audio_full_path, data, resampling_fs)

    return data
