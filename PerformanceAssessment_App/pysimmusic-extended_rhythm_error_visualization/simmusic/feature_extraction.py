import copy
import vamp
import numpy as np
import essentia
import essentia.standard as ess
import essentia.streaming as esstr
from scipy import interpolate
import scipy
import os
import string
import subprocess

import math
from bisect import bisect_left, bisect_right
import random
from pychord_tools import symbolic_analysis

import simmusic
from simmusic import segmentation
from simmusic.containers import Recording
from simmusic import utilities as utils
from simmusic import constants
from simmusic import utilities_io
from simmusic.onset_detection import GuitarOnsetDetector
from simmusic.chroma_labels import GuitarLabelTranslator
from simmusic.dtw import dtw

from pychord_tools.models import load_model
from pychord_tools.third_party import NNLSChromaEstimator
from pychord_tools.low_level_features import UidExtractor, ChromaEstimator, SegmentChromaEstimator, \
    AnnotatedBeatChromaEstimator, AnnotatedChromaSegments
from pychord_tools.low_level_features import audio_duration, smooth

SAMPLING_FREQUENCY = 44100.0
MINIMUM_FREQUENCY = 60


def extract_pitch_melodia(audio_file, pitch_file, hop_size=1024, frame_size=2048, bin_resolution=1,
                          guess_unvoiced=True, voicing_tolerance=0.2, max_freq=20000):
    """
    Wrapper function to extract pitch using Melodia algorithm in essentia

    Parameters
    ----------
    audio_file : str
        Name of the audio file
    pitch_file : str
        Name of the output pitch file
    hop_size : int
        Hop size in samples (assuming sampling rate of 44100 Hz)
    frame_size : int
        Frame size in samples (assuming sampling rate of 44100 Hz)
    bin_resolution : int
        Bin resolution of pitch in Cents scale
    guess_unvoiced : bool
        Whether or not to detect unvoiced regions
    voicing_tolerance : float
        See original documentation of this method in Essentia for this parameter
    max_freq : float
        Maximum frequency to be considered in the calculation of salience function

    Returns
    -------
    success : bool
        Success of the operation

    """
    try:
        fs = SAMPLING_FREQUENCY
        audio = ess.MonoLoader(filename=audio_file, sampleRate=fs)()
        melodia_obj = ess.PredominantPitchMelodia(hopSize=hop_size,
                                                  frameSize=frame_size,
                                                  binResolution=bin_resolution,
                                                  guessUnvoiced=guess_unvoiced,
                                                  voicingTolerance=voicing_tolerance,
                                                  maxFrequency=max_freq, minFrequency=MINIMUM_FREQUENCY)
        pitch = melodia_obj(audio)[0]
        # generating time stamps (because its equally hopped)
        t_stamps = np.array(np.arange(pitch.size)) * np.float(hop_size) / fs
        dump = np.array([t_stamps, pitch]).transpose()
        np.savetxt(pitch_file, dump, delimiter="\t", fmt='%0.3f')
    except Exception as e:
        raise e
    return True


def extract_melody_timing_features(pitch_file_std, trans_file, auto_pitch_shift=False, resize_length=None,
                                   ref_freq=constants.REF55HZ, min_segment_duration=0.1, filter_corner_segments=True,
                                   no_extraction=False,
                                   preproc_std={'func': None, 'params': None, 'feature': None}):
    """
    This function extracts several pitch and timing related features from the pitch track of a performance (imitation).

    Parameters
    ----------
    pitch_file_std : str
        Pitch file for a student's performance
    trans_file : str
        File that contains transcription of the reference lesson
    auto_pitch_shift : bool
        Whether or not try to octave shift reference pitch (automatically, if needed)
    resize_length : int (Optional)
        If given it resizes both student's and a teacher's pitch track to this many number of samples
    ref_freq : float (Optional)
        reference frequency to be considered in the analysis.
    min_segment_duration : float
        Minimum duration of a valid segment (used for filtering short segments), in Seconds
    filter_corner_segments : bool
        Whether or not to filter corner short segments. The threshold used is `min_segment_duration'
    preproc_std : dict
        Pre-processing parameters, If None, do nothing, otherwise, pass the feature from the function object specified.

    Returns
    -------
    features : dict
        Dictionary of different features

    """

    mode = 'dialogue'
    features = {}

    # Creating recording objects for both reference and a student's imitation
    ref_rec = Recording('reference', mode)
    std_rec = Recording('imitation', mode)

    # loading pitch data
    if no_extraction:
        std_rec.load_pitch_data_from_previous_extraction(pitch_file_std)
    else:
        std_rec.load_pitch_data(pitch_file_std)

    # loading transcription data
    ref_rec.load_trans_data(trans_file)

    # Converting pitch from Hz to Cents (scale)
    success = std_rec.convert_pitch_hz2cents(tonic=ref_freq, threshold=constants.PITCH_MIN_VOCAL,
                                             invalid_val=constants.INVALID_PITCH_CENTS)

    # preprocessing the student's pitch (cents) array (also used for data augmentation)
    if preproc_std['func'] is not None:
        if preproc_std['feature'] == 'cents':
            std_rec.feature.cents = preproc_std['func'](std_rec.feature.cents, **preproc_std['params'])

    if not success:
        print("Problems converting pitch from Hz to cents...")
        raise RuntimeError

    # computing ideal (step kind) pitch from the transcription (notes)
    # We assume that we generate pitch track with the same hop as it is in student's pitch (it has several adv.)
    hop_size = utils.compute_hopsize(std_rec.feature.time)
    track_time = ref_rec.segments[-1].notes[-1].end
    pitch_ideal = utilities_io.create_pitch_track_from_notes(ref_rec, hop_size, track_time)
    pitch_ideal.convert_pitch_hz2cents(tonic=ref_freq, threshold=constants.PITCH_MIN_VOCAL,
                                       invalid_val=constants.INVALID_PITCH_CENTS)
    ref_rec.feature = copy.deepcopy(pitch_ideal)

    # Sometimes student's sing in diff octave but same chroma, so we have to octave shift the pitch somehow
    mean_pitch_ref = np.mean(ref_rec.feature.cents[ref_rec.feature.cents > constants.INVALID_PITCH_CENTS])
    mean_pitch_std = np.mean(std_rec.feature.cents[std_rec.feature.cents > constants.INVALID_PITCH_CENTS])
    shift_factor = mean_pitch_std - mean_pitch_ref
    if auto_pitch_shift and shift_factor <= -400:
        ref_rec.feature.cents[ref_rec.feature.cents > constants.INVALID_PITCH_CENTS] -= 1200.0

    # TODO: we have note yet seen a scenario where the singing is octave above the reference pitch, if that's the case
    # make this pitch shift operation more generic by allowing + or - 1200 depending upon the shift_factor variable.

    # Computing segment and note indexes
    ref_rec.compute_segment_indexes()
    ref_rec.compute_note_indexes()

    # Computing the active pitch region in the student's pitch
    try:
        seg_obj = segmentation.segment_active_pitch_region(std_rec.feature.pitch, std_rec.feature.time,
                                                           filter_corner_segments, min_segment_duration)
    except Exception as e:
        raise e

    std_rec.segments.append(seg_obj)
    std_rec.compute_segment_indexes()

    # we delete the the silence regions from the pitch (Cents) track
    # This processing is more meaningful in dialogue mode of performance, in singalong mode, this might not be needed.
    ref_series = ref_rec.feature.cents[ref_rec.segments[0].start_ind:ref_rec.segments[0].end_ind]
    std_series = std_rec.feature.cents[std_rec.segments[0].start_ind:std_rec.segments[0].end_ind]
    ref_series_sil_rem = np.delete(ref_series, np.where(ref_series <= constants.INVALID_PITCH_CENTS)[0])
    std_series_sil_rem = np.delete(std_series, np.where(std_series <= constants.INVALID_PITCH_CENTS)[0])

    # Only this features needs to be computed here
    features['va_ratio'] = ref_series_sil_rem.size / float(std_series_sil_rem.size)

    if resize_length is not None:
        # there is a good reason to choose zero kind of interpolation, we do not want to be creating intermediate freqs
        # Resize_length should ideally vary with the exercise, it need not be same for all the exercises. If its varied
        # across exercises, see how does it affects the learning process. Features might have some relation with this
        # length, do not blindly change this value without proper analysis. Currently the value set is working fine for
        # exercises in the MAST dataset, they are not very long ones.
        ref_interp = interpolate.interp1d(np.arange(0, ref_series_sil_rem.size), ref_series_sil_rem, kind='zero')
        ref_series_sil_rem = ref_interp(np.linspace(0, ref_series_sil_rem.size - 1, resize_length))

        std_interp = interpolate.interp1d(np.arange(std_series_sil_rem.size), std_series_sil_rem, kind='zero')
        std_series_sil_rem = std_interp(np.linspace(0, std_series_sil_rem.size - 1, resize_length))

    dtw_output = dtw.dtw1d_GLS(ref_series_sil_rem,
                               std_series_sil_rem,
                               distType=0, hasGlobalConst=0, globalType=0, bandwidth=1, initCostMtx=1, reuseCostMtx=0,
                               delStep=0, moveStep=1, diagStep=1, initFirstCol=0, isSubsequence=0)

    # taking a point to point different in pitch (Cents) considering aligned path
    diff_series = np.abs(ref_series_sil_rem[dtw_output[2][0]] - std_series_sil_rem[dtw_output[2][1]])

    # ---------------------------
    # Extracting overall features
    # ---------------------------
    # pitch features - aggregated
    bins_abs_diff = 3000 * np.power(np.arange(0, 1.05, 0.05), 4)

    features['dtw_cost'] = dtw_output[0]
    features['dtw_norm_cost'] = dtw_output[0] / float(dtw_output[1])
    features['mean_diff'] = np.mean(diff_series)
    features['std_diff'] = np.std(diff_series)
    features['com_diff_dist'] = utils.center_of_mass_1d(np.histogram(diff_series, bins=bins_abs_diff)[0])
    features['diff_0_20'] = np.where((diff_series < 20) & (diff_series > 0))[0].size / float(diff_series.size)
    features['diff_20_50'] = np.where((diff_series < 50) & (diff_series >= 20))[0].size / float(diff_series.size)
    features['diff_50_100'] = np.where((diff_series < 100) & (diff_series >= 50))[0].size / float(diff_series.size)
    features['diff_100_Inf'] = np.where(diff_series >= 100)[0].size / float(diff_series.size)

    # pitch features - extremes
    extreme_margin = 0.25  # considering only top 10 %
    percentile = (1 - extreme_margin)
    ext_diff = np.sort(diff_series)[int(percentile * diff_series.size):]
    features['mean_diff_ext'] = np.mean(ext_diff)
    features['diff_ext'] = np.percentile(diff_series, 100 * percentile)
    features['com_diff_dist_ext'] = utils.center_of_mass_1d(np.histogram(ext_diff, bins=bins_abs_diff)[0])

    # timing/duration features - aggregated
    time_diffs = np.abs(dtw_output[2][0] - dtw_output[2][1])
    features['path_len_to_ref_len'] = (dtw_output[2][0].size - ref_series_sil_rem.size) / float(ref_series_sil_rem.size)
    features['mean_path_dev'] = np.mean(time_diffs)

    # timing duration features - extremes
    features['path_dev_ext'] = np.percentile(time_diffs, 100 * percentile)
    features['mean_path_dev_ext'] = np.mean(np.sort(time_diffs)[int(percentile * time_diffs.size):])

    # ----------------------------
    # Extracting per note features
    # ----------------------------
    # First we need to extract indexes corresponding to the boundaries of the flat notes in the references
    freq_levels = np.unique(ref_series_sil_rem)
    mean_diff_notes = []
    ratio = []
    derivative_mean_notes = []
    for freq_level in freq_levels:
        inds = np.sort(np.where(ref_series_sil_rem == freq_level)[0])
        for segment in utils.group_indices(inds):
            ind_start = np.argmin(np.abs(dtw_output[2][0] - segment[0]))
            ind_end = np.argmin(np.abs(dtw_output[2][0] - segment[1]))
            ratio.append(np.abs((ind_end - ind_start) - float(segment[1] - segment[0])))
            if std_series_sil_rem[dtw_output[2][1][ind_start:ind_end]].size > 0:
                mean_diff_notes.append(
                    np.mean(np.abs(freq_level - std_series_sil_rem[dtw_output[2][1][ind_start:ind_end]])))
                derivative_mean_notes.append(
                    np.max(np.abs(np.diff(std_series_sil_rem[dtw_output[2][1][ind_start:ind_end]], 1))))

    features['mean_note_diff'] = np.mean(mean_diff_notes)
    features['max_note_diff'] = np.max(mean_diff_notes)
    features['ratio_note_dur'] = np.std(ratio)
    features['derivative_mean_notes'] = np.max(derivative_mean_notes)

    return features


def convert_syl_2_letters(syllables0, syllables1):
    """
    convert syllable lists to letter string
    :param syllables0:
    :param syllables1:
    :return:
    """
    dict_letters2syl = {}
    dict_syl2letters = {}
    ascii_letters = string.ascii_letters
    for ii, syl in enumerate(list(set(syllables0+syllables1))):
        dict_letters2syl[ascii_letters[ii]] = syl
        dict_syl2letters[syl] = ascii_letters[ii]

    syllables0_converted = ''.join([dict_syl2letters[syl] for syl in syllables0])
    syllables1_converted = ''.join([dict_syl2letters[syl] for syl in syllables1])
    return syllables0_converted, syllables1_converted, dict_letters2syl


def convert_letters_2_num_list(syllables_letters_aligned):
    """
    Helper function to convert letters to a numeric list
    :param syllables_letters_aligned:
    :return:
    """
    counter = 0
    number_syl = [None] * len(syllables_letters_aligned)
    for ii, s in enumerate(syllables_letters_aligned):
        if s != "-":
            number_syl[ii] = counter
            counter += 1
    return number_syl


def syllable_sequence_recognition(kaldi_root,
                                  input_audio_file_name,
                                  output_file_path,
                                  segment_start):
    """
    Syllable recognition
    :param kaldi_root: kaldi root path
    :param input_file_path: the input path containing the recording .wav
    :param input_audio_file_name: wav filename without extension
    :param output_file_path: everything intermediate output goes here
    :param segment_start: utterance segment start time
    :return: syllable sequence list
    """

    recognition_command = os.path.join("/opt/kaldi", "scripts", "recognition_run.sh")
    # current_path = os.path.dirname(os.path.realpath(__file__))
    # recognition_command = os.path.join(current_path, "kaldi_script", "recognition_run.sh")

    utilities_io.create_kaldi_test_data(input_audio_file_path=os.path.join(output_file_path, "audio"),
                                        input_audio_filename=input_audio_file_name+".wav",
                                        output_data_path=os.path.join(output_file_path, "data", input_audio_file_name),
                                        segment_start=segment_start)

    # call the bash script to calculate the syllable transcription
    subprocess.call([recognition_command, kaldi_root,
                     input_audio_file_name,
                     output_file_path])

    # output alignment file
    alignment_file = os.path.join(output_file_path, "mono", input_audio_file_name, "ctm", "ctm")

    syllables = utilities_io.parse_kaldi_alignment_transcription(alignment_file,
                                                                 segment_start)

    return syllables


def syllable_sequence_alignment(kaldi_root,
                                input_audio_file_name,
                                output_file_path,
                                transcription_str,
                                segment_start):
    """
    Alignment .wav with transcription
    :param kaldi_root: kaldi root path
    :param input_audio_file_name: wav filename without extension
    :param output_file_path: everything intermediate output goes here
    :param transcription_str: transcription string
    :param segment_start: utterance segment start time
    :return: syllable sequence list
    """

    recognition_command = os.path.join("/opt/kaldi", "scripts", "alignment_run.sh")
    # current_path = os.path.dirname(os.path.realpath(__file__))
    # recognition_command = os.path.join(current_path, "kaldi_script", "alignment_run.sh")

    utilities_io.create_kaldi_test_data(input_audio_file_path=os.path.join(output_file_path, "audio"),
                                        input_audio_filename=input_audio_file_name+".wav",
                                        output_data_path=os.path.join(output_file_path, "data", input_audio_file_name),
                                        segment_start=segment_start,
                                        text_file=True,
                                        text_content=transcription_str)

    # call the bash script to calculate the syllable transcription
    subprocess.call([recognition_command, kaldi_root,
                     input_audio_file_name,
                     output_file_path])

    # output alignment file
    alignment_file = os.path.join(output_file_path, "mono_ali", input_audio_file_name, "ctm.1")

    syllables = utilities_io.parse_kaldi_alignment_transcription(alignment_file, segment_start)

    return syllables


def get_pyin_pitchtrack(audio, fs, tonic=440.0):
    """
    Compute the pitchtrack by using pyin
    :param audio:
    :param fs:
    :param tonic:
    :return:
    """
    params = {"outputunvoiced": 2.0}
    data = vamp.collect(audio, fs, "pyin:pyin", output='smoothedpitchtrack', parameters=params)
    hop, melody = data['vector']

    melody_cents = utils.hz2pitch(hertz=melody, reference=tonic)
    # melody_cents = 12 * np.log(melody / tonic) / np.log(2.0) + 49

    timestamps = 8 * 256 / float(fs) + np.arange(len(melody)) * (256 / float(fs))
    return timestamps, melody_cents


def segment_pitch_track_boundaries(time_stamps, pitch_track, boundary):
    """
    Segment the pitch track according to the boundary onset and offset
    :param time_stamps: pitch track time stamps
    :param pitch_track: pitch track in cents
    :param boundary: [onset time, offset time]
    :return: segment_pitch track
    """
    ind_onset = np.argmin(np.abs(time_stamps - boundary[0]))
    ind_offset = np.argmin(np.abs(time_stamps - boundary[1]))
    segment_time_stamps = time_stamps[ind_onset:ind_offset]
    segment_pitch_track = pitch_track[ind_onset:ind_offset]
    return segment_time_stamps, segment_pitch_track


def segments_median_difference_mean(median_syllable_ref, median_syllable_stu):
    """
    Calculate the mean of the median difference of two pitch track segments
    :param median_syllable_ref:
    :param median_syllable_stu:
    :return:
    """
    assert len(median_syllable_ref) == len(median_syllable_stu)

    mean_median_diff = np.mean(median_syllable_ref - median_syllable_stu)
    return mean_median_diff


def get_median_syllable_pitch(time_stamps, pitch_track, boundaries):
    """
    Loop through all the boundary, retrieve the pitch segment and calculate the median pitch
    :param time_stamps:
    :param pitch_track:
    :param boundaries:
    :return:
    """
    median_syllable_pitch = []
    for b in boundaries:
        segment_time_stamps, segment_pitch_track = segment_pitch_track_boundaries(time_stamps=time_stamps,
                                                                                  pitch_track=pitch_track,
                                                                                  boundary=b)
        median_syllable_pitch.append(np.median(segment_pitch_track[~np.isnan(segment_pitch_track)]))
    return np.array(median_syllable_pitch)


def get_segment_pitch_tracks(time_stamps, pitch_track, boundaries):
    segment_pitch_tracks = []
    for b in boundaries:
        segment_time_stamps, segment_pitch_track = segment_pitch_track_boundaries(time_stamps=time_stamps,
                                                                                  pitch_track=pitch_track,
                                                                                  boundary=b)
        segment_pitch_tracks.append([segment_time_stamps, segment_pitch_track])
    return segment_pitch_tracks


def check_octave_difference(median_syllable_ref,
                            median_syllable_stu):
    """
    Check if octave difference exists between ref and student pitch tracks
    :param median_syllable_ref:
    :param median_syllable_stu:
    :return: bools, octave_diff, ref_higher
    """
    mean_median_diff = segments_median_difference_mean(median_syllable_ref=median_syllable_ref,
                                                       median_syllable_stu=median_syllable_stu)
    octave_diff = False
    ref_higher = False
    if 10.5 < np.abs(mean_median_diff) < 13.5:
        octave_diff = 1
    elif 10.5 < np.abs(mean_median_diff) - 12.0 < 13.5:
        octave_diff = 2
    elif 10.5 < np.abs(mean_median_diff) - 24.0 < 13.5:
        octave_diff = 3
    else:
        pass

    if mean_median_diff > 0:
        ref_higher = True

    return octave_diff, ref_higher


def median_octave_correction(median_syllable_ref, octave_diff, ref_higher):
    """
    Correct median syllable pitch values of the student's pitch track
    :param median_syllable_ref:
    :param octave_diff: integer
    :param ref_higher:
    :return:
    """
    if ref_higher:
        return median_syllable_ref - 12.0 * octave_diff
    else:
        return median_syllable_ref + 12.0 * octave_diff


###########################################################################
# Tuning quality features.
###########################################################################

def load_spectrum_and_peaks(filename, hop_size = 2048, frame_size = 16384, min_frequency=40, max_frequency=1500, max_peaks=10000):
    """
    Calculates spectrogram and spectral peaks for the given audio file.

    :param filename: file to process.
    :param hop_size: hop size
    :param frame_size: frame size for STFT spectrum estimation
    :param min_frequency: min peak frequency
    :param max_frequency: max peak grequency
    :param max_peaks: maximum number of (the most prominent) peaks per frame
    :return:
    """
    loader = esstr.MonoLoader(filename=filename)
    framecutter = esstr.FrameCutter(hopSize=hop_size, frameSize=frame_size)
    windowing = esstr.Windowing(type="blackmanharris92", size=frame_size)
    spectrum = esstr.Spectrum()
    spectralpeaks = esstr.SpectralPeaks(orderBy="frequency",
                                        magnitudeThreshold=1e-05,
                                        minFrequency=min_frequency,
                                        maxFrequency=max_frequency,
                                        maxPeaks=max_peaks)
    tuning = esstr.TuningFrequency()
    pool = essentia.Pool()

    # connect algorithms together
    loader.audio >> framecutter.signal
    framecutter.frame >> windowing.frame >> spectrum.frame
    spectrum.spectrum >> spectralpeaks.spectrum
    spectrum.spectrum >> (pool, 'spectrum.magnitude')
    spectralpeaks.magnitudes >> tuning.magnitudes
    spectralpeaks.frequencies >> tuning.frequencies
    spectralpeaks.magnitudes >> (pool, 'peaks.magnitude')
    spectralpeaks.frequencies >> (pool, 'peaks.frequencies')
    tuning.tuningFrequency >> (pool, 'tonal.tuningFrequency')
    tuning.tuningCents >> (pool, 'tonal.tuningCents')
    essentia.run(loader)

    return np.array(pool['spectrum.magnitude']), np.array(pool['peaks.frequencies']), np.array(pool['peaks.magnitude'])


def interval_deviations_from_equal_tempered(frame_peaks, frame_magnitudes, peak_limit=10):
    """
    Calculates intervals between all peak_limit most prominent spectral peaks and
    their deviation from equal temperament grid.

    :param frame_peaks: peaks frequencies
    :param frame_magnitudes: peaks magnintudes
    :param peak_limit: number of the most prominent peaks to consider
    :return:
    """
    yc = sorted(zip(frame_peaks, frame_magnitudes), key=lambda x: x[1], reverse=True)
    l = min(peak_limit, len(yc))
    if (l > 0):
        (y, c) = zip(*yc)
        local_intervals = []
        for j in range(l):
            for k in range(j + 1, l):
                current = abs((math.log(y[j]) - math.log(y[k])) / math.log(2.0))
                # less than octave + ~quartertone (why???)
                if (current <= 1.04):
                    local_intervals.append(current)
        local_intervals = np.array(local_intervals)
        # intervals = intervals[intervals <= 1.04]
        devs = np.array([int(i * 1200) % 100 for i in local_intervals])
        devs[devs > 50] = devs[devs > 50] - 100
        return devs / 50.0
    else:
        return np.empty(0)


def sample_deviations(peaks, mags, max_frame_peaks, points = 500):
    """
    Sample entities from the set of deviations of the intervals between all peaks
    from equal temperament grid.

    :param peaks: 2D array of prominent peaks (1D array for each frame)
    :param mags: 2D array of magnitutdes the prominent peaks (1D array for each frame)
    :param max_frame_peaks: maximum of the peaks form each frame to keep.
    :param points: number of random points (frames) to sample
    :return:
    """
    all_peaks = []
    all_mags = []
    indices = range(len(peaks))
    #print("indices: ", len(indices))
    if (len(indices) > points):
        indices = random.sample(indices, points)
    total_len = 0
    for i in indices:
        # choose max_peaks for each point.
        yc = sorted(zip(peaks[i], mags[i]), key=lambda x: x[1], reverse=True)
        total_len += len(yc)
        l = min(max_frame_peaks, len(yc))
        if (l > 0):
            (y, c) = zip(*yc)
            all_peaks.extend(y[:l])
            all_mags.extend(c[:l])

    #print("all selected: ", len(all_peaks))
    #print("from: ", total_len)
    all_delta = interval_deviations_from_equal_tempered(all_peaks, all_mags, peak_limit=len(all_peaks))
    return np.array(all_delta)

def deviation_statistics(prefix, deviations, histogram_max):
    features = {}
    deviations = np.array(deviations)
    d2 = deviations * deviations
    variance0 = np.mean(d2)
    d4 = d2 * d2
    kurtosis0 = np.mean(d4) / (variance0 * variance0)
    std_sixth_moment0 = np.mean(d4 * d2) / (variance0 * variance0 * variance0)
    features[prefix + 'variance0'] = variance0
    features[prefix + 'kurtosis0'] = kurtosis0
    features[prefix + 'std_6th_moment0'] = std_sixth_moment0
    # histogram. Just the adoptation of Sunkalp's code.
    if (deviations.size > 0):
        point20 = 0.2 * histogram_max
        point50 = 0.5 * histogram_max
        point100 = histogram_max
        features[prefix + 'diff_0_20'] = np.where(deviations < point20)[0].size / float(deviations.size)
        features[prefix + 'diff_20_50'] = np.where((deviations < point50) & (deviations >= point20))[0].size / float(deviations.size)
        features[prefix + 'diff_50_100'] = np.where((deviations < point100) & (deviations >= point50))[0].size / float(deviations.size)
        features[prefix + 'diff_100_Inf'] = np.where(deviations >= point100)[0].size / float(deviations.size)
    else:
        features[prefix + 'diff_0_20'] = math.nan
        features[prefix + 'diff_20_50'] = math.nan
        features[prefix + 'diff_50_100'] = math.nan
        features[prefix + 'diff_100_Inf'] = math.nan

    # extremes
    extreme_margin = 0.25  # considering only top 25 %
    percentile = (1 - extreme_margin)
    ext_diff = np.sort(deviations)[int(percentile * deviations.size):]
    features[prefix + 'mean_diff_ext'] = np.mean(ext_diff)
    if len(deviations) > 0:
        features[prefix + 'diff_ext'] = np.percentile(deviations, 100 * percentile)
    else:
        features[prefix + 'diff_ext'] = 0
    return features


def calculate_statistics_for_deviation_from_equal_temperament(
        filename, max_frame_peaks = 10, min_frequency=100, max_frequency=3000):
    """
    Calculates 2nd moment and 4th moment for peaks deviations from the
    equal temperament grid.

    :param filename: audio file to proceed
    :param max_frame_peaks: max number of peaks to keep from each frame
    :param min_frequency: peak min frequency
    :param max_frequency: peak max frequency
    :return: (2nd moment, 4th moment)
    """
    spectra, peaks, mags = load_spectrum_and_peaks(filename, min_frequency=min_frequency, max_frequency=max_frequency)
    s = np.sum(spectra * spectra, axis = 1)
    # naive noise filtering
    threshold = np.percentile(s, 95)/10.0
    indices = s > threshold
    all_delta = sample_deviations(peaks[indices], mags[indices], max_frame_peaks = max_frame_peaks)
    return deviation_statistics('', all_delta, 0.6)

###########################################################################
# Timing features.
###########################################################################

def match_events(events, onsets, f_measure_threshold):
    # In case of performance issues for big piecs,
    # we could try to use simpler/faster local algorithm.
    big_distance = 10 ** 6
    m = scipy.spatial.distance_matrix([[x] for x in events], [[x] for x in onsets])
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(m)
    result = []

    for (x, y) in zip(row_ind, col_ind):
        if abs(events[x] - onsets[y]) < f_measure_threshold:
            result.append((x, y))
    return result

def f_measure(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0
    return 2.0 * precision*recall / (precision + recall)

def timing_features(
        anno_file,
        audio_file,
        max_spectral_centroid=3500,
        onset_threshold=2,
        series_delta=0.22,
        sample_rate=44100,
        f_measure_threshold=0.25):
    bars, beats, events, chords = symbolic_analysis.rhythm_for_file(anno_file)
    beats =  np.array(beats)
    events = np.array(events)
    chords = np.array(chords)

    is_defined = [x[0] != 'N' for x in chords]
    chords = chords[is_defined]
    events = events[is_defined]

    audio = ess.MonoLoader(filename=audio_file)()
    duration = float(len(audio)) / sample_rate
    half_ibi = (beats[1:] - beats[:-1]).mean() / 2
    start = max(events[0] - half_ibi, 0)
    end = min(events[-1] + half_ibi, duration)
    onsets = np.array(GuitarOnsetDetector(
        max_spectral_centroid=max_spectral_centroid,
        onset_threshold=onset_threshold,
        series_delta=series_delta).predict(audio_file, start, end))

    devs = []
    segments = []
    events_hits = set()
    onsets_hits = set()
    if len(events) > 0 and len(onsets) > 0:
        matching = match_events(
            events,
            onsets,
            f_measure_threshold)
        for m in matching:
            dev = onsets[m[1]] - events[m[0]]
            devs.append(dev)
            if dev > 0:
                segments.append((events[m[0]], onsets[m[1]]))
            else:
                segments.append((onsets[m[1]], events[m[0]]))
            events_hits.add(m[0])
            onsets_hits.add(m[1])
        precision = float(len(matching)) / len(onsets)
        recall = float(len(matching)) / len(events)
        f_measure_value = f_measure(precision, recall)
    else:
        precision = 0
        recall = 0
        f_measure_value = 0

    misses = [events[x] for x in range(len(events)) if x not in events_hits]
    excessives = [onsets[x] for x in range(len(onsets)) if x not in onsets_hits]
    features = {
        'events': events,
        'beats': beats,
        'bars': bars,
        'duration': duration,
        'half_ibi': half_ibi,
        'start': start,
        'end': end,
        'onsets': onsets,
        'devs': np.array(devs),
        'segments': segments,
        'f_measure' : f_measure_value,
        'precision': precision,
        'recall': recall,
        'misses': np.array(misses),
        'excessives': np.array(excessives)
    }
    return features


def timing_statistics(devs, threshold = 0.05, histogram_max = 0.3):
    features = {}
    abs_devs = np.abs(devs)
    thresholded = abs_devs - threshold
    thresholded[thresholded < 0] = 0
    features["thresholded"] = thresholded.tolist()
    features.update(deviation_statistics("dev.", abs_devs, histogram_max))
    features.update(deviation_statistics("thresholded.", thresholded, histogram_max))
    return features

###########################################################################
# Chroma features.
###########################################################################


def my_nnls_chroma_from_audio(uid, audio_path_extractor, sample_rate=44100, step_size=2048):
    # TODO: remove
    #path_db.set_audio_path(uid, uid)
    audio = essentia.standard.MonoLoader(filename=audio_path_extractor.audio_path_name(uid), sampleRate=sample_rate)()
    # estimate audio duration just for caching purposes:
    audio_duration(uid, sample_rate=sample_rate, audio_samples=audio, audio_path_extractor=audio_path_extractor)

    stepsize, chroma = vamp.collect(
        audio, sample_rate, "nnls-chroma:nnls-chroma", output="chroma", step_size=step_size,
        parameters={'useNNLS':1, 'whitening':1, 's':0.9, 'rollon':5, 'tuningmode':0, 'chromanormalize':0})["matrix"]
    # roll from 'A' based to 'C' based
    chroma = np.roll(chroma, shift=-3, axis=1)
    return chroma


class MyNNLSChromaEstimator(ChromaEstimator):
    def __init__(self, hop_size=2048, sample_rate=44100):
        super().__init__(16384, hop_size, sample_rate)

    def estimate_chroma(self, uid):
        return my_nnls_chroma_from_audio(uid, self.audio_path_extractor, self.sample_rate, self.hop_size)


class AdaptiveChromaEstimator(SegmentChromaEstimator):
    def __init__(self, frame_size=16384, hop_size=2048, sample_rate=44100, smoothing_time=0.6):
        super().__init__(frame_size, hop_size, sample_rate)
        self.smoothingTime = smoothing_time

    def fill(self, beats, durations, chroma, smoothed_chromas):
        for i in range(len(beats)):
            s = int(float(beats[i]) *
                    self.sample_rate / self.hop_size)
            d = int(float(durations[i]) *
                    self.sample_rate / self.hop_size)
            d = min(d, len(chroma) - s)
            w = eval('np.hanning(2*d)')
            w = w[d:] / np.sum(w[d:])
            w = np.reshape(w, (1, d))
            c = chroma[s:s+d]
            smoothed_chromas[i] = np.dot(w, c)

    def fill_segments_with_chroma(self, segments, chroma):
        segments.chromas = np.zeros((len(segments.start_times), 12), dtype='float32')
        self.fill(segments.start_times, segments.durations, chroma, segments.chromas)

    def get_chroma_by_beats(self, beats, chroma):
        # TODO: durations, especially for last beat.
        res = np.zeros((len(beats), 12), dtype='float32')
        durations_known = beats[1:] - beats[:-1]
        self.fill(beats, np.concatenate((durations_known, [np.mean(durations_known)])), chroma, res)
        return res


class ConstUIDExtractor(UidExtractor):
    def __init__(self, path):
        self.path = path

    def uid(self, annotation_file_name):
        return self.path


def estimate_segment_scores(
        annotation_filename,
        student_filename,
        chroma_pattern_model,
        chroma_estimator=NNLSChromaEstimator()):
    """
    Estimates averaged segments chroma scores accoriding to given annotation.

    :param annotation_filename: annotation file
    :param student_filename: Name of the performance audio file of a student
    :param chroma_pattern_model: ChromaPatternModel
        model for estimating chords quality
    :param chroma_estimator: chorma estimator
    :return:
    """
    chromaEstimator = AnnotatedBeatChromaEstimator(
        chroma_estimator=chroma_estimator,
        segment_chroma_estimator=AdaptiveChromaEstimator(),
        label_translator=GuitarLabelTranslator(),
        uid_extractor=ConstUIDExtractor(student_filename),
        roll_to_c_root=False)
    realSegments = chromaEstimator.load_chromas_for_annotation_file(annotation_filename)
    # filter out unclassified:
    is_defined = [x != 'unclassified' for x in realSegments.kinds]
    realSegments = AnnotatedChromaSegments(
        realSegments.labels[is_defined],
        realSegments.pitches[is_defined],
        realSegments.kinds[is_defined],
        realSegments.chromas[is_defined],
        realSegments.uids[is_defined],
        realSegments.start_times[is_defined],
        realSegments.durations[is_defined])

    #predicted, plu = chromaPatternModel.predict(realSegments.chromas)
    nlu = chroma_pattern_model.log_utilities_given_sequence(
        chromas=realSegments.chromas, pitched_patterns=realSegments.pitched_patterns(), normalize=True)
    lu = chroma_pattern_model.log_utilities_given_sequence(
        chromas=realSegments.chromas, pitched_patterns=realSegments.pitched_patterns(), normalize=False)

    return lu, nlu, realSegments


def picking_chroma_scores(anno_file, audio_file):
    m = load_model(os.path.join(simmusic.__path__[0], 'extractors/chord_models/solo_picking_pdf.pkl'))
    chroma_estimator1 = NNLSChromaEstimator()
    chroma_estimator2 = MyNNLSChromaEstimator()
    lu1, nlu1, real_segments1 = estimate_segment_scores(anno_file, audio_file, m, chroma_estimator1)
    predicted1, plu1 = m.predict(real_segments1.chromas)
    lu2, nlu2, real_segments2 = estimate_segment_scores(anno_file, audio_file, m, chroma_estimator2)
    predicted2, plu2 = m.predict(real_segments1.chromas)

    nlus = np.array([nlu1, nlu2])
    nlu = np.max(nlus, axis=0)

    predicts = np.array([predicted1, predicted2])
    #predicted = np.choose(np.argmax(nlus, axis=0), predicts)
    #print(predicted)

    return nlu


def strumming_chroma_scores(anno_file, audio_file):
    m = load_model(os.path.join(simmusic.__path__[0], 'extractors/chord_models/ind_pdf.pkl'))
    chroma_estimator = NNLSChromaEstimator()
    lu, nlu, real_segments = estimate_segment_scores(anno_file, audio_file, m, chroma_estimator)
    #predicted, plu = m.predict(real_segments.chromas)
    #print(predicted)
    return nlu


def chroma_statistics(chroma_scores, histogram_max = 6):
    return deviation_statistics("", -chroma_scores, histogram_max=histogram_max)
