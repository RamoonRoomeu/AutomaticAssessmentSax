"""
This file contains all the functions that perform scoring (evaluation) given the alignment/segmentation has happened.
"""
import numpy as np
from simmusic import constants
from simmusic import distance_metrics as dm
from simmusic.dtw import dtw


def score_flat_note_rule_based_1(ref_rec, std_rec, alignment, ref_freq):
    """Score flat note lesson based on set of rules (logic 1)

    Parameters
    ----------
    ref_rec : :class:`container.Recording`
        Reference recording object
    std_rec : :class:`container.Recording`
        Student's recording object
    alignment : list
        Alignment between the two recordings
    ref_freq : double
        Reference frequency for performing Hz to Cents conversion

    Returns
    -------
    feedback : list
        Feedback (scores) for each note. List of dict

    """
    # Converting frequency of notes from Hz to Cent (scale)
    ref_rec.convert_notes_hz2cents(tonic=ref_freq)

    feedback = []
    for align in alignment:
        seg_std = align['segments'][0]
        seg_ref = align['segments'][1]
        note_scores = []
        for note in align['notes']:
            n_std = std_rec.segments[seg_std].notes[note[0]]
            n_ref = ref_rec.segments[seg_ref].notes[note[1]]
            if n_std.start_ind == constants.INVALID_INDEX or n_std.end_ind == constants.INVALID_INDEX:
                score = constants.INVALID_SCORE
            else:
                diff_cents = np.abs(std_rec.feature.cents[n_std.start_ind:n_std.end_ind + 1] - n_ref.cent)
                ind_hits = np.where(diff_cents < constants.SVAR_THRESHOLD_CENTS_MEDIUM)[0]
                n_hits = len(ind_hits)
                score = float(n_hits) / float(len(diff_cents))
            note_scores.append({'note': note[0], 'pitch_score': score})
        # TODO: type of the feedback (data structure) has to change, once we have all the use cases we can refactor it.
        feedback.append({'segment': seg_std, 'notes': note_scores})
    return feedback


def score_flat_note_rule_based_2(ref_rec, std_rec, alignment, ref_freq, threshold_array=None):
    """Score flat note lesson based on set of rules (logic 2)

    This function performs scoring based on a 3 tiered threshold. Sung pitch samples are divided into 3 bins depending
    on their proximity with the reference freqency. An weighted average of those #samples is considered for scoring.

    Parameters
    ----------
    ref_rec : :class:`container.Recording`
        Reference recording object
    std_rec : :class:`container.Recording`
        Student's recording object
    alignment: list
        Alignment between the two recordings
    ref_freq : double
        Reference frequency for performing Hz to Cents conversion
    threshold_array : list or np.ndarray, Optional
        Array that defines the 3 tier thresholds and relevant details for scoring notes

    Returns
    -------
    feedback : list
        Feedback (scores) for each note. List of dict

    """
    # Converting frequency of notes from Hz to Cent (scale)
    ref_rec.convert_notes_hz2cents(tonic=ref_freq)

    fraction_eval = 1.0
    wght_tight = 1.0
    wght_medium = 1.0
    wght_loose = 1.0
    if threshold_array is None:
        threshold_array = np.abs([0, constants.SVAR_THRESHOLD_CENTS_TIGHT, constants.SVAR_THRESHOLD_CENTS_MEDIUM,
                                  constants.SVAR_THRESHOLD_CENTS_LOOSE, np.abs(constants.INVALID_PITCH_CENTS)])
    feedback = []
    for align in alignment:
        seg_std = align['segments'][0]
        seg_ref = align['segments'][1]
        note_scores = []
        for note in align['notes']:
            n_std = std_rec.segments[seg_std].notes[note[0]]
            n_ref = ref_rec.segments[seg_ref].notes[note[1]]
            duration = n_std.end - n_std.start
            if duration <= 0.5:
                fraction_eval = 0.4
                wght_tight = 1.0
                wght_medium = 0.8
                wght_loose = 0.4
            elif duration <= 1.0:
                fraction_eval = 0.7
                wght_tight = 1
                wght_medium = 0.6
                wght_loose = .1
            elif duration <= 2:
                fraction_eval = 0.8
                wght_tight = 1
                wght_medium = 0.6
                wght_loose = .05
            elif 2.0 < duration:
                # TODO do the latest change in Riyaz code, to handle long notes if they are a part of a sequence!!
                fraction_eval = 0.9
                wght_tight = 1
                wght_medium = 0.4
                wght_loose = 0
            if n_std.start_ind == constants.INVALID_INDEX or n_std.end_ind == constants.INVALID_INDEX:
                score = constants.INVALID_SCORE
            else:
                diff_cents = np.abs(std_rec.feature.cents[n_std.start_ind:n_std.end_ind + 1] - n_ref.cent)
                diff_cents = np.sort(diff_cents)
                diff_cents_selected = diff_cents[:np.round(fraction_eval * len(diff_cents)).astype(np.int)]
                hist, bins = np.histogram(diff_cents_selected, bins=threshold_array)
                score = ((hist[0] * wght_tight) + (hist[1] * wght_medium) + (hist[2] * wght_loose)) / float(
                    len(diff_cents_selected))
            note_scores.append({'note': note[0], 'pitch_score': score})
        feedback.append({'segment': seg_std, 'notes': note_scores})
    return feedback


def score_phrases_rule_based(ref_rec, std_rec, alignment):
    """Score Phrase type lesson based on set of rules

    This function first aligns the student recording with the reference and then goes for a sample by sample distance.
    Finally the distance is mapped to a score. The scoring is done at a flat level

    Parameters
    ----------
    ref_rec : :class:`container.Recording`
        Reference recording object
    std_rec : :class:`container.Recording`
        Student's recording object
    alignment : list
        Alignment between the two recordings

    Returns
    -------
    feedback : list
        Feedback (scores) for each phrase. List of dict

    :note: Some improvements
        1) Consider flat and non flat regions, use a different logic to map distance -> score for these regions
        2) Perform DTW alignment using an improved version of the algorithm in which unvoiced regions do not result into
        snapping, instead during unvoiced region dtw path goes diagonal.
        3) Implement a local alignment method and perform alignment of only voiced regions (unlike global alignment in
        DTW)
        4) For more todos and improvements read insighs.txt resulted from Riyaz experiments.

    """
    feedback = []
    for align in alignment:
        seg_std = std_rec.segments[align['segments'][0]]
        seg_ref = ref_rec.segments[align['segments'][1]]

        # TODO: if the diff between two segments size is huge, decide what to do? partial mark, or zero!
        dist = compute_dissimilarity_phrase(ref_rec.feature.cents[seg_ref.start_ind: seg_ref.end_ind],
                                            std_rec.feature.cents[seg_std.start_ind: seg_std.end_ind])

        score = distance_to_score_mapping(dist)

        feedback.append({'segment': align['segments'][0], 'notes': [], 'overall': score})
    return feedback


def distance_to_score_mapping(distance):
    """ Maps distance to score

    This function maps distance value to a score. This mapping is highly contextual and might depend on the
    characteristics of the melodic shape and distance measure.

    Parameters
    ----------
    distance: double
        Distance computed between two melodic elements

    Returns
    -------
    score : double
        Score assigned for a distance

    :Note: In the future this mapping has to be learned and used contextually (several mapping functions?)

    """
    if distance < 500:
        score = 0.95
    elif distance < 1000:
        score = 0.8 + 0.15 * (1000 - distance) / 500
    elif distance < 2000:
        score = 0.6 + 0.2 * (2000 - distance) / 1000
    elif distance < 4000:
        score = 0.4 + 0.2 * (4000 - distance) / 2000
    elif distance < 8000:
        score = 0.2 + 0.2 * (8000 - distance) / 4000
    elif distance < 16000:
        score = 0.2 * (16000 - distance) / 8000
    else:
        score = max(0, 0.1 + 0.2 * (50000 - distance) / 15000)
    return score


def compute_distance_for_path(tseries1, tseries2, path):
    """ Computes distance along a alignment path.

     This function computes sample to sample distance given the DTW alignment path between two the time series.
     What makes it relevant is that we can include domain knowledge while calculating this distance, something that is
     not that easy to incorporate in the DTW or for that matter in any alignment algorithm. An example of this context
     could be using separate distance metric for flat and non-flat regions in time series.

    Parameters
    ----------
    tseries1 : array_like, np.ndarray
        Input time series (feature array, eg. pitch)
    tseries2 : array_like, np.ndarray
        Input time series (feature array, eg. pitch)
    path : array_like, np.ndarray (N,2)
        Alignment path between two time series

    Returns
    -------
    distance : double
        Aggregate sample to sample distance

    :note: Improvements; 1) Instead of Euclidean over the vector, try also manhattan distance
    """
    pairs = []
    for ii in range(path[0].shape[0]):
        if constants.INVALID_PITCH_CENTS != tseries1[path[0][ii]] and \
                        tseries2[path[1][ii]] != constants.INVALID_PITCH_CENTS:
            pairs.append([tseries1[path[0][ii]], tseries2[path[1][ii]]])
    return dm.euclidean_distance(np.array(pairs)[:, 0], np.array(pairs)[:, 1])


def compute_dissimilarity_phrase(tseries1, tseries2):
    """ Computes dissimilarity value between two time series (pitch tracks in our case)

    This function computes perceptual (musical) distance between two melodic time series

    Parameters
    ----------
    tseries1 : array_like, np.ndarray
        Input time series (feature array, eg. pitch)
    tseries2 : array_like, np.ndarray
        Input time series (feature array, eg. pitch)

    Returns
    -------
    distance : double
        Perceptual distance between two melodic time series

    """
    output = dtw.dtw1d_GLS(tseries1, tseries2,
                           distType=0, hasGlobalConst=0, globalType=0, bandwidth=1, initCostMtx=1, reuseCostMtx=0,
                           delStep=2, moveStep=1, diagStep=1, initFirstCol=0, isSubsequence=0)
    dist = compute_distance_for_path(tseries1, tseries2, output[2])

    return dist

# solfege scoring
def convert_onsets_2_syllable_seq(onsets, seq_duration, segment_start, hopsize_t):
    """
    Convert onset sequence 2 syllable sequence
    :param onsets: onset sequence [[onset time, onset label], ...]
    :param seq_duration: sequence length in time
    :param segment_start: sequence segment start time, not include the silence
    :param hopsize_t: hopsize in time
    :return: sequence syllable [sil sil sil .. DO DO DO ... SI SI SI ...]
    """
    onsets_frame = np.round(np.array([sgo[0] for sgo in onsets]) / hopsize_t)

    syllable_seq = [onsets[0][1]]

    current_syllable = onsets[0][1]

    for ii_sample in range(int(round(segment_start / hopsize_t)), int(round(seq_duration / hopsize_t))):

        if ii_sample in onsets_frame:
            idx_onset = np.where(onsets_frame == ii_sample)
            idx_onset = idx_onset[0][0]
            current_syllable = onsets[idx_onset][1]

        syllable_seq.append(current_syllable)

    return syllable_seq


def syllable_seq_score_onset(ground_truth_syllable_seq, student_syllable_seq):
    """
    Calculate the overlapping area as the performance score
    :param ground_truth_syllable_seq: ground truth syllable sequence
    :param student_syllable_seq: student syllable sequence
    :return: syllable performance score
    """
    sample_correct = 0
    for ii in range(len(ground_truth_syllable_seq)):
        if ground_truth_syllable_seq[ii] == student_syllable_seq[ii]:
            sample_correct += 1

    return sample_correct / float(len(ground_truth_syllable_seq))


def convert_kaldi_ctm_2_onset_seq(kaldi_ctm):
    """
    Convert kaldi ctm transcription to onset seq
    :param kaldi_ctm:
    :return:
    """
    onsets = []
    for s in kaldi_ctm:
        onsets.append([s.start, s.label])
    return onsets


def syllable_individual_score_onset(boundaries_ref, boundaries_stu, tol):
    """
    Calculate individual syllable onset score,
    when boundaries_ref and boundaries_stu do not have equal length
    :param boundaries_ref:
    :param boundaries_stu:
    :param tol:
    :return:
    """

    grading_syllable = [0.0] * len(boundaries_ref)
    # rule 1: if student onset is within the teacher's onset window, and if their labels are the same,
    # calculate the score. If their labels are not the same, and their labels are both not sil, score = 0
    for ii in range(len(boundaries_ref)):
        onset_time = boundaries_ref[ii][0]
        onset_label = boundaries_ref[ii][2]
        for jj in range(len(boundaries_stu)):
            if onset_time-tol < boundaries_stu[jj][0] < onset_time+tol:
                if onset_label == boundaries_stu[jj][2] and grading_syllable[ii] == 0.0:
                    grading_syllable[ii] = 1.0 - abs(onset_time - boundaries_stu[jj][0]) / tol
                elif onset_label != boundaries_stu[jj][1] and onset_label != "sil" and boundaries_stu[jj][1] != "sil":
                    grading_syllable[ii] = 0.0
    # rule 2: if there is a student onset is between two reference onsets, and this student onset label is not sil,
    # score = 0
    for ii in range(len(boundaries_ref) - 1):
        onset_time_current = boundaries_ref[ii][0]
        off_time_current = boundaries_ref[ii][1]
        if off_time_current-tol > onset_time_current+tol:
            for jj in range(len(boundaries_stu)):
                if onset_time_current+tol < boundaries_stu[jj][0] < off_time_current-tol and boundaries_stu[jj][2] != "sil":
                    grading_syllable[ii] = 0.0
                    break
    return grading_syllable
    # return [[boundaries_ref[ii][1], grading_syllable[ii]] for ii in range(len(boundaries_ref))]


def syllable_individual_score_onset_equal_length(boundaries_ref, boundaries_stu, tol):
    # We assume the one-by-one correspondence
    # between reference boundaries (boundaries_ref) and student boundaries (boundaries_stu)
    # rule: if student onset is within the teacher's onset window calculate the score.
    assert len(boundaries_ref) == len(boundaries_stu)
    grading_syllable = [0.0] * len(boundaries_stu)
    for ii in range(len(boundaries_ref)):
        onset_time = boundaries_ref[ii][0]
        if onset_time - tol < boundaries_stu[ii][0] < onset_time + tol:
            grading_syllable[ii] = 1.0 - abs(onset_time - boundaries_stu[ii][0]) / tol
    return grading_syllable


def syllable_individual_score_pitch(median_syllable_ref, median_syllable_stu, tol):
    """
    Grade individual syllable pitch
    :param median_syllable_ref: median values of the reference syllable pitch track cents
    :param median_syllable_stu: median values of the student syllable pitch track cents
    :param tol: tolerance
    :return:
    """
    assert len(median_syllable_ref) == len(median_syllable_stu)

    grading_syllable = [0.0] * len(median_syllable_ref)
    for ii in range(len(median_syllable_ref)):
        if np.abs(median_syllable_ref[ii] - median_syllable_stu[ii]) < tol:
            grading_syllable[ii] = 1.0 - abs(median_syllable_ref[ii] - median_syllable_stu[ii])/tol

    return grading_syllable