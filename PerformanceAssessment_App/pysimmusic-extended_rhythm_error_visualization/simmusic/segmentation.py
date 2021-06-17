"""
This file contains functions that perform segmentation and alignment of the melodic elements (in the audio recordings)
"""
import copy
from simmusic import utilities as utils
from simmusic import utilities_io
from simmusic import constants
import numpy as np
from simmusic.containers import Note, Segment
from simmusic.dtw import dtw
from simmusic.exceptions import SegmentNotFound


# TODO: improve import statements, take care of he relative imports, import from the installed copy, or from a package

def segment_notes_playalong(ref_rec, std_rec, trans_file):
    """Segment notes for play-along mode of recording

    This function segment notes in a recording given that the student recording is done in a play-along mode (complete
    time sync).

    Parameters
    ----------
    ref_rec : :class:`container.Recording`
        Reference recording object
    std_rec : :class:`container.Recording`
        Student's recording object
    trans_file : str
        Name of the transcription file to be used in the segmentation process.

    Returns
    -------
    alignment : list
        Alignment between the two recorings. List of dict

    """
    # ASSUMPTION: Playalong assumes complete sync in time between what a student is playing and what he is listening to.
    # It implies that we can basically segment recordings based on time stamps of theoretical transcription

    # loading transcription data
    ref_rec.load_trans_data(trans_file)

    # Since its a sing along copying the segment and note file from reference to the student's recording
    # This is basically saying segmentation of student's file is same as that of the reference
    std_rec.segments = copy.deepcopy(ref_rec.segments)

    # Computing indexes from time values of segments and notes
    std_rec.compute_note_indexes()

    # Segmentation is done, now populating alignment data, for this mode of recording its just one to one mapping.
    alignment = []
    for ii, segment in enumerate(ref_rec.segments):
        note_alignments = [(jj, jj) for jj in range(len(segment.notes))]
        alignment.append({'segments': (ii, ii), 'notes': note_alignments})
    return alignment


def segment_notes_dialogue(ref_rec, std_rec, trans_file, tonic, threshold, invalid_val, filter_corner_segments=True,
                           filtering_threshold=0.1, valid_note_duration=0.1):
    """Segment notes for Dialogue mode of recording

    This function segment notes in a recording given that the student recording is done in a dialogue mode (i.e. a
    student repeated after the reference recording). This function performs dtw based global alignment to segment the
    notes

    Parameters
    ----------
    ref_rec : :class:`container.Recording`
        Reference recording object
    std_rec : :class:`container.Recording`
        Student's recording object
    trans_file : str
        Name of the transcription file to be used in the segmentation process.
    tonic : double
        Reference frequency for Hz to Cents conversion
    threshold : float
        Minimum value of a pitch sample to be considered as a voiced region
    invalid_val : float
        Value of the pitch in cents to be assigned for the unvoiced regions
    filter_corner_segments : bool
        Whether or not to filter spurious voiced segments at the edges
    filtering_threshold : double
        Threshold to be used for filtering spurious vocal segments at the edges
    valid_note_duration : float
        Valid note duration, below which we wil discard the notes.

    Returns
    -------
    alignment: list
        Alignment between the two recorings. List of dict

    """
    # loading transcription data
    ref_rec.load_trans_data(trans_file)

    # computing ideal (step kind) pitch from the transcription (notes)
    # We assume that we generate pitch track with the same hop as it is in student's pitch (it has several adv.)
    hop_size = utils.compute_hopsize(std_rec.feature.time)
    track_time = ref_rec.segments[-1].notes[-1].end
    pitch_ideal = utilities_io.create_pitch_track_from_notes(ref_rec, hop_size, track_time)
    pitch_ideal.convert_pitch_hz2cents(tonic=tonic, threshold=threshold, invalid_val=invalid_val)

    # NOTE: we start considering (ideal) pitch track generated from the transcription as the reference pitch track.
    # This is very generic because when we want to do the same operation with a real reference pitch track, none of the
    # following code will change
    ref_rec.feature = copy.deepcopy(pitch_ideal)

    # Computing segment and note indexes
    ref_rec.compute_segment_indexes()
    ref_rec.compute_note_indexes()

    # Computing the active pitch region in the student's pitch
    try:
        seg_obj = segment_active_pitch_region(std_rec.feature.pitch, std_rec.feature.time, filter_corner_segments,
                                              filtering_threshold)
    except Exception as e:
        raise e
    std_rec.segments.append(seg_obj)
    std_rec.compute_segment_indexes()

    dtw_output = dtw.dtw1d_GLS(pitch_ideal.cents[ref_rec.segments[0].start_ind:ref_rec.segments[0].end_ind],
                               std_rec.feature.cents[std_rec.segments[0].start_ind:std_rec.segments[0].end_ind],
                               distType=0, hasGlobalConst=0, globalType=0, bandwidth=1, initCostMtx=1, reuseCostMtx=0,
                               delStep=0, moveStep=1, diagStep=1, initFirstCol=0, isSubsequence=0)

    notes_std = []
    for note in ref_rec.segments[0].notes:
        # Note that we iterate on reference notes, hence we see if they are present on not. Additional notes from
        # student might not be considered here.
        ind_start_ref = np.argmin(np.abs(dtw_output[2][0] + ref_rec.segments[0].start_ind - note.start_ind))
        ind_end_ref = np.argmin(np.abs(dtw_output[2][0] + ref_rec.segments[0].start_ind - note.end_ind))
        # TODO: add a check that point of minimum value should not be farther than 2 3 samples! (basically the jump
        # in dtw)
        ind_start_std = dtw_output[2][1][ind_start_ref] + std_rec.segments[0].start_ind
        ind_end_std = dtw_output[2][1][ind_end_ref] + std_rec.segments[0].start_ind
        notes_std.append(Note(pitch=note.pitch,
                              start=std_rec.feature.time[ind_start_std],
                              end=std_rec.feature.time[ind_end_std],
                              start_ind=ind_start_std,
                              end_ind=ind_end_std))
    std_rec.segments[0].notes = notes_std

    # Segmentation is done, now populating alignment data
    # Note that we create a note entry for even a invalid (extremely short) note in student's recording, its just that
    # after filtering it we do not consider it in the evaluation
    alignment = []
    for ii, segment in enumerate(std_rec.segments):
        note_alignments = []
        for jj, note in enumerate(segment.notes):
            if (note.end - note.start) > valid_note_duration:
                note_alignments.append((jj, jj))
        alignment.append({'segments': (ii, ii), 'notes': note_alignments})
    return alignment


def segment_phrase_dialogue(ref_rec, std_rec, filter_corner_segments=True, filtering_threshold=0.1):
    """Segment Phrase-based exercises for Dialogue mode of recording

    This function performs segmentation of voiced regions (phrases) based on pitch (pitch estimation algorithms also
    perform Voice Activity Detection)

    Parameters
    ----------
    ref_rec : :class:`container.Recording`
        Reference recording object
    std_rec : :class:`container.Recording`
        Student's recording object
    filter_corner_segments: bool
        Whether or not to filter spurious voiced segments at the edges
    filtering_threshold : double
        Threshold to be used for filtering spurious vocal segments at the edges

    Returns
    -------
    alignment : list
        Alignment between the two recorings. List of dict

    """
    # ASSUMPTION: both reference and student's recordings contain same complete lesson sung only once!! NOTE: if this
    #  assumption is not valid, we would need some more info (mapping timestamps etc) for considering multiple
    # iterations

    # picking active pitch region in the reference recording
    try:
        seg_obj = segment_active_pitch_region(ref_rec.feature.pitch, ref_rec.feature.time, filter_corner_segments,
                                              filtering_threshold)
    except Exception as e:
        raise e
    ref_rec.segments.append(seg_obj)

    # picking active pitch region in the student's recording
    try:
        seg_obj = segment_active_pitch_region(std_rec.feature.pitch, std_rec.feature.time, filter_corner_segments,
                                              filtering_threshold)
    except Exception as e:
        raise e
    std_rec.segments.append(seg_obj)

    alignment = [{'segments': (0, 0), 'notes': []}]
    return alignment


def segment_active_pitch_region(pitch, time, filter_corner_segments=True, filtering_threshold=0.1):
    """Segments active voice region in a pitch sequence

    Parameters
    ----------
    pitch : np.ndarray
        Pitch samples array corresponding to the audio recording
    time : np.ndarray
        Time stamps of the extracted acoustic feature(s)
    filter_corner_segments : bool
        Whether or not to filter spurious voiced segments at the edges
    filtering_threshold : double
        Threshold to be used for filtering spurious vocal segments at the edges

    Returns
    -------
    segment : :class:`container.Segment`
        Object of :class:`container.Segment` class

    """
    # The segmentation here is just picking out voiced regions
    ind_active = np.where(pitch > constants.PITCH_MIN_VOCAL)[0]
    segments = utils.group_indices(ind_active)
    segments_selected = []
    if filter_corner_segments:
        for seg in segments:
            if (time[seg[1]] - time[seg[0]]) > filtering_threshold:
                segments_selected.append(seg)
    segments_selected = np.array(segments_selected)
    if segments_selected.size > 0:
        seg_obj = Segment(start=time[segments_selected[0, 0]],
                          end=time[segments_selected[-1, 1]],
                          start_ind=segments_selected[0, 0],
                          end_ind=segments_selected[-1, 1],
                          seg_type='phrase')
    else:
        raise SegmentNotFound
    return seg_obj


def note_segmentation_pitch_quantization1(pitch, time):
    """Written by Oriol (minor variable name changes by sankalp), see if its relevant in any use case or else remove it
    Segment a sequence of pitch values into notes. Return end and start times of each note in the sequence.

    Parameters
    ----------
    pitch : np.ndarray
        Pitch samples array corresponding to the audio recording
    time : np.ndarray
        Time stamps of the extracted acoustic feature(s)

    Returns
    -------
    note sequence : list
        List of tuple with (note midi number, start frame number, end frame number)

    """
    # TODO: I see many shortcomings in this code 1) Else to reset value of p 2) It should not be absolute 4,
    # but time threshold! Its a very simplistic snapping based eval, see if this is of any value!

    p = 0  # counter
    notes = []

    for ii, pitch_value in enumerate(pitch[1:]):
        semitone = utils.hz2pitch(pitch_value)
        # compare to previous pitch
        if round(semitone) == round(utils.hz2pitch(pitch[ii - 1])) and 20 < pitch_value < 5000:
            if p == 0:
                start_ind = ii - 1
            p += 1
        if p >= 4:
            if not round(semitone) == round(utils.hz2pitch(pitch[ii - 1])) or ii == len(pitch) - 1:
                note_name = int(round(utils.hz2pitch(pitch[ii - 1])))
                notes.append(Note(pitch=utils.pitch2hz(note_name), start=time[start_ind], end=time[ii],
                                  start_ind=start_ind, end_ind=ii, midi_number=note_name))
                p = 0
    return notes


def note_segmentation_pitch_quantization2(pitch, time, vicinity_thsld=50, note_grid=None, duration_threshold=0.2):
    """ Performs note segmentation using pitch quantization

    This function performs note segmentation given a sequence of pitch using simple pitch quantization method. Pitch
    samples that happen to continuously fall in the vicity of a note are segmented and labelled as that note.

    Parameters
    ----------
    pitch : np.ndarray
        Pitch samples array corresponding to the audio recording
    time : np.ndarray
        Time stamps of the extracted acoustic feature(s)
    vicinity_thsld : double
        Vicinity threshold for quantization (in Cents)
    note_grid : array_like, np.ndarray
        Array of pitch (cents) values to which pitch samples are quantized
    duration_threshold : double
        Duration below which the notes will be filtered (in seconds)

    Returns
    -------
    note sequence : list
        List of :class:`containers.Note` objects

    """
    cents = constants.CENTS_PER_OCTAVE * np.log2(pitch / constants.REF55HZ)
    notes = []
    start_times = []
    if note_grid is None:
        note_grid = 100 * np.arange(48)
        note_grid_hz = constants.REF55HZ * np.power(2, note_grid / constants.CENTS_PER_OCTAVE)
    else:
        note_grid_hz = constants.REF55HZ * np.power(2, note_grid / constants.CENTS_PER_OCTAVE)
    for ii, grid_val in enumerate(note_grid):
        # just to make process fast lets find in one shot all the points which are atleast closer than narrow threshold
        ind_narrow = np.where(np.abs(cents - grid_val) < vicinity_thsld)[0]
        if len(ind_narrow) > 0:
            segments = utils.group_indices(ind_narrow)
            for segment in segments:
                if (time[segment[1]] - time[segment[0]]) > duration_threshold:
                    notes.append(Note(pitch=note_grid_hz[ii],
                                      start=time[segment[0]],
                                      end=time[segment[1]],
                                      start_ind=segment[0],
                                      end_ind=segment[1],
                                      cent=grid_val))
                    start_times.append(time[segment[0]])

    # sorting the notes according to the onset times
    arg_sort = np.array(np.argsort(start_times))
    notes_sorted = []
    for ii in arg_sort:
        notes_sorted.append(notes[ii])
    return notes_sorted
