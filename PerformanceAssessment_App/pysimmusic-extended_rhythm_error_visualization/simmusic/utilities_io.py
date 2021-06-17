import codecs
import copy
import os
import json

import numpy as np
import soundfile as sf
import music21
from simmusic import constants
from simmusic import containers
from simmusic import utilities as utils


def error_opening_file(filename):
    """Prints error opening file

    Parameters
    ----------
    filename : str
        Name of the file
    """
    print("Error opening file: %s" % filename)


def dump_feedback_textfile(rec, feedback, filename):
    """Writes feedback file

    This function dumps feedback object to a text file

    Parameters
    ----------
    rec : :class:containers.Recording
        Recording object
    feedback : list
        Feedback generate from the scoring algorithm
    filename : str
        Output file name

    Returns
    -------
    success : bool
        True or False based on the success of the operation

    """
    # noinspection PyBroadException
    try:
        fid = codecs.open(filename, 'w', encoding='utf-8')

        for feedback_seg in feedback:
            if 'overall' in feedback_seg:
                fid.write("%f\t%f\t%f\n" % (rec.segments[feedback_seg['segment']].start,
                                            rec.segments[feedback_seg['segment']].end,
                                            feedback_seg['overall']))
            notes = rec.segments[feedback_seg['segment']].notes
            for feedback_note in feedback_seg['notes']:
                fid.write("%f\t%f\t%f\n" % (notes[feedback_note['note']].start,
                                            notes[feedback_note['note']].end,
                                            feedback_note['pitch_score']))

        fid.close()
    except:
        print("There was a problem dumping the feedback file. Returning False...")
        return False
    return True


def dump_feedback_json(ref_rec, std_rec, alignment, feedback, filename):
    """Writes feedback file

    This function dumps feedback object to a json file

    Parameters
    ----------
    ref_rec : :class:containers.Recording
        Reference recording object
    std_rec : :class:containers.Recording
        Student's recording object
    alignment : dict
        Alignment between the reference and the student's recording
    feedback : dict
        Feedback from the algorithm
    filename : str
        Output (feedback) file name

    Returns
    -------
    success : bool
        True or False based on the success of the operation

    """
    output = {'overall': {'global': None, 'local': []}, 'pitch': {'global': None, 'local': []}}
    local_feedback = {'start': None, 'end': None, 'start_ref': None, 'end_ref': None, 'score': None, 'comment': None}
    # repacking feedback object
    _feedback = {}
    for elem in feedback:
        if elem['segment'] not in _feedback:
            _feedback[elem['segment']] = {}
        for note_fdbk in elem['notes']:
            if note_fdbk['note'] not in _feedback[elem['segment']]:
                _feedback[elem['segment']][note_fdbk['note']] = note_fdbk['pitch_score']

    # noinspection PyBroadException
    try:
        for align in alignment:
            seg_std = align['segments'][0]
            seg_ref = align['segments'][1]
            for note in align['notes']:
                n_std = std_rec.segments[seg_std].notes[note[0]]
                n_ref = ref_rec.segments[seg_ref].notes[note[1]]

                local_output = copy.deepcopy(local_feedback)
                local_output['start'] = n_std.start
                local_output['end'] = n_std.end
                local_output['start_ref'] = n_ref.start
                local_output['end_ref'] = n_ref.end
                local_output['score'] = _feedback[seg_std][note[0]]
                output['pitch']['local'].append(local_output)
        json.dump(output, codecs.open(filename, 'w', encoding='utf-8'))
    except:
        print("There was a problem dumping the feedback file. Returning False...")
        return False

    return True


def parse_transcription_file(filename):
    """Reads transcription file

    Parameters
    ----------
    filename : str
        Name of the transcription file

    Returns
    -------
    note sequence : list
        List of :class:`containers.Note` objects

    """
    notes = []
    if not os.path.isfile(filename):
        error_opening_file(filename)
        print("Please provide a valid transcription file. Returning None object...")
        return None

    fid = codecs.open(filename, 'r', encoding='utf-8')
    lines = fid.readlines()
    for line in lines:
        sline = line.split('\t')
        notes.append(containers.Note(float(sline[2]), float(sline[0]), float(sline[1]), symbol=sline[3].strip()))
    fid.close()
    return notes


def create_pitch_track_from_notes(rec_obj, hop_size, track_length):
    """
    This function generates a pitch track from the transcription.

    Parameters
    ----------
    rec_obj : :class:`containers.Recording`
        Recording object
    hop_size : float
        Hop size for the pitch track to be generated (in seconds)
    track_length : float
        Length of the pitch track to be generated (in seconds)

    Returns
    -------
    feature : :class:`containers.Feature`
        Feature object loaded with pitch feature

    """
    # TODO: make provision for synth track in the class definition
    n_samples = np.ceil(track_length/hop_size).astype(np.int)
    pitch = constants.PITCH_HZ_ZERO * np.ones(n_samples)
    time = hop_size*np.arange(0, n_samples)

    for seg in rec_obj.segments:
        for note in seg.notes:
            start_ind = np.argmin(np.abs(time - note.start))
            end_ind = np.argmin(np.abs(time - note.end))
            pitch[start_ind:end_ind + 1] = note.pitch
    ftr_obj = containers.Feature(time, pitch)
    return ftr_obj


def create_pitch_track_from_transcription(rec_obj, num_elements, mode='fixedLenNoSilence'):
    """A substitute to utilities_io.create_pitch_track_from_notes that does not
    require a pitch feature in recording object to create the pitch track.
    Pitch track created only using the transcription information

    @Author: Baris Bozkurt

    Parameters
    ----------
    rec_obj : class:`containers.Recording'
        Recording object
    num_elements: int
        Number of points/elements in the pitch track to be synthesized
    mode: str
        Mode of synthesis
            * 'fixedLenNoSilence': Fixed length series where silences are not involved
            * 'constantHop': constant hop size synthesis (not implemented)

    Notes
    -----
        Adds '.start_ind','.end_ind','.dur' in each note and
        'synth_pitch_track_cents' to the Recording object (see notes on top)

    """
    tonic = constants.REF55HZ
    threshold = constants.PITCH_MIN_VOCAL
    invalid_val = constants.INVALID_PITCH_CENTS

    pitch_track = np.zeros(num_elements)
    if mode == 'fixedLenNoSilence':
        # Collecting pitch and duration information in lists
        durs = []
        pitches = []
        for seg in rec_obj.segments:
            for note in seg.notes:
                durs.append(note.end - note.start)
                pitches.append(note.pitch)
        sum_dur = sum(durs)
        # Deciding number of points for each note in the synth pitch series
        hop = sum_dur / num_elements
        num_points = [round(dur / hop) for dur in durs]
        # correcting if rounding ends up in extra or missing points
        while sum(num_points) < num_elements:
            num_points[-1] += 1

        while sum(num_points) > num_elements:
            num_points[-1] -= 1

        # Synthesizing the pitch series and storing info(start, end, duration)
        #   about synth pitch series in the Recording object
        pitch_ind = 0
        for noteInd, notePitch in enumerate(pitches):
            rec_obj.segments[0].notes[noteInd].start_ind = pitch_ind
            note_num_samples = num_points[noteInd]
            pitch_track[pitch_ind:pitch_ind + note_num_samples] = notePitch
            pitch_ind += note_num_samples
            rec_obj.segments[0].notes[noteInd].end_ind = pitch_ind - 1
            rec_obj.segments[0].notes[noteInd].dur = note_num_samples
    # Conversion to cents
    synth_pitch_track_cents = utils.hz2cents_array(pitch_track, tonic, threshold, invalid_val)

    # Shifting pitch track to the first octaves for simplicity of alignment-comparison
    while np.median(synth_pitch_track_cents) > 1200:
        synth_pitch_track_cents -= 1200

    rec_obj.synth_pitch_track_cents = synth_pitch_track_cents
    return True


def create_kaldi_test_data(input_audio_file_path,
                           input_audio_filename,
                           output_data_path,
                           segment_start,
                           text_file=False,
                           text_content=None):
    """
    Create kaldi data files --  wav.scp, segments, utt2spk, spk2utt
    :param input_audio_file_path: input audio path
    :param input_audio_filename: input audio filename
    :param output_data_path: the place to the kaldi files
    :param text_file: whether to write the text file
    :param text_content: the syllable sequence content of the text
    :return: bool
    """
    # exercise_name, start_time, end_time and spk are hardcoded for music-critic demo
    exercise_name = "demo"
    end_time = 100
    spk = "demo_spk"

    if "." in input_audio_filename:
        input_audio_filename = input_audio_filename.split(".")[0]

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)

    # write wav.scp
    with open(os.path.join(output_data_path, 'wav.scp'), "w") as f:
        f.write(exercise_name + '_' + input_audio_filename + ' ' +
                os.path.join(input_audio_file_path, input_audio_filename+".wav"))
        f.write('\n')

    # write segments
    with open(os.path.join(output_data_path, 'segments'), "w") as f:
        x, fs = sf.read(os.path.join(input_audio_file_path, input_audio_filename+".wav"))
        if end_time > len(x) / float(fs) - 0.05:
            end_time = len(x) / float(fs) - 0.05
        f.write(exercise_name + '_' + input_audio_filename + '_000' + ' '
                + exercise_name + '_' + input_audio_filename + ' '
                + str(segment_start) + ' ' + str(end_time))
        f.write('\n')

    # write utt2spk
    with open(os.path.join(output_data_path, 'utt2spk'), "w") as f:
        f.write(exercise_name + '_' + input_audio_filename + '_000' + ' ' + spk)
        f.write('\n')

    # write spk2utt
    with open(os.path.join(output_data_path, 'spk2utt'), "w") as f:
        f.write(spk)
        f.write(' ' + exercise_name + '_' + input_audio_filename + '_000')
        f.write('\n')

    # write text
    if text_file:
        with open(os.path.join(output_data_path, 'text'), "w") as f:
            f.write(exercise_name + '_' + input_audio_filename + '_000' + ' ' + text_content)
            f.write('\n')

    return True


def parse_kaldi_alignment_transcription(transcription_file,
                                        segment_start):
    """
    Read kaldi alignment ctm file
    :param transcription_file: transcription filename
    :param segment_start: segment file start time
    :return:
    """
    if not os.path.isfile(transcription_file):
        error_opening_file(transcription_file)
        print("Please provide a valid transcription file. Returning None object...")
        return None

    syllables = []
    with open(transcription_file, "r") as f:
        content = f.readlines()
        for line in content:
            line_elements = line.split(" ")
            if line_elements[4] == "<eps>":
                label = "sil"
            else:
                label = line_elements[4]
            syllables.append(containers.Syllable(start=float(line_elements[2])+segment_start,
                                                 duration=float(line_elements[3]),
                                                 label=label))
            print(float(line_elements[2])+segment_start, float(line_elements[3]), label)
    return syllables

def create_kaldi_test_data(input_audio_file_path,
                           input_audio_filename,
                           output_data_path,
                           segment_start,
                           text_file=False,
                           text_content=None):
    """
    Create kaldi data files --  wav.scp, segments, utt2spk, spk2utt
    :param input_audio_file_path: input audio path
    :param input_audio_filename: input audio filename
    :param output_data_path: the place to the kaldi files
    :param text_file: whether to write the text file
    :param text_content: the syllable sequence content of the text
    :return: bool
    """
    # exercise_name, start_time, end_time and spk are hardcoded for music-critic demo
    exercise_name = "demo"
    end_time = 100
    spk = "demo_spk"

    if "." in input_audio_filename:
        input_audio_filename = input_audio_filename.split(".")[0]

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)

    # write wav.scp
    with open(os.path.join(output_data_path, 'wav.scp'), "w") as f:
        f.write(exercise_name + '_' + input_audio_filename + ' ' +
                os.path.join(input_audio_file_path, input_audio_filename+".wav"))
        f.write('\n')

    # write segments
    with open(os.path.join(output_data_path, 'segments'), "w") as f:
        x, fs = sf.read(os.path.join(input_audio_file_path, input_audio_filename+".wav"))
        if end_time > len(x) / float(fs) - 0.05:
            end_time = len(x) / float(fs) - 0.05
        f.write(exercise_name + '_' + input_audio_filename + '_000' + ' '
                + exercise_name + '_' + input_audio_filename + ' '
                + str(segment_start) + ' ' + str(end_time))
        f.write('\n')

    # write utt2spk
    with open(os.path.join(output_data_path, 'utt2spk'), "w") as f:
        f.write(exercise_name + '_' + input_audio_filename + '_000' + ' ' + spk)
        f.write('\n')

    # write spk2utt
    with open(os.path.join(output_data_path, 'spk2utt'), "w") as f:
        f.write(spk)
        f.write(' ' + exercise_name + '_' + input_audio_filename + '_000')
        f.write('\n')

    # write text
    if text_file:
        with open(os.path.join(output_data_path, 'text'), "w") as f:
            f.write(exercise_name + '_' + input_audio_filename + '_000' + ' ' + text_content)
            f.write('\n')

    return True


def parse_kaldi_alignment_transcription(transcription_file,
                                        segment_start):
    """
    Read kaldi alignment ctm file
    :param transcription_file: transcription filename
    :param segment_start: segment file start time
    :return:
    """
    if not os.path.isfile(transcription_file):
        error_opening_file(transcription_file)
        print("Please provide a valid transcription file. Returning None object...")
        return None

    syllables = []
    with open(transcription_file, "r") as f:
        content = f.readlines()
        for line in content:
            line_elements = line.split(" ")
            if line_elements[4] == "<eps>":
                label = "sil"
            else:
                label = line_elements[4]
            syllables.append(containers.Syllable(start=float(line_elements[2])+segment_start,
                                                 duration=float(line_elements[3]),
                                                 label=label))
            print(float(line_elements[2])+segment_start, float(line_elements[3]), label)
    return syllables


def parse_music_xml(music_xml_file, quarter_time, segment_start):
    """
    Parse the music xml file
    :param music_xml_file: music xml filename
    :param quarter_time: quarter note length in second
    :param segment_start: the recognition start time
    :return: list of Syllable objects
    """
    # parse music XML to score
    s = music21.converter.parse(music_xml_file)
    syllables = []  # store Syllable object
    median_pitch = []  # score the pitch frequency in cents
    syllable_onset_time = 0.0
    for p in s.parts:
        for ii, n in enumerate(p.flat.notesAndRests):
            # the first syllable start time should be adjusted according to the segment start
            if ii == 0:
                start_time = segment_start
                dur = n.duration.quarterLength * quarter_time - start_time
            else:
                start_time = syllable_onset_time
                dur = n.duration.quarterLength * quarter_time

            if n.name == "rest":
                label = "sil"
            else:
                label = constants.dict_solfege_name[n.name]
                median_pitch.append(n.pitch.frequency)

            syllables.append(containers.Syllable(start=start_time,
                                                 duration=dur,
                                                 label=label))

            syllable_onset_time += n.duration.quarterLength * quarter_time
        break
    return syllables, utils.hz2pitch(np.array(median_pitch))
