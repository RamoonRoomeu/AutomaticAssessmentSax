"""
This file contains all the wrapper codes for performing evaluation in different contexts.
"""
from simmusic import segmentation
from simmusic import scoring
from simmusic.containers import Recording
from simmusic import utilities_io
from simmusic import constants


def evaluate_flatnote_playalong_rule_based(pitch_file_std, trans_file, feedback_filename,
                                           scoring_logic=1, output_format='json'):
    """Evaluate flatnote lesson sung in play along mode using rule based method.

    This function uses a rule based approach to perform evaluation of a performance of a flat note (slow/medium paced)
    type an exercise given the recording is done in a sing along mode (i.e. performance completely in sync with
    the reference in terms of the timings).

    Parameters
    ----------
    pitch_file_std : str
        Name of the pitch file corresponding to a student's recording.
    trans_file : str
        Name of the transcription file
    feedback_filename : str
        Name of the feedback file to be generated as the output
    scoring_logic : int
        Either 0 or 1 for selecting two types of rule based evaluation logic
    output_format : str
        Either 'json' or 'text' for specifying the format of the feedback file (output of the method)

    Returns
    -------
    success : bool
        True or False depending on the success of the operation

    """

    # Some hard coded params for this setting of evaluation
    mode = 'play-along'
    ref_freq = constants.REF55HZ

    # Creating recording objects for both reference and a student's imitation
    ref_rec = Recording('reference', mode)
    std_rec = Recording('imitation', mode)

    # loading pitch data
    # ref_rec.load_pitch_data(pitch_file_ref)
    std_rec.load_pitch_data(pitch_file_std)

    # Converting pitch from Hz to Cents (scale)
    success = std_rec.convert_pitch_hz2cents(tonic=ref_freq, threshold=constants.PITCH_MIN_VOCAL,
                                             invalid_val=constants.INVALID_PITCH_CENTS)
    if not success:
        print("Problems converting pitch from Hz to cents...")
        return False

    # performing segmentation and alignment
    alignment = segmentation.segment_notes_playalong(ref_rec, std_rec, trans_file)

    # Now that we have all the information about the segments in student's recording and its alignment with refernce,
    # lets perform scoring of the student's recording
    if scoring_logic == 0:
        feedback = scoring.score_flat_note_rule_based_1(ref_rec, std_rec, alignment, ref_freq)
    elif scoring_logic == 1:
        feedback = scoring.score_flat_note_rule_based_2(ref_rec, std_rec, alignment, ref_freq)
    else:
        print("Please provide a valid scoring logic (Currently supported either 0 or 1)")
        return False

    # We have all the scores, lets dump it in a desired format
    if output_format == 'text':
        success = utilities_io.dump_feedback_textfile(std_rec, feedback, feedback_filename)
    elif output_format == 'json':
        success = utilities_io.dump_feedback_json(ref_rec, std_rec, alignment, feedback, feedback_filename)
    else:
        print('Please specify a valid format of the feedback file')
        success = False
    return success


def evaluate_flatnote_dialogue_rule_based(pitch_file_std, trans_file, feedback_filename,
                                          scoring_logic=1, output_format='json'):
    """Evaluate flatnote lesson sung in dialogue mode using rule based method.

    This function uses a rule based approach to perform evaluation of performance of a flat note (slow/medium paced)
    type an exercise given the performance is recorded in a dialogue mode (i.e. first a teacher and then student).

    Parameters
    ----------
    pitch_file_std : str
        Name of the pitch file corresponding to a student's recording.
    trans_file : str
        Name of the transcription file
    feedback_filename : str
        Name of the feedback file to be generated as the output
    scoring_logic : int
        Either 0 or 1 for selecting two types of rule based evaluation logic
    output_format : str
        Either 'json' or 'text' for specifying the format of the feedback file (output of the method)

    Returns
    -------
    success : bool
        True or False depending on the success of the operation

    """
    # Some hard coded params for this setting of evaluation
    mode = 'dialogue'
    ref_freq = constants.REF55HZ

    # Creating recording objects for both reference and a student's imitation
    ref_rec = Recording('reference', mode)
    std_rec = Recording('imitation', mode)

    # loading pitch data
    std_rec.load_pitch_data(pitch_file_std)

    # Converting pitch from Hz to Cents (scale)
    success = std_rec.convert_pitch_hz2cents(tonic=ref_freq, threshold=constants.PITCH_MIN_VOCAL,
                                             invalid_val=constants.INVALID_PITCH_CENTS)
    if not success:
        print("Problems converting pitch from Hz to cents...")
        return False

    # performing segmentation and alignment
    try:
        alignment = segmentation.segment_notes_dialogue(ref_rec, std_rec, trans_file, tonic=ref_freq,
                                                        threshold=constants.PITCH_MIN_VOCAL,
                                                        invalid_val=constants.INVALID_PITCH_CENTS)
    except Exception as e:
        raise e
    # Now that we have all the information about the segments in student's recording and its alignment with refernce,
    # lets perform scoring of the student's recording
    if scoring_logic == 0:
        feedback = scoring.score_flat_note_rule_based_1(ref_rec, std_rec, alignment, ref_freq)
    elif scoring_logic == 1:
        feedback = scoring.score_flat_note_rule_based_2(ref_rec, std_rec, alignment, ref_freq)
    else:
        print("Please provide a valid scoring logic (Currently supported either 0 or 1)")
        return False

    # We have all the scores, lets dump it in a desired format
    if output_format == 'text':
        success = utilities_io.dump_feedback_textfile(std_rec, feedback, feedback_filename)
    elif output_format == 'json':
        success = utilities_io.dump_feedback_json(ref_rec, std_rec, alignment, feedback, feedback_filename)
    else:
        print('Please specify a valid format of the feedback file')
        success = False
    return success

    return success


def evaluate_phrase_dialogue_rule_based(pitch_file_ref, pitch_file_std, feedback_filename):
    """Evaluate phrase type lessons sung in dialogue mode using rule based method.

    This function uses a rule based approach to perform evaluation of performance of phrase(song segments) type
     exercise given the performanance is recorded in a dialogue (async) mode.

    Parameters
    ----------
    pitch_file_ref : str
        Name of the pitch file corresponding to the reference recording.
    pitch_file_std : str
        Name of the pitch file corresponding to a student's recording.
    feedback_filename : str
        Name of the feedback file to be generated as the output

    Returns
    -------
    success : bool
        True or False depending on the success of the operation

    """
    # Some hard coded params for this setting of evaluation
    mode = 'dialogue'
    ref_freq = constants.REF55HZ

    # Creating recording objects for both reference and a student's imitation
    ref_rec = Recording('reference', mode)
    std_rec = Recording('imitation', mode)

    # loading pitch data
    ref_rec.load_pitch_data(pitch_file_ref)
    std_rec.load_pitch_data(pitch_file_std)

    # TODO: check if both the pitch tracks have same sampling rate! its crucial we will compare the samples!

    # Converting pitch from Hz to Cents (scale)
    success = ref_rec.convert_pitch_hz2cents(tonic=ref_freq, threshold=constants.PITCH_MIN_VOCAL,
                                             invalid_val=constants.INVALID_PITCH_CENTS)
    if not success:
        print("Problems converting pitch from Hz to cents...")
        return False
    success = std_rec.convert_pitch_hz2cents(tonic=ref_freq, threshold=constants.PITCH_MIN_VOCAL,
                                             invalid_val=constants.INVALID_PITCH_CENTS)
    if not success:
        print("Problems converting pitch from Hz to cents...")
        return False

    # performing segmentation
    try:
        alignment = segmentation.segment_phrase_dialogue(ref_rec, std_rec)
    except Exception as e:
        raise e
    # Now that we have all the information about the segments in student's recording and its alignment with refernce,
    # lets perform scoring of the student's recording
    feedback = scoring.score_phrases_rule_based(ref_rec, std_rec, alignment)

    # We have all the scores, lets dump it in a desired format
    success = utilities_io.dump_feedback_textfile(std_rec, feedback, feedback_filename)

    return success


# This part of the code basically runs all the evaluation functions, NOte that these are not the unittests!
# Baris wanted a place where all these functions can be run, without getting into the complexities of unit tests.
if __name__ == '__main__':
    import os

    exts = {'ref_pitch': '.pitch', 'user_pitch': '.pitch_user', 'transcription': '.trans', 'feedback': '.fdbk',
            'log': '.log', 'params': '.params'}

    scratch_dir = os.path.join(os.path.dirname(__file__), 'scratch_dir')
    if not os.path.isdir(scratch_dir):
        os.makedirs(scratch_dir)
    ################################################################################################
    # Testing the code for scoring flat notes played as play-along mode (first rule based approach)
    ################################################################################################
    filename = os.path.join(os.path.dirname(__file__), '../tests/data/riyaz_psds/0AL0nEmjVR/0AL0nEmjVR')
    output_file = os.path.join(scratch_dir, '0AL0nEmjVR' + exts['feedback'])
    scoring_type = 0
    if os.path.isfile(output_file):
        os.remove(output_file)
    print("----1----")
    print("Running test for -evaluate_flatnote_playalong_rule_based()")
    print("Input file: %s" % filename)
    print("Output file: %s" % output_file)
    print("--------")
    evaluate_flatnote_playalong_rule_based(filename + exts['user_pitch'],
                                           filename + exts['transcription'],
                                           output_file,
                                           scoring_logic=scoring_type)
    ################################################################################################
    # Testing the code for scoring flat notes played as play-along mode (second rule based approach)
    ################################################################################################
    filename = os.path.join(os.path.dirname(__file__), '../tests/data/riyaz_psds/zzzx4FqSC5/zzzx4FqSC5')
    output_file = os.path.join(scratch_dir, 'zzzx4FqSC5' + exts['feedback'])
    scoring_type = 1
    if os.path.isfile(output_file):
        os.remove(output_file)
    print("----2----")
    print("Running test for -evaluate_flatnote_playalong_rule_based()")
    print("Input file: %s" % filename)
    print("Output file: %s" % output_file)
    print("--------")
    evaluate_flatnote_playalong_rule_based(filename + exts['user_pitch'],
                                           filename + exts['transcription'],
                                           output_file,
                                           scoring_logic=scoring_type)

    ################################################################################################
    # Testing the code for scoring flat notes played as play-along mode (second rule based approac
    ################################################################################################
    filename = os.path.join(os.path.dirname(__file__), '../tests/data/other/0AL0nEmjVR/0AL0nEmjVR')
    output_file = os.path.join(scratch_dir, '0AL0nEmjVR_dialogue' + exts['feedback'])
    scoring_type = 1
    if os.path.isfile(output_file):
        os.remove(output_file)
    print("----3----")
    print("Running test for -evaluate_flatnote_dialogue_rule_based()")
    print("Input file: %s" % filename)
    print("Output file: %s" % output_file)
    print("--------")
    evaluate_flatnote_dialogue_rule_based(filename + exts['user_pitch'],
                                          filename + exts['transcription'],
                                          output_file,
                                          scoring_logic=scoring_type)
    ################################################################################################
    # Testing the code for scoring phrases played as dialogue mode (rule based approach)
    ################################################################################################
    filename = os.path.join(os.path.dirname(__file__), '../tests/data/other/yaman_MDN_NRS/yaman_MDN_NRS')
    output_file = os.path.join(scratch_dir, 'yaman_MDN_NRS' + exts['feedback'])
    if os.path.isfile(output_file):
        os.remove(output_file)
    print("----4----")
    print("Running test for -evaluate_phrase_dialogue_rule_based()")
    print("Input file: %s" % filename)
    print("Output file: %s" % output_file)
    print("--------")
    evaluate_phrase_dialogue_rule_based(filename + exts['ref_pitch'],
                                        filename + exts['user_pitch'],
                                        output_file)
