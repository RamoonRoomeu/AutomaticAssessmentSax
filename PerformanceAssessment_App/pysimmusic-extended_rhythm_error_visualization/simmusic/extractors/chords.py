import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import io

from pychord_tools.third_party import NNLSChromaEstimator
from pychord_tools.low_level_features import SmoothedStartingBeatChromaEstimator, BeatChromaEstimator
from  pychord_tools.models import load_model
import simmusic

def transcribe_chords(beats_array, sound_filename, chroma_pattern_model):
    """

    Transcribes chords for given audio file

    Parameters
    ----------
    beats_array : np.array
        floats array of times of the beats on which chords are played
    sound_filename : str
        Name of the sound audio file
    chroma_pattern_model : ChromaPatternModel
        model for estimating chords quality

    Returns
    -------
    chords : np.array
        numpy array of PitchedPattern len(chords) == len(beats_array)
    """
    chromaEstimator = NNLSChromaEstimator()
    segmentChromaEstimator = SmoothedStartingBeatChromaEstimator(smoothing_time=0.6)

    beatsChromas = segmentChromaEstimator.get_chroma_by_beats(
        beats_array, chromaEstimator.estimate_chroma(sound_filename))
    chords, strengths = chroma_pattern_model.predict(beatsChromas)
    return chords


def briefLabels(labels):
    return [s.replace(':min', 'm') for s in labels]

def exercisePNG(realSegments, predictedPatterns, chroma_pattern_model):
    truePatterns = realSegments.pitched_patterns()
    bottomLabels = np.empty(len(realSegments.labels), dtype='object')
    for i in range(len(realSegments.labels)):
        # print(truePatterns[i], type(truePatterns[i]), predictedPatterns[i], type(predictedPatterns[i]), truePatterns[i] == predictedPatterns[i])
        if (truePatterns[i] == predictedPatterns[i]):
            bottomLabels[i] = ''
        else:
            bottomLabels[i] = str(predictedPatterns[i])

    topLabels = np.empty(len(realSegments.labels), dtype='object')
    for i in range(len(realSegments.labels)):
        c = realSegments.labels[i]
        topLabels[i] = c + '\n' + "%.2f" % realSegments.start_times[i]

    normCorrectness = chroma_pattern_model.correctness_given_sequence(realSegments.chromas, truePatterns,
                                                                  normalize=True)
    normLu = chroma_pattern_model.log_utilities_given_sequence(realSegments.chromas, truePatterns,
                                                          normalize=True)

    rawBalance = chroma_pattern_model.balance_given_sequence(realSegments.chromas, truePatterns,
                                                         normalize=False)
    m = np.array([np.ones(rawBalance.shape) + rawBalance / 30.0,
                  np.exp(normCorrectness),
                  np.exp(normLu)])
    fig, ax = plt.subplots(figsize=(15, 5), dpi=80)
    ax.pcolor(m, cmap='RdYlGn', norm=plt.Normalize(0, 1, clip=True))
    ax.set_yticks(np.arange(3) + 0.5, minor=False)
    ax.set_yticklabels(['balance', 'outside', 'overall'])
    ax.xaxis.set_ticks_position('none')
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=True, top=False, labeltop=False, labelsize=6,
                   colors='red')
    ax.set_xticks(0.5 + np.arange(0, len(bottomLabels)))
    ax.set_xticklabels(briefLabels(bottomLabels))
    ax.xaxis.set_ticks_position('none')

    ax2 = ax.twiny()
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labeltop=True, labelbottom=False, labelsize=8)
    ax2.set_xticks(0.5 + np.arange(0, len(topLabels), 2))
    ax2.set_xticklabels(briefLabels(topLabels[0::2]))
    ax2.set_xlim(ax.get_xlim())
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    res = buf.getvalue()
    buf.close()
    return res


def estimate_grade_bpm(bpm, count_down_beats, number_of_beats_to_estimate, ref_filename, student_filename, chroma_pattern_model = None):
    """

    Estimates grade for chord exercise

    Parameters
    ----------
    bpm : float
        beats per minute rate
    count_down_beats : int
        number of countdown beats ("silent" beats in the beginning of the exercise).
    number_of_beats_to_estimate : int
        number of beats in evaluated segment (after countdown)
    ref_filename : str
        Name of the reference performance audio file
    student_filename : str
        Name of the performance audio file of a student
    chroma_pattern_model : ChromaPatternModel
        model for estimating chords quality

    Returns
    -------
    grade, chords_scores, png_bytes : int, np.array, bytes
        grade obtained, scores for individual chords len(chords_scores) == len(beats_array), image for visualization.
        grade varies from 1 (lowest) to 4 (best).
        chord_scores vary from 0 (lowset) to 1 (best).
        png_bytes: image in PNG format:
                   Upper X-labels: reference chords and time stamps.
                   Top color bar: overall grade.
                   Middle color bar: intensity of tones which are "outside" of chord.
                   Low color bar: measure of tones balance.
                   Bottob x-labels: if there are supposed student's mistakes:
                                    chords, which are actually played instead of the reference chords.

    """
    beats_array = 60.0 / bpm * (count_down_beats + np.arange(
        0, float(number_of_beats_to_estimate), dtype='float'))
    duration = 60.0 / bpm * (count_down_beats + number_of_beats_to_estimate)
    return estimate_grade(beats_array, ref_filename, student_filename, duration, chroma_pattern_model)

def estimate_grade(ref_beats_array, ref_filename, student_filename, student_duration, chroma_pattern_model = None, student_beats_array = None):
    """

    Estimates grade for chord exercise

    Parameters
    ----------
    beats_array : np.array
        floats array of times of the beats on which chords are played
        in reference sound
    ref_filename : str
        Name of the reference performance audio file
    student_filename : str
        Name of the performance audio file of a student
    student_duration : float
        student_duration of the evaluated segment (must be greater than last beat value)
    chroma_pattern_model : ChromaPatternModel
        model for estimating chords quality
    student_beats_array : np.array
        floats array of times of the beats on which chords are played
        in student sound (if deifferent from "reference")

    Returns
    -------
    grade, chords_scores, png_bytes : int, np.array, bytes
        grade obtained, scores for individual chords len(chords_scores) == len(beats_array), image for visualization.
        grade varies from 1 (lowest) to 4 (best).
        chord_scores vary from 0 (lowset) to 1 (best).
        png_bytes: image in PNG format:
                   Upper X-labels: reference chords and time stamps.
                   Top color bar: overall grade.
                   Middle color bar: intensity of tones which are "outside" of chord.
                   Low color bar: measure of tones balance.
                   Bottob x-labels: if there are supposed student's mistakes:
                                    chords, which are actually played instead of the reference chords.

    """

    if (chroma_pattern_model == None):
        chroma_pattern_model = load_model(os.path.join(simmusic.extractors.__path__[0], 'chord_models/ind_pdf.pkl'))

    expectedChords = transcribe_chords(ref_beats_array, ref_filename, chroma_pattern_model)

    if student_beats_array == None:
        student_beats_array = ref_beats_array

    chromaEstimator = BeatChromaEstimator(
        student_beats_array,
        expectedChords,
        student_duration,
        chroma_estimator=NNLSChromaEstimator(),
        beat_chroma_estimator=SmoothedStartingBeatChromaEstimator(smoothing_time=0.6),
        uid = "")
    realSegments = chromaEstimator.load_chromas(student_filename)

    predicted, plu = chroma_pattern_model.predict(realSegments.chromas)
    nlu = chroma_pattern_model.log_utilities_given_sequence(
        chromas=realSegments.chromas, pitched_patterns=realSegments.pitched_patterns(), normalize=True)
    lu = chroma_pattern_model.log_utilities_given_sequence(
        chromas=realSegments.chromas, pitched_patterns=realSegments.pitched_patterns(), normalize=False)
    print('lu', lu)
    grade = lu.sum() / len(lu)
    if grade > -4:
        grade14 = 4
    elif grade > -8:
        grade14 = 3
    elif grade > -25:
        grade14 = 2
    else:
        grade14 = 1
    print("grade, grade14", grade, grade14)
    return grade14, np.exp(nlu), exercisePNG(realSegments, predicted, chroma_pattern_model)
