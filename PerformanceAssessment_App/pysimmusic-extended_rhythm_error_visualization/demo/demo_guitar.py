import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import simmusic.extractors.guitar as guitar
import simmusic.latency as ltncy

import os

# The path to the test data
pysim_chords_test_data_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'data/guitar'))


def demo_latency():
    # calculate reference onsets
    onsets = ltncy.onsets(os.path.join(pysim_chords_test_data_dir, '147_backing_track.mp3'))
    latency, _ = ltncy.latency(
        onsets,
        os.path.join(pysim_chords_test_data_dir,
                     '227_recording-0-2019-03-20T10-22-36-312Z-0.wav'))
    print("Latency: ", latency)
    return latency


def demo_strumming(latency = 0.075):
    results =\
        guitar.assess_strumming_exercise(
            os.path.join(pysim_chords_test_data_dir, 'l2ex2.json'),
            os.path.join(pysim_chords_test_data_dir, 'l2ex2.ly'),
            latency,
            os.path.join(pysim_chords_test_data_dir, '237_recording-0-2019-03-20T10-42-59-548Z-0.wav'))
    print("#########################################################################")
    print("Strumming Grades: Overall, Pitch, Rhythm, Tuning")
    print(results["Overall"], results["Pitch"], results["Rhythm"], results["Tuning"])
    with open("strumming.png", "wb") as out_file:
       out_file.write(results["ImageBytes"])
    print("PNG is written to strumming.png")
    print("#########################################################################")


def demo_picking(latency = 0.075):
    results =\
        guitar.assess_picking_exercise(
            os.path.join(pysim_chords_test_data_dir, 'l3ex1.json'),
            os.path.join(pysim_chords_test_data_dir, 'l3ex1.ly'),
            latency,
            os.path.join(pysim_chords_test_data_dir, '288_recording-0-2019-03-20T11-49-56-632Z-0.wav'))
    print("#########################################################################")
    print("Picking Grades: Overall, Pitch, Rhythm, Tuning")
    print(results["Overall"], results["Pitch"], results["Rhythm"], results["Tuning"])
    with open("picking.png", "wb") as out_file:
       out_file.write(results["ImageBytes"])
    print("PNG is written to picking.png")
    print("#########################################################################")

def demo_excessive(latency = 0.075):
    results =\
        guitar.assess_picking_exercise(
            os.path.join(pysim_chords_test_data_dir, 'l2ex1.json'),
            os.path.join(pysim_chords_test_data_dir, 'l2ex1.ly'),
            latency,
            os.path.join(pysim_chords_test_data_dir, '384_recording-0-2019-03-20T16-09-51-732Z-0.wav'))
    print("#########################################################################")
    print("Picking Grades: Overall, Pitch, Rhythm, Tuning")
    print(results["Overall"], results["Pitch"], results["Rhythm"], results["Tuning"])
    with open("excessive.png", "wb") as out_file:
       out_file.write(results["ImageBytes"])
    print("PNG is written to excessive.png")
    print("#########################################################################")

def demo_silence(latency = 0.075):
    results =\
        guitar.assess_picking_exercise(
            os.path.join(pysim_chords_test_data_dir, 'l3ex1.json'),
            os.path.join(pysim_chords_test_data_dir, 'l3ex1.ly'),
            latency,
            os.path.join(pysim_chords_test_data_dir, '288_silence.wav'))

    print("#########################################################################")
    print("Picking Grades: Overall, Pitch, Rhythm, Tuning")
    print(results["Overall"], results["Pitch"], results["Rhythm"], results["Tuning"])
    with open("silence.png", "wb") as out_file:
       out_file.write(results["ImageBytes"])
    print("PNG is written to silence.png")
    print("#########################################################################")


"""
def demo_strange(latency = 0.075):
    results =\
        guitar.assess_picking_exercise(
            "/home/seva/pysidemo_excessivemmusic-experiments/guitar_for_beginners/data/exercises/Leah17Dec2018/Lesson02Ex1/l2ex1.json",
            "/home/seva/pysimmusic-experiments/guitar_for_beginners/data/exercises/Leah17Dec2018/Lesson02Ex1/l2ex1.ly",
            latency,
            "/home/seva/Downloads/136_d04c9400590d4ca2b6ff2f5601937b10.wav")

    print("#########################################################################")
    print("Picking Grades: Overall, Pitch, Rhythm, Tuning")
    print(results["Overall"], results["Pitch"], results["Rhythm"], results["Tuning"])
    with open("picking.png", "wb") as out_file:
       out_file.write(results["ImageBytes"])
    print("PNG is written to picking.png")
    print("#########################################################################")
"""

latency = demo_latency()
#demo_strange(latency)
demo_strumming(latency)
demo_picking(latency)
demo_silence(latency)
demo_excessive(latency)


################
#from simmusic.extractors import guitar
#import simmusic
#pysim_chords_test_data_dir = os.path.abspath('data/guitar')
#from pychord_tools.models import load_model
#assessment_workflow = load_model(os.path.join(simmusic.__path__[0], 'extractors/guitar_models/strumming_workflow.pkl'))
#
#timing_features = feature_extraction.timing_features(
#    os.path.join(pysim_chords_test_data_dir, 'l2ex1.json'),
#    os.path.join(pysim_chords_test_data_dir, '384_recording-0-2019-03-20T16-09-51-732Z-0.wav'),
#    max_spectral_centroid=assessment_workflow.onset_max_spectral_centroid,
#    onset_threshold=assessment_workflow.onset_threshold,
#    series_delta=assessment_workflow.onset_series_delta)
#
#results = guitar.assess_guitar_exercise(
#    os.path.join(pysim_chords_test_data_dir, 'l2ex2.json'),
#    os.path.join(pysim_chords_test_data_dir, 'l2ex2.ly'),
#    0.075,
#    os.path.join(pysim_chords_test_data_dir, '237_recording-0-2019-03-20T10-42-59-548Z-0.wav'))
#    assessment_workflow)
#
