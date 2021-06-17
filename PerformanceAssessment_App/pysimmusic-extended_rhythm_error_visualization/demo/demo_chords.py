"""
Demo: "how to evaluate student's guitar chord playing?"
"""

import simmusic.extractors.chords as mc_grade
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import os

# The path to the test data
pysim_chords_test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests/data/chords'))

referencePathname = os.path.join(pysim_chords_test_data_dir, 'audio/seva_demo.mp3')
studentPathname = os.path.join(pysim_chords_test_data_dir, 'audio/502.mp3')

grade, chords_scores, png_bytes =\
    mc_grade.estimate_grade_bpm(
        # recoridngs have
        60, # 60 bpm tempo
        4,  # 4 beats countdown in the begining
        32, # 32 beats i  total to evaluate
        referencePathname,
        studentPathname)

print("Grade: ", grade)
print("Chord scores: ", chords_scores)

with open('demo.png', 'wb') as f:
    f.write(png_bytes)

# Visualization plot contains:
# Upper X-labels: reference chords and time stamps.
# Top color bar: overall grade.
# Middle color bar: intensity of tones which are "outside" of chord.
# Low color bar: measure of tones balance.
# Bottob x-labels: if there are supposed student's mistakes:
#                  chords, which are actually played instead of the reference chords.
print("Visualization is written to: demo.png")
