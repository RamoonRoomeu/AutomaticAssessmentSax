import numpy as np
import re
from pychord_tools.labels import PITCH_CLASS_NAMES, LabelTranslator, SHORTCUTS, UNCLASSIFIED, note_to_number

###########################################################################
# Lables for the guitar course
###########################################################################

GUITAR_KINDS = ['maj', 'min', '5', '1']
GUITAR_MIREX_KINDS = ['', ':min', ':(5)', ':(1)']
GUITAR_NAMES = np.empty(48, dtype='object')
for p in range(len(PITCH_CLASS_NAMES)):
    for c in range(len(GUITAR_MIREX_KINDS)):
        GUITAR_NAMES[p * 4 + c] = PITCH_CLASS_NAMES[p] + GUITAR_KINDS[c]


class GuitarLabelTranslator(LabelTranslator):
    MAJ_DEGREES = set(('3', '5'))
    MIN_DEGREES = set(('b3', '5'))
    POWER_DEGREES = set(('5'))
    SINGLE_DEGREES = set(('1'))

    def chord_kinds(self):
        return GUITAR_KINDS

    def chord_mirex_kinds(self):
        return GUITAR_MIREX_KINDS

    def chord_names(self):
        return GUITAR_NAMES

    def chord_kinds_number(self):
        return len(GUITAR_KINDS)

    def chords_number(self):
        return len(GUITAR_NAMES)

    def label_to_pitch_and_kind(self, label):
        parts_and_bass = label.split('/')
        parts = parts_and_bass[0].split(':')
        note = parts[0]
        if note[0] == 'N':
            return 9, 'unclassified'
        pitch = note_to_number(note)
        if len(parts) == 1:
            kind = 'maj'
        else:
            kind = parts[1].split('/')[0]
        if kind in SHORTCUTS:
            kind = SHORTCUTS[kind]
        degrees = set(re.sub("[\(\)]", "", kind).split(','))
        # TODO after the dataset is fixed (bass -> pitch class set).
        if len(parts_and_bass) > 1:
            degrees.add(parts_and_bass[1])
        if len(parts_and_bass) > 1:
            degrees.add(parts_and_bass[1])
        if degrees == self.MAJ_DEGREES:
            kind = 'maj'
        elif degrees == self.MIN_DEGREES:
            kind = 'min'
        elif degrees == self.POWER_DEGREES:
            kind = '5'
        elif degrees == self.SINGLE_DEGREES or len(degrees) == 0:
            kind = '1'
        else:
            kind = UNCLASSIFIED
        return pitch, kind


