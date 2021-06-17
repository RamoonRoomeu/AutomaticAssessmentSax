"""These classes define containers to hold data in a convenient form (for example, notes, segments etc)
"""
import numpy as np
import os
import json
from simmusic import utilities as utils
from simmusic import utilities_io
from simmusic.constants import RECORDING_MODES
from simmusic import constants


# TODO: improve import statements, take care of he relative imports, import from the installed copy, or from a package


# TODO: add a note type, which can either be a flat note or an ornament
class Note(object):
    """A note event class.

    Attributes
    ----------
    pitch : float
        Note pitch (Hz).
    cent : float
        Note pitch (in cents).
    start : float
        Note starting time (seconds).
    end : float
        Note ending time (seconds).
    symbol : str
        Note symbol (E, F or 'Sa' etc).
    start_ind : int
        Starting index of the Note.
    end_ind : int
        Ending index of the Note.
    midi_number : int
        Midi number of the note.

    """

    def __init__(self, pitch, start, end, cent=None, symbol=None, start_ind=None, end_ind=None, midi_number=None):
        self.pitch = pitch
        self.start = start
        self.end = end
        self.cent = cent
        self.symbol = symbol
        self.start_ind = start_ind
        self.end_ind = end_ind
        self.midi_number = midi_number

    def __repr__(self):
        return 'Note(start={:f}, end={:f}, pitch={}, cent={}, symbol={}, start_ind={}, end_ind={})'.format(
            self.start,
            self.end,
            self.pitch,
            self.cent,
            self.symbol,
            self.start_ind,
            self.end_ind)


class Segment(object):
    """A time segment class.

    Attributes
    ----------
    start : float
        Segment starting time (seconds).
    end : float
        Segment ending time (seconds).
    seg_type : str (optional)
        Segment type (qualifier), Options: `phrase`, `silence`
    notes : list
        List of :class:`simmusic.Note` objects
    start_ind : int
        Starting index of the Segment.
    end_ind : int
        Ending index of the Segment.

    """

    def __init__(self, start, end, seg_type, start_ind=None, end_ind=None):
        self.start = start
        self.end = end
        self.seg_type = seg_type
        self.notes = []
        self.start_ind = start_ind
        self.end_ind = end_ind

    def __repr__(self):
        return 'Segment(start={:f}, end={:f}, type={}, notes={}, start_ind={}, end_ind={})'.format(self.start, self.end,
                                                                                                   self.seg_type,
                                                                                                   '<%s of length %d>' %
                                                                                                   (type(
                                                                                                       self.notes).
                                                                                                    __name__,
                                                                                                    len(self.notes)),
                                                                                                   self.start_ind,
                                                                                                   self.end_ind)


class Feature(object):
    """Acoustic feature class.

    Attributes
    ----------
    time : np.ndarray
        Time stamps of the extracted acoustic feature(s)
    pitch : np.ndarray
        Pitch samples array corresponding to the audio recording

    """

    def __init__(self, time=None, pitch=None):
        self.time = time
        self.pitch = pitch
        self.cents = np.array([])

    def __repr__(self):
        return 'Feature(time={}, pitch={}, cents={})'.format('%s of shape %s' % (type(self.time), self.time.shape),
                                                             '%s of shape %s' % (type(self.pitch), self.pitch.shape),
                                                             '%s of shape %s' % (type(self.cents), self.cents.shape))

    def convert_pitch_hz2cents(self, tonic, threshold, invalid_val):
        """Converts pitch from Hz to Cents.

        This function converts the pitch feature from hz scale to Cent scale (1200 cents in one octave).

        Parameters
        ----------
        tonic : float
            The reference frequency for hz 2 cents conversion
        threshold : float
            Minimum value of a pitch sample to be considered as a voiced region
        invalid_val : float
            Value of the pitch in cents to be assigned for the unvoiced regions

        Returns
        -------
        success : bool
            True or False based on the success of the operation

        """
        if self.pitch is not None:
            self.cents = utils.hz2cents_array(self.pitch, tonic, threshold, invalid_val)
        else:
            print("You should load pitch data (Hz) before converting it to cents...")
            return False
        return True


class Recording(object):
    """Object for representing a recording.

     This object comprises features and related properties (transcription/segments etc)

    Attributes
    ----------
    rec_type : str
        Type of recording, either 'reference' or a 'imitation'
    mode : str
        Mode of recording; 'play-along', 'dialouge' or 'improvisation'
    segments : list
        List of :class:`simmusic.Segment` objects
    feature : :class:`simmusic.Feature`
        Instance of :class:`simmusic.Feature` object

    """

    def __init__(self, rec_type, mode):
        self.rec_type = rec_type
        self.mode = mode
        self.segments = []
        self.feature = None

        if self.mode not in RECORDING_MODES:
            print("Provide a valid recoding mode to continue...")

    def __repr__(self):
        return 'Recording (type={}, mode={}, segments={}, feature={})'.format(self.rec_type, self.mode,
                                                                              '<%s of length %d>' % (
                                                                                  type(self.segments).__name__,
                                                                                  len(self.segments)), self.feature)

    def load_pitch_data(self, pitch_file):
        """Loads the pitch features.

        Parameters
        ----------
        pitch_file : str
            Name of the file from which the pitch data is read

        Returns
        -------
        success : bool
            True or False based on the success of the operation

        """
        # parsing pitch file
        if os.path.isfile(pitch_file):
            tp = np.loadtxt(pitch_file)
            self.feature = Feature(time=tp[:, 0], pitch=tp[:, 1])
        else:
            utilities_io.error_opening_file(pitch_file)
            return False
        return True

    def load_pitch_data_from_previous_extraction(self, pitch_file):
        """
        Loads the pitch features from pre-computed file
        :param pitch_file:
        :return:
        """
        # parsing pitch file
        if os.path.isfile(pitch_file):
            pitch = np.array(json.load(open(pitch_file))["pitch"])
            time = np.array(np.arange(pitch.size)) * np.float(512) / 44100
            self.feature = Feature(time=time, pitch=pitch)
        else:
            utilities_io.error_opening_file(pitch_file)
            print("Please provide a valid pitch file. Returning False...")
            return False
        return True

    def load_trans_data(self, trans_file):
        """Loads the transcription data.

        Parameters
        ----------
        trans_file : str
            Name of the transcription file from which note data is read

        Returns
        -------
        success : bool
            True or False based on the success of the operation

        Note
        ----
        Currently the parsing assumes a text file and a particular structure. Change it in the future and make
        it generic
        """
        # parsing trans_file (#TODO: this file should later change to a marter json doc)
        notes = utilities_io.parse_transcription_file(trans_file)
        if notes is None:
            return False
        self.segments.append(Segment(start=notes[0].start, end=notes[-1].end, seg_type='phrase'))
        self.segments[-1].notes = notes
        return False

    def compute_segment_indexes(self):
        """Computing array index of segments.

        Returns
        -------
        success : bool
            True or False based on the success of the operation

        """
        # noinspection PyBroadException
        try:
            for segment in self.segments:
                segment.start_ind = np.argmin(np.abs(self.feature.time - segment.start))
                segment.end_ind = np.argmin(np.abs(self.feature.time - segment.end))
        except:
            print('Something went wrong while trying to compute segment indexes')
            return False
        return True

    def compute_note_indexes(self):
        """Computing array index of Notes.

        Returns
        -------
        success : bool
            True or False based on the success of the operation

        """
        # noinspection PyBroadException
        try:
            for segment in self.segments:
                for note in segment.notes:
                    note.start_ind = np.argmin(np.abs(self.feature.time - note.start))
                    note.end_ind = np.argmin(np.abs(self.feature.time - note.end))
                    if np.abs(self.feature.time[note.start_ind] - note.start) >= constants.MIN_TIME_DIFF_ALLOWED:
                        note.start_ind = constants.INVALID_INDEX
                    if np.abs(self.feature.time[note.end_ind] - note.end) >= constants.MIN_TIME_DIFF_ALLOWED:
                        note.end_ind = constants.INVALID_INDEX
        except:
            print("Something went wrong while trying to compute note indexes")
            return False
        return True

    def convert_pitch_hz2cents(self, tonic, threshold, invalid_val):
        """Converting pitch from Hz to Cents.

        This function converts the pitch feature from hz scale to Cent scale (1200 cents in one octave).

        Parameters
        ----------
        tonic : float
            The reference frequency for hz 2 cents conversion
        threshold : float
            Minimum value of a pitch sample to be considered as a voiced region
        invalid_val : float
            Value of the pitch in cents to be assigned for the unvoiced regions

        Returns
        -------
        success : bool
            True or False based on the success of the operation

        """
        success = self.feature.convert_pitch_hz2cents(tonic, threshold, invalid_val)
        return success

    def convert_notes_hz2cents(self, tonic):
        """Converting note frequency from Hz to Cents.

        This function converts note frequency from hz scale to Cent scale (1200 cents in one octave).

        Parameters
        ----------
        tonic : float
            The reference frequency for hz 2 cents conversion

        Returns
        -------
        success : bool
            True or False based on the success of the operation

        """
        # noinspection PyBroadException
        try:
            for seg in self.segments:
                for note in seg.notes:
                    note.cent = utils.hz2cents(note.pitch, tonic)
        except:
            print("Something went wrong while converting note frequncy from Hz to Cents scale")
            return False
        return True

    def get_all_note_pitches(self):
        """Collecting all note pitches in an array

        Returns
        ----------
            Numpy array containing note pitches
        """
        pitches = []
        for seg in self.segments:
            for note in seg.notes:
                pitches.append(note.pitch)
        return np.array(pitches)

    def correct_octave_errors_in_trans(self):
        """Corrects octave errors in transcription by moving each pitch within octave
        closeness to the median pitch
        @author: Baris Bozkurt

        Outputs
        ----------
            Modifies the note frequencies in the Recording object (see notes on top)
        Returns
        -------
        success : bool
            True or False
        """
        # TODO: should we replace while by if statements?, This operation can be done more efficiently using numpy
        pitches = self.get_all_note_pitches()

        # note by note octave correction, condition: all notes should be with octave distance to the median
        median_pitch = np.median(pitches)
        for ind in range(len(self.segments[0].notes)):
            while (self.segments[0].notes[ind].pitch / median_pitch) > 2:
                self.segments[0].notes[ind].pitch /= 2
            while (self.segments[0].notes[ind].pitch / median_pitch) < 0.5:
                self.segments[0].notes[ind].pitch *= 2

        # shifting of the complete transcription if median is larger than 1200
        while np.median(self.get_all_note_pitches()) > 1200:
            for ind in range(len(self.segments[0].notes)):
                self.segments[0].notes[ind].pitch /= 2

        return True

# TODO: containers for alignment and feedback!

class Syllable(object):
    """
    A syllable event class
    """
    def __init__(self, start, duration, label):
        self.start = start
        self.duration = duration
        self.label = label

    def __repr__(self):
        return 'Note(start={:f}, duration={:f}, label={})'.format(self.start, self.duration, self.label)