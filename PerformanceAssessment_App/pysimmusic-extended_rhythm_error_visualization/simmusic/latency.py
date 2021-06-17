import essentia.standard as ess
import numpy as np
from scipy.signal import find_peaks

from madmom.features.onsets import SpectralOnsetProcessor
from madmom.audio.filters import LogarithmicFilterbank


def remove_latency(in_audio_filename, out_audio_filename, latency, format='wav', sample_rate=44100):
    """
    Produces audiofile with latency removed.

    :param in_audio_filename: original file to process
    :param out_audio_filename: pathname to write the result
    :param latency: latency to remove in seconds
    :param format: format of the output file
    :param sample_rate: sample rate
    :return:
    """
    audio = ess.MonoLoader(filename=in_audio_filename)()
    frames = int(latency * sample_rate)
    audio = audio[frames:]
    ess.MonoWriter(filename=out_audio_filename, format=format)(audio)


def onsets(filename, fps=500):
    """
    Calculates onsets function (not actual onsets!)

    :param filename: audio file to process
    :param fps: frames per second (defines detection precision)
    :return: onsets array
    """
    sodf = SpectralOnsetProcessor(onset_method='complex_flux', fps=fps,
                                  filterbank=LogarithmicFilterbank,
                                  num_bands=24, log=np.log10)
    return sodf(filename)

def correlations(ref_onsets, sample_filename, fps=500, bpm=60):
    """
    Calculates correlation vector between onset function values for reference and sample sounds
    with all lags.

    :param ref_onsets: reference onsets function values.
    :param sample_filename: sample audio filename
    :param fps: frames per seconds (should be the same as for ref_onsets calculation)
    :param bpm: beats per minute
    :return: correlations array
    """
    others_onsets = onsets(sample_filename, fps)
    # period should be less than bpm cycle
    max_frame = int(fps * 60.0 / bpm * 0.75)
    n = min(len(ref_onsets), len(others_onsets))
    return [np.correlate(ref_onsets[:n], np.roll(others_onsets[:n], x))[0] for x in range(0, -max_frame, -1)]

def latency(ref_onsets, sample_filename, fps=500, bpm=60, kind='max', threshold=0.09):
    """
    Calculates latency by crosscorrelation peaks.

    :param ref_onsets: reference onsets vector
    :param sample_filename: sample onsets vector
    :param fps: frames per seconds
    :param bpm: clicks per minute
    :param xc: precalculated crosscorrelation, if any
    :param kind: 'nearest_to_zero' or 'max'
    :param threshold: height threshold (is effective only for 'nearest_to_zero' kind)
    :return: latency, crosscorrelation with lags array
    """
    xc = correlations(ref_onsets, sample_filename, fps=fps, bpm=bpm)
    if kind == 'nearest_to_zero':
        peaks, _ = find_peaks(xc/max(xc), height=threshold)
        l = peaks[0] / fps
    elif kind == 'max':
        l = float(np.argmax(xc)) / fps
    else:
        raise ValueError('kind must be either "max" or "nearest_to_zero"')
    return l, xc

