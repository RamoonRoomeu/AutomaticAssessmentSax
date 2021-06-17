import joblib
import numpy as np

from essentia import *
from essentia.standard import *
from madmom.features.onsets import SpectralOnsetProcessor, OnsetPeakPickingProcessor
from madmom.audio.filters import LogarithmicFilterbank
from scipy.signal import find_peaks
import madmom.audio.signal as signal
from sklearn import preprocessing


def rms_centroids(filename, frameSize = 1024, hopSize = 512, sampleRate=44100):
    # load our audio into an array
    audio = MonoLoader(filename=filename, sampleRate=44100)()

    # create the pool and the necessary algorithms
    w = Windowing()
    spec = Spectrum()
    rms = RMS()
    centroid = Centroid(range=int(sampleRate/2))
    cs = []
    rmss = []
    # compute the centroid for all frames in our audio and add it to the pool
    for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize):
        sf = spec(w(frame))
        cs.append(centroid(sf))
        rmss.append(rms(sf))
    return np.array(rmss), np.array(cs)


def combine_series(events, delta):
    """
    Combine all events of series with inner distances
    less then delta.


    Parameters
    ----------
    events : list or numpy array
        Events to be combined.
    delta : float
        Combination delta.
    Returns
    -------
    numpy array
        Combined events.

    """
    # add a small value to delta, otherwise we end up in floating point hell
    delta += 1e-12
    # return immediately if possible
    if len(events) <= 1:
        return events
    # create working copy
    events = np.array(events, copy=True)
    # set start position
    idx = 0
    # iterate over all remaining events
    for right in events[1:]:
        if right - events[idx] > delta:
            idx += 1
        events[idx] = right
    return events[:idx + 1]


class GuitarOnsetDetector:
    # TODO: ad hoc!!!! Derive it automatically.
    def __init__(
            self,
            max_spectral_centroid = 3500,
            onset_threshold = 0.7,
            series_delta=0.22):
        self.max_spectral_centroid = max_spectral_centroid
        self.onset_threshold = onset_threshold
        self.series_delta = series_delta

    def predict(self, audiofile, start, end, fs=44100):

        # Onset detection using Spectral Onset Processor
        #def onset_SOP(audiofile, delta, method):
        # Detection function
        fps = 200
        sodf = SpectralOnsetProcessor('superflux', diff_frames=20)
        sodf.processors[-1]  # doctest: +ELLIPSIS
        det_function = sodf(audiofile, fps=fps)
        det_function_norm = det_function / (max(det_function))

        # Dynamic threashold
        C_t = 0.99
        H = 100
        delta = 0.1

        din_th = np.zeros(len(det_function_norm))
        for m in range(H, len(det_function_norm)):
            din_th[m] = C_t * np.median(det_function_norm[m - H:m + H]) + delta

        # Peak detection
        peaks, _ = find_peaks(det_function_norm, distance=fps / 10, height=din_th)
        onset_array = peaks / fps

        return onset_array

    # def predict(self, audio_filename, start, end, fs = 44100):
    #     # fps must be a divisor of fs to obtain integer hopSize
    #     # (it just simplifies the code below)
    #     fps=180
    #     fs = 44100
    #     hopSize = int(fs/fps)
    #     sodf = SpectralOnsetProcessor(onset_method='superflux', fps=fps,
    #                                   filterbank=LogarithmicFilterbank,
    #                                   num_bands=24, log=np.log10)
    #     sodf_onsets = sodf(audio_filename)
    #     # "fusion" with rms-diff.
    #     rms, cs = rms_centroids(audio_filename, frameSize=1024, hopSize=hopSize, sampleRate=fs)
    #     rms = signal.smooth(rms, int(fs / hopSize * 0.2))
    #     rms = preprocessing.scale(rms, with_mean=False, copy=False)
    #     rms = rms[1:] - rms[:-1]
    #
    #     sodf_onsets[rms <= 0] = 0
    #     #sodf_onsets = sodf_onsets * np.power(rms, 0.01)
    #     #sodf_onsets[np.isnan(sodf_onsets)] = 0
    #
    #     proc = OnsetPeakPickingProcessor(
    #         fps=fps, threshold=self.onset_threshold)
    #     p_onsets = proc(sodf_onsets)
    #     p_onsets = combine_series(p_onsets, self.series_delta)
    #     p_onsets = p_onsets[(p_onsets >= start) & (p_onsets <= end)]
    #     smoothed = []
    #     for i in range(len(p_onsets)):
    #         onset = p_onsets[i]
    #         duration = 0.5
    #         if (i < len(p_onsets) - 1):
    #             duration = min((p_onsets[i + 1] - p_onsets[i]), duration)
    #         window_len = int(duration * fs / hopSize)
    #         s = int(float(onset) * fs / hopSize)
    #         d = min(window_len, len(cs) - s)
    #         w = eval('np.hanning(2*d)')
    #         w = w[d:] / np.sum(w[d:])
    #         w = np.reshape(w, (1, d))
    #         c = cs[s:s + d]
    #         smoothed.append(np.dot(w, c)[0])
    #     result = []
    #     for i in range(len(p_onsets)):
    #         if smoothed[i] < self.max_spectral_centroid:
    #             result.append(p_onsets[i])
    #     return result

    def save_model(self, file_name):
        joblib.dump(self, file_name)
