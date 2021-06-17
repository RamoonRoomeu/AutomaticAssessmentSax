from simmusic.feature_extraction import syllable_sequence_alignment
from simmusic.feature_extraction import syllable_sequence_recognition
from simmusic.feature_extraction import get_median_syllable_pitch
from simmusic.feature_extraction import check_octave_difference
from simmusic.feature_extraction import get_pyin_pitchtrack
from simmusic.feature_extraction import median_octave_correction
from simmusic.feature_extraction import get_segment_pitch_tracks
from simmusic.feature_extraction import convert_syl_2_letters
from simmusic.feature_extraction import convert_letters_2_num_list
from simmusic.scoring import convert_onsets_2_syllable_seq
from simmusic.scoring import syllable_seq_score_onset
from simmusic.scoring import syllable_individual_score_onset_equal_length
from simmusic.scoring import convert_kaldi_ctm_2_onset_seq
from simmusic.scoring import syllable_individual_score_pitch
from simmusic.utilities_io import parse_music_xml
from simmusic import utilities as utils
import nwalign3 as nw
import numpy as np
import os


class SolfegeGradeEstimator(object):
    """
    Compare teacher and student recording, give a score
    """

    def __init__(self,
                 output_file_path,
                 quarter_time,
                 segment_start,
                 segment_duration):
        """
        :param output_file_path: everything intermediate output goes here
        :param quarter_time: quarter note length time, in second
        :param segment_start: segment start time, in second
        :param segment_duration: segment duration, in second
        """
        self.output_file_path = output_file_path
        self.quarter_time = quarter_time
        self.segment_start = segment_start
        self.segment_duration = segment_duration
        self.syllable_seq_ref = None
        self.syllable_seq_stu = None
        self.score_overall_onset = None
        self.score_syllable_onset = None
        self.score_overall_pitch = None
        self.score_syllable_pitch = None
        self.dict_assessment_syllable_stu = None

        self.timestamps_ref = None
        self.timestamps_stu = None
        self.melody_cents_ref = None
        self.melody_cents_stu = None
        self.onsets_ref = None
        self.onsets_stu = None
        self.boundaries_ref = None
        self.boundaries_stu = None
        self.pitch_track_segments_stu = None
        self.median_pitch_ref_syllable = None
        self.median_pitch_stu_syllable = None
        self.boundaries_ref_align = None
        self.boundaries_stu_align = None
        self.median_syllable_pitch_ref_align = None
        self.median_syllable_pitch_stu_align = None
        self.align_recog = "recog"
        self.kaldi_root = "/opt/kaldi"
        self.hopsize_t = 0.025
        # self.kaldi_root = "/Users/ronggong/Documents_using/github/kaldi"

    def audio_resample(self, input_audio_filename):
        """
        Resample audio to 44100 Hz
        :param input_audio_filename: as the name
        :return:
        """
        fn = os.path.basename(input_audio_filename)
        data = utils.resample_kaldi(input_audio_filename=input_audio_filename,
                                    output_audio_file_path=os.path.join(self.output_file_path, "audio"),
                                    output_filename=fn,
                                    resampling_fs=44100)
        return data

    def get_syllable_seq_ref(self, input_xml_filename_ref):
        """
        :param input_xml_filename_ref: music xml full filename, reference
        :return:
        """

        syllables_ref, self.median_pitch_ref_syllable = \
            parse_music_xml(music_xml_file=input_xml_filename_ref,
                            quarter_time=self.quarter_time,
                            segment_start=self.segment_start)
        self.onsets_ref = convert_kaldi_ctm_2_onset_seq(syllables_ref)
        self.syllable_seq_ref = convert_onsets_2_syllable_seq(onsets=self.onsets_ref,
                                                              seq_duration=self.segment_duration,
                                                              segment_start=self.segment_start,
                                                              hopsize_t=self.hopsize_t)
        return self.syllable_seq_ref, self.onsets_ref

    def get_syllable_seq_stu(self, input_audio_filename_stu):
        """
        :param input_audio_filename_stu: wav full filename, student
        :return:
        """
        fn = os.path.basename(input_audio_filename_stu)
        fn = os.path.splitext(fn)[0]

        if self.align_recog == "recog":
            syllables_stu = syllable_sequence_recognition(kaldi_root=self.kaldi_root,
                                                          input_audio_file_name=fn,
                                                          output_file_path=self.output_file_path,
                                                          segment_start=self.segment_start)
        else:
            syllables_stu = syllable_sequence_alignment(kaldi_root=self.kaldi_root,
                                                        input_audio_file_name=fn,
                                                        output_file_path=self.output_file_path,
                                                        transcription_str=self.transcription_str,
                                                        segment_start=self.segment_start)
        self.onsets_stu = convert_kaldi_ctm_2_onset_seq(syllables_stu)
        self.syllable_seq_stu = convert_onsets_2_syllable_seq(onsets=self.onsets_stu,
                                                              seq_duration=self.segment_duration,
                                                              segment_start=self.segment_start,
                                                              hopsize_t=self.hopsize_t)
        return self.syllable_seq_stu, self.onsets_stu

    @staticmethod
    def onsets_2_boundaries(onsets):
        """
        Convert onset list to boundary list, label, assuming the last onset label is sil
        :param onsets: [[onset time, label], ...]
        :return: [[onset, offset, label], ...]
        """
        return [[onsets[ii][0], onsets[ii+1][0], onsets[ii][1]]
                for ii in range(len(onsets)-1) if onsets[ii][1] != "sil"]

    @staticmethod
    def boundaries_post_processing(boundaries):
        """
        Post processing the boundaries list, fix the consecutive syllable segments with the same name
        :param boundaries: python list [[onset, offset, label], ...]
        :return:
        """
        for ii in range(len(boundaries)-1, 0, -1):
            if boundaries[ii][0] == boundaries[ii-1][1] and \
                    boundaries[ii][2] == boundaries[ii-1][2]:
                boundaries[ii-1][1] = boundaries[ii][1]
                boundaries.pop(ii)

    def get_overall_score_onset(self):
        """
        :return: score
        """
        self.score_overall_onset = syllable_seq_score_onset(ground_truth_syllable_seq=self.syllable_seq_ref,
                                                            student_syllable_seq=self.syllable_seq_stu)
        return self.score_overall_onset

    def reference_feature_extraction(self, filename_input_ref_xml):
        # ---REFERENCE feature extraction part---
        # ---This part can be ran right after the recording interface launched---

        # ref syllable seq
        syllable_seq_ref, onset_ref = self.get_syllable_seq_ref(input_xml_filename_ref=filename_input_ref_xml)

        self.boundaries_ref = self.onsets_2_boundaries(self.onsets_ref)

    def student_feature_extraction(self, filename_input_stu_audio):
        # ---STUDENT feature extraction part---
        # ---It should be ran after student's recording been done---
        audio_stu = self.audio_resample(input_audio_filename=filename_input_stu_audio)

        # student syllable seg
        syllable_seq_stu, onset_stu = self.get_syllable_seq_stu(input_audio_filename_stu=filename_input_stu_audio)

        self.boundaries_stu = self.onsets_2_boundaries(self.onsets_stu)

        if self.align_recog == "recog":
            self.boundaries_post_processing(self.boundaries_stu)

        self.timestamps_stu, self.melody_cents_stu = get_pyin_pitchtrack(audio=audio_stu, fs=44100)

        self.pitch_track_segments_stu = get_segment_pitch_tracks(time_stamps=self.timestamps_stu,
                                                                 pitch_track=self.melody_cents_stu,
                                                                 boundaries=self.boundaries_stu)

        self.median_pitch_stu_syllable = get_median_syllable_pitch(time_stamps=self.timestamps_stu,
                                                                   pitch_track=self.melody_cents_stu,
                                                                   boundaries=self.boundaries_stu)

    def syllable_sequences_alignment(self):
        # ---syllable sequence alignment---
        # student assessment syllable-level dictionary
        self.dict_assessment_syllable_stu = \
            [{"note_name": None, "pitch_segment": None, "pronunciation": None, "pitch": None, "onset": None}
             for ii in range(len(self.boundaries_stu))]

        if self.align_recog == "recog":
            # convert syllable sequence to letter sequence for the alignment
            syllables_ref_letters, syllables_stu_letters, dict_letters_2_syl = \
                convert_syl_2_letters(syllables0=[s[2] for s in self.boundaries_ref],
                                      syllables1=[s[2] for s in self.boundaries_stu])

            # Needleman-Wunsch sequence-sequence alignment
            syllables_ref_letters_aligned, syllables_stu_letters_aligned = \
                nw.global_align(syllables_ref_letters, syllables_stu_letters)

            print(syllables_ref_letters_aligned)
            print(syllables_stu_letters_aligned)

            # fill the assessment score dictionary
            counter = 0
            counter_corresponding_syllables = 0
            for ii, letter in enumerate(syllables_stu_letters_aligned):
                if letter != "-":
                    self.dict_assessment_syllable_stu[counter]["note_name"] = dict_letters_2_syl[letter]
                    self.dict_assessment_syllable_stu[counter]["pitch_segment"] = self.pitch_track_segments_stu[counter]
                    if syllables_ref_letters_aligned[ii] == "-":
                        # if stu syllable is related to nothing in reference
                        self.dict_assessment_syllable_stu[counter]["pronunciation"] = 0.0
                        self.dict_assessment_syllable_stu[counter]["pitch"] = 0.0
                        self.dict_assessment_syllable_stu[counter]["onset"] = 0.0
                    elif syllables_ref_letters_aligned[ii] == syllables_stu_letters_aligned[ii]:
                        # if aligned syllables have the same name
                        self.dict_assessment_syllable_stu[counter]["pronunciation"] = 1.0
                        self.dict_assessment_syllable_stu[counter]["pitch"] = None
                        self.dict_assessment_syllable_stu[counter]["onset"] = None
                        counter_corresponding_syllables += 1
                    elif syllables_ref_letters_aligned[ii] != syllables_stu_letters_aligned[ii]:
                        # if aligned syllables have different names
                        self.dict_assessment_syllable_stu[counter]["pronunciation"] = 0.0
                        self.dict_assessment_syllable_stu[counter]["pitch"] = None
                        self.dict_assessment_syllable_stu[counter]["onset"] = None
                        counter_corresponding_syllables += 1
                    else:
                        pass
                    counter += 1

            num_syl_ref = convert_letters_2_num_list(syllables_letters_aligned=syllables_ref_letters_aligned)
            num_syl_stu = convert_letters_2_num_list(syllables_letters_aligned=syllables_stu_letters_aligned)

            assert len(num_syl_ref) == len(num_syl_stu)

            shared_syl_list = []
            for ii in range(len(num_syl_ref)):
                if num_syl_ref[ii] is not None and num_syl_stu[ii] is not None:
                    shared_syl_list.append([num_syl_ref[ii], num_syl_stu[ii]])

            assert counter_corresponding_syllables == len(shared_syl_list)

            self.boundaries_ref_align = [self.boundaries_ref[n[0]] for n in shared_syl_list]
            self.boundaries_stu_align = [self.boundaries_stu[n[1]] for n in shared_syl_list]
            self.median_syllable_pitch_ref_align = self.median_pitch_ref_syllable[[n[0] for n in shared_syl_list]]
            self.median_syllable_pitch_stu_align = self.median_pitch_stu_syllable[[n[1] for n in shared_syl_list]]

        else:
            # fill the assessment score dictionary
            for ii in range(len(self.boundaries_stu)):
                self.dict_assessment_syllable_stu[ii]["note_name"] = self.boundaries_stu[ii][2]
                self.dict_assessment_syllable_stu[ii]["pitch_segment"] = self.pitch_track_segments_stu[ii]
                self.dict_assessment_syllable_stu[ii]["pronunciation"] = 1.0
                self.dict_assessment_syllable_stu[ii]["pitch"] = None
                self.dict_assessment_syllable_stu[ii]["onset"] = None
            self.boundaries_ref_align = self.boundaries_ref
            self.boundaries_stu_align = self.boundaries_stu
            self.median_syllable_pitch_ref_align = self.median_pitch_ref_syllable
            self.median_syllable_pitch_stu_align = self.median_pitch_stu_syllable

    def onset_assessment(self):
        # ---ONSET assessment score---
        # overall onset score
        self.get_overall_score_onset()
        # individual onset score
        self.score_syllable_onset = \
            syllable_individual_score_onset_equal_length(boundaries_ref=self.boundaries_ref_align,
                                                         boundaries_stu=self.boundaries_stu_align,
                                                         tol=0.20)

    def pitch_assessment(self):
        # ---PITCH assessment score---
        # check octave difference between reference and student
        octave_diff, ref_higher = check_octave_difference(median_syllable_ref=self.median_syllable_pitch_ref_align,
                                                          median_syllable_stu=self.median_syllable_pitch_stu_align)

        # adjust ref octave to student octave
        if octave_diff:
            self.median_pitch_ref_syllable = \
                median_octave_correction(median_syllable_ref=self.median_pitch_ref_syllable,
                                         octave_diff=octave_diff,
                                         ref_higher=ref_higher)
            self.median_syllable_pitch_ref_align = \
                median_octave_correction(median_syllable_ref=self.median_syllable_pitch_ref_align,
                                         octave_diff=octave_diff,
                                         ref_higher=ref_higher)

        # individual syllable pitch score
        self.score_syllable_pitch = \
            syllable_individual_score_pitch(median_syllable_ref=self.median_syllable_pitch_ref_align,
                                            median_syllable_stu=self.median_syllable_pitch_stu_align,
                                            tol=1.0)

    def fill_the_rest_scores(self):
        # fill the rest individual syllable pitch and onset score
        counter = 0
        for ii in range(len(self.dict_assessment_syllable_stu)):
            if self.dict_assessment_syllable_stu[ii]["pitch"] is None and \
                    self.dict_assessment_syllable_stu[ii]["onset"] is None:
                self.dict_assessment_syllable_stu[ii]["pitch"] = self.score_syllable_pitch[counter]
                self.dict_assessment_syllable_stu[ii]["onset"] = self.score_syllable_onset[counter]
                counter += 1

    def score_plot_no_reference(self, visualization_output_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        import matplotlib.colors as mcol
        import matplotlib.cm as cm
        import matplotlib.patches as patches
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        cnorm = mcol.Normalize(vmin=0.0, vmax=1.0)
        cpick = cm.ScalarMappable(norm=cnorm, cmap="RdYlGn")
        cpick.set_array(np.array([]))

        fontsize = 15
        text_color = "k"
        plt.figure(figsize=(self.onsets_ref[-1][0] * 2, 3))
        gs = gridspec.GridSpec(1, 1, height_ratios=[1])

        segment_pitch_tracks_stu = [f["pitch_segment"] for f in self.dict_assessment_syllable_stu]
        score_syllable_pitch = [f["pitch"] for f in self.dict_assessment_syllable_stu]
        score_syllable_onset = [[f["note_name"], f["onset"]] for f in self.dict_assessment_syllable_stu]
        score_pronunciation = [f["pronunciation"] for f in self.dict_assessment_syllable_stu]

        # student
        ax1 = plt.subplot(gs[0])

        # draw teacher's note
        for ii in range(len(self.boundaries_ref)):
            note_color = "#999999"
            rect = patches.Rectangle((self.boundaries_ref[ii][0],
                                     self.median_pitch_ref_syllable[ii]-0.25),
                                     self.boundaries_ref[ii][1]-self.boundaries_ref[ii][0],
                                     0.5,
                                     linewidth=1,
                                     edgecolor=note_color,
                                     facecolor=note_color,
                                     alpha=0.5)
            ax1.add_patch(rect)
            plt.text(self.boundaries_ref[ii][0],
                     self.median_pitch_ref_syllable[ii]-0.25,
                     self.boundaries_ref[ii][2],
                     color=text_color)

        for ii, (time_stamps, melody_cents) in enumerate(segment_pitch_tracks_stu):
            hex_color = '#%02x%02x%02x' % cpick.to_rgba(score_syllable_pitch[ii], bytes=True)[:3]
            ax1.plot(time_stamps, melody_cents, color=hex_color)

        y_high = ax1.get_ylim()[1]
        y_low = ax1.get_ylim()[0]
        for ii in range(len(self.boundaries_stu)):
            hex_color = '#%02x%02x%02x' % cpick.to_rgba(score_syllable_onset[ii][1], bytes=True)[:3]
            lower_bound_onset_stu = (self.median_pitch_stu_syllable[ii]-1.0-y_low)/(y_high-y_low)
            higher_bound_onset_stu = (self.median_pitch_stu_syllable[ii]+1.0-y_low)/(y_high-y_low)
            plt.axvline(self.boundaries_stu[ii][0],
                        lower_bound_onset_stu,
                        higher_bound_onset_stu,
                        color=hex_color)
            text_color_stu = "g" if score_pronunciation[ii] == 1 else "r"
            plt.text(self.boundaries_stu[ii][0],
                     self.median_pitch_stu_syllable[ii] + 1.0,
                     self.boundaries_stu[ii][2],
                     color=text_color_stu)

        ax1.set_ylabel("Student", fontsize=fontsize)
        # ax1.set_xlabel("Time (s)", fontsize=fontsize)
        ax1.axis('tight')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="1%", pad=0.05)

        plt.colorbar(cpick, cax=cax)

        plt.savefig(visualization_output_path, bbox_inches='tight')
        plt.show()
