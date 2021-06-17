import numpy as np
import codecs


def parse_text_feedback_file(filename):
    """
    This function parses feedback file written in a text format (flat notes one)
    :param filename:
    :return:
    """
    fid = codecs.open(filename, 'r', encoding='utf-8')
    lines = fid.readlines()
    feedback = []
    for line in lines:
        sline = line.split('\t')
        feedback.append((float(sline[0].strip()), float(sline[1].strip()), float(sline[2].strip())))
    fid.close()
    return feedback


def compare_text_feedback_files(file1, file2):
    """
    This function compares two feedback files if they are same. Note that only the time stamps and scores are compared
    and not the feedback string as that can change across different versions of the app
    :param file1:
    :param file2:
    :return:
    """
    feedback1 = parse_text_feedback_file(file1)
    feedback2 = parse_text_feedback_file(file2)

    if len(feedback1) != len(feedback2):
        print("Feedback files have different lengths")
        return False
    # comparing files
    for ii, elem in enumerate(feedback1):
        diff = np.abs(np.round(np.array(feedback1[ii][:3]), 3) - np.round(np.array(feedback2[ii][:3]), 3))
        if np.max(diff) > 0.01:
            print("Two feedback files are not same at line %d\n" % ii)
            return False
    return True
