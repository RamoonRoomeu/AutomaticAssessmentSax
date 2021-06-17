
class SegmentNotFound(Exception):
    """
    A class to handle custom exceptions
    """

    def __init__(self):
        Exception.__init__(self,"We could not detect any valid segment in the pitch data")