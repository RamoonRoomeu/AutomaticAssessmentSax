#!/usr/bin/env python

from setuptools import find_packages, setup, Extension
from distutils.sysconfig import *
import numpy

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

try:
    from Cython.Distutils import build_ext
except ImportError:
    from distutils.command import build_ext
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

py_inc = [get_python_inc()]

if use_cython:
    ext_modules += [
        Extension("simmusic.dtw.dtw", ["simmusic/dtw/cdtw.pyx", "simmusic/dtw/dtw.c"],
                  include_dirs=['simmusic/dtw', numpy.get_include()] + py_inc),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("simmusic.dtw.dtw", ["simmusic/dtw/dtw.c"],
                  include_dirs=['simmusic/dtw', numpy.get_include()] + py_inc),
    ]


setup(name='simmusic',
      version='0.4-beta',
      description='Tool for automatic assessment of musical performances',
      author='TECSOME / MTG UPF',
      packages=find_packages(),
      test_suite="tests",
      install_requires=['numpy', 'Cython', 'vamp', 'resampy',
          'nwalign3', 'matplotlib', 'joblib', 'xmltodict',
          'music21', 'sndfile', 'madmom', 'Pillow'
      ],
      package_date={'simmusic': ['extractors/notes_singing_models/*.joblib']},
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      setup_requires=['pytest-runner'],
      include_package_data=True
      )
