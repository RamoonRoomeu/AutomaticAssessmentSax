![build passing](https://app.codeship.com/projects/7ed135d0-8b20-0135-9500-563e3a0cc28a/status?branch=beta-kadenze)
This build is failing on the codeship servers because of the requirement of essentia :| once that is sorted out it becomes functional. 

# pysimmusic

## Introduction
Python tools for analysing similarity of music performances.

This repository contains utilities and algorithms to be used in CAMUT and TECSOME projects, but is separate to make it easier to develop.

# Installation
## Step 1: Virtualenv
It is recommended to install pysimmusic and dependencies into a virtualenv. We recommend to use python3. You can do it as follows:
```
virtualenv -p python3 envname
source envname/bin/activate
```
## Step 2: Requirements
```
pip install -r requirements.txt
```
## Step 3: Setuptools
Setuptools version should be > 25. On linux just run:
```
sudo apt-get install python3-setuptools
``` 
## Step 4: Install pysimmusic
```
python3 setup.py install
```

# Run tests
Tests can be run using pytest package
```
pytest tests
```

# Create documentation
Documentation (in docs folder) can be automatically generated using these commands
```
cd docs
make html
```
# Grading guitar chord exercises

Demo is in demo/demo_chords.py

Documentation is provided for the "entry-point" function:
simmusic.extractors.chords.estimate_grade_bpm

