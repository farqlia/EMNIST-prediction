#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup
import re, io

# run pip install -e . to make this a package

# solution found in https://stackoverflow.com/a/39671214
# not the best, but it works
__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    io.open('emnist_prediction/__init__.py', encoding='utf_8_sig').read()
    ).group(1)


with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    required = f.read().splitlines()

setup(
    name='emnist-prediction',
    version=__version__,
    packages=['emnist_prediction'],
    description='EMNIST Prediction.' ,
    author='Julia Farganus',
    author_email='juliafarganus@gmail.com',
    maintainer_email='juliafarganus@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3.10'
    ],
    # long_description=open(os.path.join(os.path.dirname(__file__), 'index.rst')).read(),
    install_requires=required,
)