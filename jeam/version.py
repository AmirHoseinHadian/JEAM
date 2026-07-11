from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 0  # use '' for first of series, number for 1 and above
_version_extra = 'alpha'  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "CRDDM: A package for modeling continuous response tasks using diffusion decision models"
# Long description will go up on the pypi page
long_description = """

CRDDM
========
``CRDDM`` is a Python package for modeling reponse time and choice in continuous 
response tasks using diffusion decision models. 

The main aim of this package is to provide an easy to ues implementation for the 
likelihood functions of continuous response models. For the models that do not 
have exact likelihood function we used an integral equaiton method.

Documentation
=============
The latest documentation can be found here: https://crddm.readthedocs.io/

License
=======
``CRDDM`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2025--, Amir Hosein Hadian Rasanan,
University of Basel.
"""

NAME = "CRDDM"
MAINTAINER = "Amir Hosein Hadian Rasanan"
MAINTAINER_EMAIL = "amir.h.hadian@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/amirhoseinhadian/crddm"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Amir Hosein Hadian Rasanan"
AUTHOR_EMAIL = "amir.h.hadian@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'CRDDM': [pjoin('data', '*')]}
REQUIRES = ["numpy", "scipy", "numba", "pandas"]