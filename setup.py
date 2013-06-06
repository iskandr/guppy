#!/usr/bin/python

import glob
import platform
import sys
import os

from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
  name='guppy',
  version='1.0',
  maintainer='Russell Power',
  maintainer_email='russell.power@gmail.com',
  url='http://github.com/rjpower/guppy',
  install_requires=['pycuda'],
  description='...',
  package_dir={'': '.'},
  packages=['guppy'],
)
