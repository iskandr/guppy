#!/usr/bin/python

import glob
import platform
import sys
import os

from setuptools import setup, Extension

setup(
  name='honeycomb',
  version='0.0.1',
  maintainer='Russell Power',
  maintainer_email='russell.power@gmail.com',
  url='https://github.com/iskandr/honeycomb', 
  install_requires=['pycuda'],
  description='...',
  package_dir={'': '.'},
  packages=['honeycomb'],
)
