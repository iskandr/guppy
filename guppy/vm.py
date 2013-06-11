#!/usr/bin/env python
import os

from pycuda import autoinit

import pycuda.compiler
import pycuda.driver

here = os.path.abspath(__file__)
source_file = os.path.dirname(here) + '/vm.cu'
mod = pycuda.compiler.SourceModule(open(source_file).read())