import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("cythonized/gradient.pyx", compiler_directives={"language_level":"3"}), include_dirs=[numpy.get_include()],)