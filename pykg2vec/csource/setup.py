from distutils.core import setup, Extension

setup(name='file_handler', version='1.0', ext_modules=[Extension('file_handler', ['file_handler.c'])])
