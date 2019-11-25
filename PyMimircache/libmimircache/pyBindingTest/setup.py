from distutils.core import setup, Extension
setup(name="PyMimircache", version="1.0",
      ext_modules=[Extension("PyMimircache", ["readerType.c"])])