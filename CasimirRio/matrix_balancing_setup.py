from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("matrix_balancing",
                             ["matrix_balancing.pyx"])]
    )
