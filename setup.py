import sys
from numpy.distutils.core import Extension, setup


__author__ = "Anders S. Christensen"
__copyright__ = "Copyright 2016"
__credits__ = ["Anders S. Christensen et al. (2016) https://github.com/qmlcode/qml"]
__license__ = "MIT"
__version__ = "0.4.0.12"
__maintainer__ = "Anders S. Christensen"
__email__ = "andersbiceps@gmail.com"
__status__ = "Beta"
__description__ = "Quantum Machine Learning"
__url__ = "https://github.com/qmlcode/qml"


FORTRAN = "f90"

# GNU (default)
COMPILER_FLAGS = ["-O3",  "-m64", "-march=native", "-fPIC",
                    "-Wno-maybe-uninitialized", "-Wno-unused-function", "-Wno-cpp"]
LINKER_FLAGS = ["-lgomp"]


# For clang without OpenMP: (i.e. most Apple/mac system)
if sys.platform == "darwin" and all(["gnu" not in arg for arg in sys.argv]):
    COMPILER_FLAGS = ["-O3", "-m64", "-march=native", "-fPIC"]
    LINKER_FLAGS = []


ext_ = Extension(name = 'ml_tools.descriptors.radial_numerial_integration.ge',
                sources = [
                      'ml_tools/descriptors/radial_numerial_integration//gaussian_expansion.f90',
                  ],
                extra_f90_compile_args = COMPILER_FLAGS,
                extra_f77_compile_args = COMPILER_FLAGS,
                extra_compile_args = COMPILER_FLAGS ,
                extra_link_args = LINKER_FLAGS,
                language = FORTRAN,
                # f2py_options=['--quiet']
                )



def setup_pepytools():

    setup(

        name="ml_tools",
        packages=['ml_tools','ml_tools.compressor','ml_tools.descriptors',
                    'ml_tools.descriptors.radial_numerial_integration',
                    'ml_tools.kernels','ml_tools.math_utils','ml_tools.model_selection',
                    'ml_tools.models','ml_tools.split','ml_tools.utils',
                    'ml_tools.hasher',
                    ],

        # metadata
        version=__version__,
        author=__author__,
        author_email=__email__,
        platforms = 'Any',
        description = __description__,
        keywords = [],
        classifiers = [],
        url = __url__,

        # set up package contents
        ext_modules = [
              ext_,
        ],
)

if __name__ == '__main__':

    setup_pepytools()