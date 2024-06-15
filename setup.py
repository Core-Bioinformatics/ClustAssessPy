from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

__version__ = "1.0.1"

# Path to the C++ extension source files
cpp_source_dir = "cpp_functions"
eigen_include_dir = os.path.join(cpp_source_dir, "include")

ext_modules = [
    Pybind11Extension(
        "snn_functions",
        [os.path.join(cpp_source_dir, "snn_cpp_functions.cpp")],
        include_dirs=[
            eigen_include_dir
        ],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=['-std=c++11'],
    ),
]

setup(
    name='ClustAssessPy',
    version=__version__,
    packages=find_packages(),
    description='Python package for systematic assessment of clustering results stability on single-cell data.',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'pandas',
        'scanpy',
        'umap-learn',
        'seaborn',
        'matplotlib',
        'scipy',
        'networkx',
        'plotnine',
        'pynndescent',
        'leidenalg',
        'louvain',
        "igraph"
    ],
    license='MIT',
    author="Rafael Kollyfas",
    author_email="rk720@cam.ac.uk",
    python_requires='>=3.7',
    keywords=['clustering', 'stability', 'assessment', 'machine learning', 'graph', 'network', 'community', 'detection'],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
