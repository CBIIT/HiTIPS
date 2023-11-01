from setuptools import setup, find_packages

setup(
    name="hitips",
    version="0.1",
    packages=find_packages(),
    install_requires=[
		    'numpy',
		    'opencv-python-headless',
		    'scikit-image',
		    'scipy',
		    'Pillow',
		    'pandas',
		    'matplotlib',
		    'btrack',  # Ensure this exists on PyPI
		    'imageio',
		    'tifffile',
		    'aicsimageio',
		    'deepcell',  # Ensure this exists on PyPI
		    'scikit-learn',
		    'hmmlearn',
		    'PyQt5',
		    'cellpose',
		    'tensorflow',
		    'joblib',
		    'dask',
		    'nd2reader',
		    'imaris_ims_file_reader',
		    'qimage2ndarray'
		],

    entry_points={
        'console_scripts': [
            'hitips=hitips.HiTIPS:main',
        ],
    },
)

