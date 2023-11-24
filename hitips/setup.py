from setuptools import setup

setup(
    name='HiTIPS',
    version='1.0.0',
    author='Adib Keikhosravi',
    author_email='adib.k.bme@gmail.com',
    description='High throughput image processing software for anlyzing cell dynamics and DNA/RNA ',
    packages=['your_package'],
    install_requires=[
        'pyqt',
        'scipy',
        'pandas',
        'pillow',
        'matplotlib',
        'imageio',
        'tifffile',
        'scikit-image==0.18.3',
        'btrack',
        'qimage2ndarray',
        'aicsimageio',
        'cellpose',
        'opencv-python-headless',
        'deepcell',
        'hmmlearn',
        'aicsimageio[nd2]',
        'nd2reader'
    ],
    entry_points={
        'console_scripts': [
            'HiTIPS=hitips.HiTIPS:main',
        ],
    },
)
