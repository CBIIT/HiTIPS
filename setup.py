from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="hitips",
    version="1.1.0",
    author="keikhosravi",
    author_email="adib.keikhosravi@nih.gov",
    description="HiTIPS: High-Throughput Image Processing Software for FISH data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CBIIT/HiTIPS",
    packages=find_packages(),
    package_data={ 'hitips': ['cell_config.json', 'Roboto-Light.ttf', 'Roboto-Bold.ttf']},
    install_requires=[
        'numpy==1.23.5',
        'scikit-image==0.18.3',
        'scipy==1.11.3',
        'Pillow==9.0.1',  
        'pandas==1.4.2',
        'matplotlib==3.5.1',
        'btrack==0.4.3',  
        'imageio==2.31.5',
        'tifffile==2023.9.26',
        'aicsimageio==4.7.0',
        'scikit-learn==1.1.1',
        'hmmlearn==0.3.0',
        'PyQt5==5.15.10',
        'cellpose==2.0.5',
        'tensorflow==2.8.4',
        'dask==2022.2.1',
        'nd2reader==3.3.0',
        'imaris_ims_file_reader==0.1.8',
        'qimage2ndarray==1.9.0',
        'spatial_efd==1.2.1',
        'pydantic==1.10.9',
        'aicspylibczi==3.1.2',
        'opencv-python-headless==4.9.0.80'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'hitips=hitips.HiTIPS:main',
        ],
    },
    license="MIT",
    keywords="high-throughput imaging FISH analysis cell segmentation signal quantification",
)

