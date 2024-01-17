HiTIPS
======

**HiTIPS** (High-Throughput Image Processing Software) is a comprehensive tool crafted for the analysis of high-throughput imaging datasets. Specifically designed for FISH (Fluorescence In Situ Hybridization) data, HiTIPS incorporates cutting-edge image processing and machine learning algorithms, delivering automated solutions for cell and nucleus segmentation, FISH signal identification, and quantification of signal attributes.

Key Features
------------

- **Automated Segmentation**: Efficiently segments cells and nuclei.
- **FISH Signal Identification**: Accurate localization and identification of FISH signals.
- **Quantitative Analysis**: Measures signal intensity and distribution.
- **Customizable Interface**: Provides flexibility for customization and integrating plugins.
- **High-Throughput Support**: Designed for processing large-scale imaging datasets.
- **Extendable Algorithms**: Incorporates new methodologies for enhancing current analysis routines.
- **Plugin Support**: Supports the creation and integration of new analysis routines.
  
Hardware and Software Prerequisites
-----------------------------------

Hardware Requirements
^^^^^^^^^^^^^^^^^^^^^

- **CPU**: Multi-core processor (e.g., Intel i7 or AMD Ryzen 7).
- **RAM**: Minimum 16GB (32GB recommended for large datasets).
- **Storage**: SSD with 500GB or more of available space.
- **GPU**: Optional but recommended, especially if using CUDA-enhanced functionalities.

Software Requirements
^^^^^^^^^^^^^^^^^^^^^

- **Operating System**: 64-bit Linux distribution (e.g., Ubuntu, CentOS, Fedora).
- **Python**: Version 3.7 or newer.
- **Package Manager**: Latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/products/distribution>`_.

Installation
------------

Using Conda and Pip
^^^^^^^^^^^^^^^^^^^

1. **Create a Conda Environment**::

    conda create --name hitips_env python=3.8
    conda activate hitips_env
   
2. **Install HiTIPS**::

    pip install hitips
   
3. **Launch HiTIPS**::

    hitips
   
Usage
-----

- Launch HiTIPS using the command hitips.
- Introduce your high-throughput imaging dataset into the software.
- Navigate through the available analysis options and specify your desired tasks.
- Modify the analysis parameters fitting your requirements.
- Initiate the analysis process.
- Review and interpret the produced outcomes.
- Save or export the results as required.


HiTIPS user interface
---------------------
HiTIPS unser interface consists of two windows: one for displaying the image and realtime results of cell and spots segmentation, and the other one for image and dataset loading, image selection for display and selecting analsysis parameters as well as running the analysis. 

.. image:: images/user_interface.png
   :alt: User Interface
   :align: center

