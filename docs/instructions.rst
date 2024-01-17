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
   


HiTIPS user interface
---------------------
HiTIPS unser interface consists of two windows: one for displaying the image and realtime results of cell and spots segmentation, and the other one for image and dataset loading, image selection for display and selecting analsysis parameters as well as running the analysis. 

.. image:: images/user_interface.png
   :alt: User Interface
   :align: center


Data loading from CellVoyager
-----------------------------
- From “Input Output” tab click on “Load MetaData” button.

- Using the “Select Metadata Files…” window navigate into the folder containing all the images and metadata saved by the instrument.

- You should only see the file “MeasurementData.mlf”. Select this file and click ”Open”.

.. image:: images/metadata_loading.png
   :alt: Metadata Loading
   :align: center
   
Loading plate information from CellVoyager
------------------------------------------
- Once the plate information is loaded in HiTIPS you should see the “pixel size” value under device label and the ”Display” checkbox is active.

.. image:: images/display_checkbox.png
   :alt: Display Checkbox
   :align: center

- Now check the “Display” checkbox to see all the wells containing images with their corresponding fields (FOV), Z planes and time points (Time).

  -- The color of wells containing image data on the well-plate layout will change to green now.
  
  -- To view other images you can click on the well, field, z-plane or timepoint of interest.
  
  -- This should also load the first image of the dataset into the display window.

.. image:: images/image_selection.png
   :alt: Image Selection
   :align: center
   
Loading Bioformat image files
-----------------------------

- From “Input Output” tab click on “Load Images” button 

.. image:: images/load_bioformat.png
   :alt: Load Bioformat
   :align: center
   
- Using the “Select Image Files…” window navigate into the folder containing all the images (ome.tiff, czi, nd2, ims, etc.) and select all the images required to be analyzed.

.. image:: images/select_bioformat.png
   :alt: Select Bioformat
   :align: center
   
   
Image Display window
--------------------

- After loading plate information and checking display checkbox all the active channels in the dataset will be active and the channel information will be displayed next to the channel name.

- You can also check the “Max.Z” checkbox for each channel to display the maximum projection of the Z-stack.

.. image:: images/display_window.png
   :alt: Display Window
   :align: center   
   
- You can select the color of the channel by right-clicking on text next to its checkbox and selecting the color from the dropdown menu.


Adjusting channel intensities
-----------------------------

- Up to five channels can be displayed in displayed and the display intensity can be separately adjusted for each channel separately.

.. image:: images/image_channels.png
   :alt: Image Channels
   :align: center 
   
- You can adjust the intensity of each channel by selecting the specific channel from the combobox under the image and using the right slider to set the maximum intensity and left slider to set the minimum intensity of the image.

  -- Note: adjusting the image intensity on the display window will not change the input image intensities for processing algorithms (nuclei segmentation, spot detection, etc.). These algorithms read and process the raw data.

.. image:: images/intensity_adjustment.png
   :alt: Intensity Adjustment
   :align: center 
   
   
Visualizing nuclei segmentation results
---------------------------------------
- To visualize the nuclei segmentation results, check the “Nuclei” checkbox on the right side of the display window. 

- From the combobox under this checkbox you can select how you would like to visualize the segmentation results. You can select nuclei boundary, area or the nuclei index. The examples are shown below. 

.. image:: images/nuclei_segmentation.png
   :alt: Nuclei Segmentation
   :align: center 
   
   
Visualizing spot detection results
----------------------------------

- To visualize the spot detection results, check the “Spots” checkbox on the right side of the display window. 

- From the combobox under this checkbox you can select how you would like to visualize the spot detection results results. You can select circles around spots or spot boundary. 

.. image:: images/spot_image.png
   :alt: Spot Image
   :align: center 
   

