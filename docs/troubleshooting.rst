Troubleshooting
===============


Reporting Issues
----------------

If you encounter any issues or errors while using HiTIPS, we encourage you to report them on our GitHub repository's issue tracker. This helps us to improve the software and assist you with troubleshooting. Please provide as much detail as possible, including steps to reproduce the problem, error messages, and screenshots if relevant.

Report issues here: https://github.com/CBIIT/HiTIPS/issues


Installation troubleshooting
----------------------------

After installaion is a finished, if the HiTIPS GUI doesn't run without recieving any errors, 
first uninstall opencv-python::
     
    pip uninstall opencv-python

Then try installing the headless version of the package (this version is not bundled with PyQT and doesn't interefere with hiTIPS GUI)::

    pip install opencv-python-headless


General Usage Tips
------------------

- Always check the "Display" checkbox after loading the metadata or images and before adjusting analysis parameters or loading analysis configuration file.

- If you are running HiTIPs in parallel mode, under "Available Resources" tab, select the number of CPU cores you would like to use for your analysis. If you are only using CPU methods, the number of cores you can use will be determined by the number of cores that are avaible to you and the system memory. However, if you are using any GPU method (such as CellPose) in your analysis pipeline, we recommend using maximum 5 cores. The GPU selection is depricated and will be removed in new versions.

- If you are running HiTIPS on live cell data and need to track cells and spots, maks sure you also check "Nuclei Mask", "Nuclei Info", and "Spots Location" checkboxes under "Results" tab.


Resolve Installation Issues for deepcell-toolbox On Windows
===========================================================

The error encountered indicates a requirement for Microsoft Visual C++ 14.0 or greater due to the need to compile C extensions in the Python package. Follow these steps to resolve the issue:

1. Install Microsoft C++ Build Tools
------------------------------------

Visit the `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ website and download the installer. Ensure to select C++ build tools during installation, including the Windows 10 SDK and the latest MSVC v142 x64/x86 build tools.

2. Run the Installer
--------------------

Execute the downloaded installer and follow the prompts. Make sure the necessary components for C++ development are selected, especially the MSVC build tools and the Windows SDK.

3. Restart Your Computer
------------------------

After installing the C++ Build Tools, restart your computer to ensure the new configurations are applied and the necessary paths are set.

4. Re-attempt Installation
--------------------------

Once your system is equipped with the necessary C++ tools, retry installing the `deepcell-toolbox` package. The package should now compile successfully.

5. Check for Errors
-------------------

Monitor the installation process for any errors. If encountered, review the error messages for specifics that might indicate missing dependencies or additional required configurations.

6. Environment Setup
--------------------

Verify that your Python environment is correctly set up and that the version of Python used is compatible with the `deepcell-toolbox` package. The package may not support older Python versions.

Ensure these steps are followed accurately to resolve the installation issues related to the `deepcell-toolbox` package.

