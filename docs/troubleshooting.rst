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



Resolve Installation Issue for deepcell-toolbox
===============================================

The error encountered indicates a requirement for Microsoft Visual C++ 14.0 or greater for the installation of the ``deepcell-toolbox`` package. Follow these steps to resolve the issue:

1. Install Microsoft C++ Build Tools
------------------------------------

   Download the Microsoft C++ Build Tools from the provided link:

   - `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_

   Ensure to include the C++ build tools during the installation, along with the Windows 10 SDK and the latest MSVC v142 x64/x86 build tools.

2. Run the Installer
--------------------

   Execute the installer and follow the on-screen instructions to install the necessary build tools. The correct installation of these tools is essential for compiling C extensions in Python packages.

3. Restart Your Computer
------------------------

   After installing the C++ Build Tools, restart your computer to ensure that the changes are applied and the necessary paths are set in the environment.

4. Re-attempt Installation
--------------------------

   With the build tools installed, try to reinstall the ``deepcell-toolbox`` package. The installation should now be able to compile the required C extensions successfully.

5. Check for Errors
-------------------

   If the installation fails again, review the error messages for any indications of the problem. The error output can provide valuable insights into what went wrong and how to fix it.

6. Environment Setup
--------------------

   Make sure your Python environment is correctly set up, and you are using a compatible version of Python for the ``deepcell-toolbox`` installation. The package may not be compatible with older Python versions.