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



