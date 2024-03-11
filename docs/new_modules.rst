Adding Analysis Module 
======================

This guide explains how to add new nuclei or spot detection modules to the ``ImageAnalyzer`` class for biological image analysis.


Creating a Pull Request
-----------------------

Before adding your new module, ensure your changes are ready to be shared with the HiTIPS repository:

1. Fork the repository on GitHub.
2. Clone your fork locally and create a new branch for your feature.
3. Make your changes locally, committing them to your branch.

   .. code-block:: bash

       git add .
       git commit -m "Add new nuclei detection module"

4. Push your changes to your fork on GitHub.

   .. code-block:: bash

       git push origin feature_branch_name

5. Go to your fork on GitHub and click the ‘New pull request’ button.
6. Ensure the base repository is set to CBIIT/HiTIPS and the base branch is the one you want your changes pulled into.
7. Review your changes, then create the pull request.

Merging the Pull Request
------------------------

Once your pull request has been reviewed and approved:

1. Merge the pull request via the GitHub interface.
2. Fetch the updated main branch to your local repository.

   .. code-block:: bash

       git checkout main
       git pull origin main

3. Delete your local feature branch if desired.


Define the New Nuclei Detection Module
--------------------------------------

Define a new module within the ``ImageAnalyzer`` class for nuclei detection. For example, ``custom_nuclei_segmenter``.

.. code-block:: python

    def custom_nuclei_segmenter(self, input_img, **kwargs):
        """
        Custom module to segment nuclei in an image.

        Parameters:
            input_img (numpy.ndarray): Input image for nuclei segmentation.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: boundary (numpy.ndarray), mask (numpy.ndarray).
        """
        # Your implementation
        return boundary, mask

Integrate the New Nuclei Module
-------------------------------

Update the ``neuceli_segmenter`` module to include your new detection module. Add an `if` condition that checks for your module's name and calls your custom function.

.. code-block:: python

    def neuceli_segmenter(self, input_img, pixpermic=None):
        # Existing code
        # ...

        if self.gui_params.NucDetectMethod_currentText == "YourModuleName":
            boundary, mask = self.custom_nuclei_segmenter(input_img, **kwargs)
        
        return boundary, mask

Integrate the New Nuclei Detection Module into the GUI
------------------------------------------------------

To make the new nuclei detection module available in the GUI, you need to update the ``NucDetectMethod`` QComboBox within the ``analyzer`` class. Follow these steps:

1. Add the new module to the Nuclei Detection Module ComboBox

   Locate the ``NucDetectMethod`` QComboBox initialization in the ``analyzer`` class. Add a new item to the ComboBox that corresponds to your new nuclei detection module. 

   .. code-block:: python

       self.NucDetectMethod.addItem("YourMethodName")

   Replace ``"YourMethodName"`` with the name of your new nuclei detection module. This name will be displayed in the GUI and should be descriptive and user-friendly.

2. Update the GUI logic to handle the new module

   In the module where you handle the selection changes of the ``NucDetectMethod`` QComboBox (usually connected to a signal like ``currentIndexChanged``), add a conditional block to set the parameters or call the function associated with your new module.

   .. code-block:: python

       def INITIALIZE_SEGMENTATION_PARAMETERS(self):
           if self.NucDetectMethod.currentText() == "YourMethodName":
               # Set the parameters or call your custom segmentation module
               pass

   Ensure you replace ``"YourMethodName"`` with the exact string you used in the ComboBox item. This block can be used to initialize specific parameters or trigger your custom module for nuclei detection.

3. Ensure that your module is selectable and triggers the correct functionality in the GUI. Test the GUI to ensure that when your module is selected, the appropriate segmentation parameters are set or adjusted, and the module executes correctly when invoked.

By following these steps, you integrate your new nuclei detection module into the HiTIPS application, allowing users to select and use it directly from the graphical interface.


Define the New Spot Detection Module
------------------------------------

Define a new module within the ``ImageAnalyzer`` class for spot detection. For example, ``custom_spot_detector``.

.. code-block:: python

    def custom_spot_detector(self, input_img, **kwargs):
        """
        Custom module to detect spots in an image.

        Parameters:
            input_img (numpy.ndarray): Input image for spot detection.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: final_spots, binary image with detected spots.
        """
        # Your implementation
        return final_spots

Integrate the New Spot Detection Module
---------------------------------------

Update the ``SpotDetector`` module to include your new spot detection module. Add code at the beginning to handle your module.

.. code-block:: python

    def SpotDetector(self, **kwargs):
        spot_detection_method = kwargs.get('spot_detection_method', "DefaultMethod")
        
        if spot_detection_method == "YourSpotDetectionMethod":
            final_spots = self.custom_spot_detector(input_img, **kwargs)
            spots_df, bin_img_g, labeled_spots = self.spots_information(final_spots, input_image_raw, **kwargs)
        
        # Rest of the existing code

Integrate the New Spot Detection Module into the GUI
-----------------------------------------------------

To incorporate the new spot detection module into the HiTIPS application's GUI, you need to update the interface elements related to spot detection. This involves adding the new module to a QComboBox and adjusting the GUI's logic to utilize the new module when selected.

1. Update the Spot Detection Module ComboBox

   Find the QComboBox that lists the spot detection modules. This could be a part of the spot detection settings in the GUI. Add an entry for your new spot detection module:

   .. code-block:: python

       self.SpotDetectMethod.addItem("YourSpotDetectionMethod")

   Replace ``"YourSpotDetectionMethod"`` with the name you’ve chosen for your new spot detection module. The name should be clear and descriptive, as it will be visible in the GUI for users to select.

2. Modify the GUI Logic to Include the New Module

   In the part of your GUI code where the selection of the spot detection module is handled (typically connected to a signal like ``currentIndexChanged`` of the QComboBox), add a condition to check for your new module and set the appropriate parameters or call the related function:

   .. code-block:: python

       def UPDATE_SPOT_DETECTION_PARAMETERS(self):
           if self.SpotDetectMethod.currentText() == "YourSpotDetectionMethod":
               # Initialize parameters or invoke your custom spot detection
               pass

   Ensure that ``"YourSpotDetectionMethod"`` matches the string used in the ComboBox. This section of code will be responsible for configuring any specific settings or initiating your custom module when the user selects it from the GUI.

3. Test the Integration

   After integrating the new module into the GUI, thoroughly test the functionality to ensure that selecting the new module updates the GUI as expected and that the spot detection process works correctly with the chosen settings. This may involve checking parameter adjustments, ensuring the module is triggered properly, and verifying the output is as expected.

By incorporating these steps into the HiTIPS application, users will be able to select and utilize the new spot detection module directly from the graphical interface, enhancing the tool's flexibility and functionality.



Test Your Modules
-----------------

Test the new modules with various images to ensure accuracy and robustness.

Update the Documentation
------------------------

Document your modules in the project's documentation, detailing their overview, usage, and any specific requirements.

Commit Your Changes
-------------------

Commit the changes to the project repository, ensuring all new code is properly documented and tested.

Conclusion
----------

Adding new detection modules to the ``ImageAnalyzer`` class expands its capabilities for biological image analysis. Adhere to best practices in coding, documentation, and testing for successful integration.
