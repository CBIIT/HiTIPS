Adding Analysis Method 
======================

This guide explains how to add new nuclei or spot detection methods to the ``ImageAnalyzer`` class for biological image analysis.


Creating a Pull Request
-----------------------

Before adding your new method, ensure your changes are ready to be shared with the HiTIPS repository:

1. Fork the repository on GitHub.
2. Clone your fork locally and create a new branch for your feature.
3. Make your changes locally, committing them to your branch.

   .. code-block:: bash

       git add .
       git commit -m "Add new nuclei detection method"

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


Define the New Nuclei Detection Method
--------------------------------------

Define a new method within the ``ImageAnalyzer`` class for nuclei detection. For example, ``custom_nuclei_segmenter``.

.. code-block:: python

    def custom_nuclei_segmenter(self, input_img, **kwargs):
        """
        Custom method to segment nuclei in an image.

        Parameters:
            input_img (numpy.ndarray): Input image for nuclei segmentation.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: boundary (numpy.ndarray), mask (numpy.ndarray).
        """
        # Your implementation
        return boundary, mask

Integrate the New Nuclei Method
-------------------------------

Update the ``neuceli_segmenter`` method to include your new detection method. Add an `if` condition that checks for your method's name and calls your custom function.

.. code-block:: python

    def neuceli_segmenter(self, input_img, pixpermic=None):
        # Existing code
        # ...

        if self.gui_params.NucDetectMethod_currentText == "YourMethodName":
            boundary, mask = self.custom_nuclei_segmenter(input_img, **kwargs)
        
        return boundary, mask

Integrate the New Nuclei Detection Method into the GUI
------------------------------------------------------

To make the new nuclei detection method available in the GUI, you need to update the ``NucDetectMethod`` QComboBox within the ``analyzer`` class. Follow these steps:

1. Add the new method to the Nuclei Detection Method ComboBox

   Locate the ``NucDetectMethod`` QComboBox initialization in the ``analyzer`` class. Add a new item to the ComboBox that corresponds to your new nuclei detection method. 

   .. code-block:: python

       self.NucDetectMethod.addItem("YourMethodName")

   Replace ``"YourMethodName"`` with the name of your new nuclei detection method. This name will be displayed in the GUI and should be descriptive and user-friendly.

2. Update the GUI logic to handle the new method

   In the method where you handle the selection changes of the ``NucDetectMethod`` QComboBox (usually connected to a signal like ``currentIndexChanged``), add a conditional block to set the parameters or call the function associated with your new method.

   .. code-block:: python

       def INITIALIZE_SEGMENTATION_PARAMETERS(self):
           if self.NucDetectMethod.currentText() == "YourMethodName":
               # Set the parameters or call your custom segmentation method
               pass

   Ensure you replace ``"YourMethodName"`` with the exact string you used in the ComboBox item. This block can be used to initialize specific parameters or trigger your custom method for nuclei detection.

3. Ensure that your method is selectable and triggers the correct functionality in the GUI. Test the GUI to ensure that when your method is selected, the appropriate segmentation parameters are set or adjusted, and the method executes correctly when invoked.

By following these steps, you integrate your new nuclei detection method into the HiTIPS application, allowing users to select and use it directly from the graphical interface.


Define the New Spot Detection Method
------------------------------------

Define a new method within the ``ImageAnalyzer`` class for spot detection. For example, ``custom_spot_detector``.

.. code-block:: python

    def custom_spot_detector(self, input_img, **kwargs):
        """
        Custom method to detect spots in an image.

        Parameters:
            input_img (numpy.ndarray): Input image for spot detection.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: final_spots, binary image with detected spots.
        """
        # Your implementation
        return final_spots

Integrate the New Spot Detection Method
---------------------------------------

Update the ``SpotDetector`` method to include your new spot detection method. Add code at the beginning to handle your method.

.. code-block:: python

    def SpotDetector(self, **kwargs):
        spot_detection_method = kwargs.get('spot_detection_method', "DefaultMethod")
        
        if spot_detection_method == "YourSpotDetectionMethod":
            final_spots = self.custom_spot_detector(input_img, **kwargs)
            spots_df, bin_img_g, labeled_spots = self.spots_information(final_spots, input_image_raw, **kwargs)
        
        # Rest of the existing code

Integrate the New Spot Detection Method into the GUI
-----------------------------------------------------

To incorporate the new spot detection method into the HiTIPS application's GUI, you need to update the interface elements related to spot detection. This involves adding the new method to a QComboBox and adjusting the GUI's logic to utilize the new method when selected.

1. Update the Spot Detection Method ComboBox

   Find the QComboBox that lists the spot detection methods. This could be a part of the spot detection settings in the GUI. Add an entry for your new spot detection method:

   .. code-block:: python

       self.SpotDetectMethod.addItem("YourSpotDetectionMethod")

   Replace ``"YourSpotDetectionMethod"`` with the name you’ve chosen for your new spot detection method. The name should be clear and descriptive, as it will be visible in the GUI for users to select.

2. Modify the GUI Logic to Include the New Method

   In the part of your GUI code where the selection of the spot detection method is handled (typically connected to a signal like ``currentIndexChanged`` of the QComboBox), add a condition to check for your new method and set the appropriate parameters or call the related function:

   .. code-block:: python

       def UPDATE_SPOT_DETECTION_PARAMETERS(self):
           if self.SpotDetectMethod.currentText() == "YourSpotDetectionMethod":
               # Initialize parameters or invoke your custom spot detection
               pass

   Ensure that ``"YourSpotDetectionMethod"`` matches the string used in the ComboBox. This section of code will be responsible for configuring any specific settings or initiating your custom method when the user selects it from the GUI.

3. Test the Integration

   After integrating the new method into the GUI, thoroughly test the functionality to ensure that selecting the new method updates the GUI as expected and that the spot detection process works correctly with the chosen settings. This may involve checking parameter adjustments, ensuring the method is triggered properly, and verifying the output is as expected.

By incorporating these steps into the HiTIPS application, users will be able to select and utilize the new spot detection method directly from the graphical interface, enhancing the tool's flexibility and functionality.



Test Your Methods
-----------------

Test the new methods with various images to ensure accuracy and robustness.

Update the Documentation
------------------------

Document your methods in the project's documentation, detailing their overview, usage, and any specific requirements.

Commit Your Changes
-------------------

Commit the changes to the project repository, ensuring all new code is properly documented and tested.

Conclusion
----------

Adding new detection methods to the ``ImageAnalyzer`` class expands its capabilities for biological image analysis. Adhere to best practices in coding, documentation, and testing for successful integration.
