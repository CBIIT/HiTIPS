Adding Analysis Method 
======================

This guide explains how to add new nuclei or spot detection methods to the ``ImageAnalyzer`` class for biological image analysis.

1. Define the New Nuclei Detection Method
-----------------------------------------

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

2. Integrate the New Nuclei Method
----------------------------------

Update the ``neuceli_segmenter`` method to include your new detection method. Add an `if` condition that checks for your method's name and calls your custom function.

.. code-block:: python

    def neuceli_segmenter(self, input_img, pixpermic=None):
        # Existing code
        # ...

        if self.gui_params.NucDetectMethod_currentText == "YourMethodName":
            boundary, mask = self.custom_nuclei_segmenter(input_img, **kwargs)
        
        return boundary, mask

3. Define the New Spot Detection Method
---------------------------------------

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

4. Integrate the New Spot Detection Method
------------------------------------------

Update the ``SpotDetector`` method to include your new spot detection method. Add code at the beginning to handle your method.

.. code-block:: python

    def SpotDetector(self, **kwargs):
        spot_detection_method = kwargs.get('spot_detection_method', "DefaultMethod")
        
        if spot_detection_method == "YourSpotDetectionMethod":
            final_spots = self.custom_spot_detector(input_img, **kwargs)
            spots_df, bin_img_g, labeled_spots = self.spots_information(final_spots, input_image_raw, **kwargs)
        
        # Rest of the existing code

5. Implement Your Algorithms
----------------------------

Implement the segmentation or detection algorithms within the respective methods. Use libraries like OpenCV or TensorFlow as needed.

6. Update Class and Method Documentation
----------------------------------------

Update the class and method docstrings to include descriptions of your new methods, their parameters, and usage examples.

7. Test Your Methods
--------------------

Test the new methods with various images to ensure accuracy and robustness.

8. Update the Documentation
---------------------------

Document your methods in the project's documentation, detailing their overview, usage, and any specific requirements.

9. Commit Your Changes
----------------------

Commit the changes to the project repository, ensuring all new code is properly documented and tested.

Conclusion
----------

Adding new detection methods to the ``ImageAnalyzer`` class expands its capabilities for biological image analysis. Adhere to best practices in coding, documentation, and testing for successful integration.
