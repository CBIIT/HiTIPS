import numpy as np
import cv2
from scipy.ndimage import label
from scipy import ndimage
from PIL import Image
from skimage.draw import circle_perimeter
import math
from skimage.filters import median, gaussian, sobel
from skimage.morphology import disk, binary_closing, skeletonize, binary_opening, binary_erosion
from skimage.filters import threshold_li
from skimage.segmentation import watershed, find_boundaries
from skimage.measure import regionprops, regionprops_table
import pandas as pd
from cellpose import models, utils
import tensorflow as tf
from tensorflow.keras import backend as K
from skimage.transform import rescale, resize
from scipy.special import erf


class ImageAnalyzer(object):
    
    """
    A class for analyzing and processing biological images, focusing on segmentation and tracking.
    
    Attributes
    ----------
    gui_params (object): Configuration parameters from the GUI or other settings.
    
    Methods
    -------
    __init__(self, gui_params):
        Initializes the ImageAnalyzer with configuration parameters.

    neuceli_segmenter(self, input_img, pixpermic=None):
        Segments nuclei in an image using various methods.

    deepcell_segmenter(self, input_img, mmp=None):
        Segments nuclei using DeepCell's NuclearSegmentation application.

    cellpose_segmenter(self, input_img, use_GPU, cell_dia=None):
        Segments cells using the Cellpose model.

    segmenter_function(self, input_img, cell_size=None, first_threshold=None, second_threshold=None):
        Advanced segmentation using blurring, thresholding, and watershed.

    watershed_scikit(self, input_img, cell_size=None, first_threshold=None, second_threshold=None):
        Image segmentation using scikit-image's watershed algorithm.

    max_z_project(self, image_stack):
        Maximum intensity projection from an image stack.

    SpotDetector(self, **kwargs):
        Detects spots in an image using various methods.

    spots_information(self, bin_img_log, max_project, gaussian_fit=False, min_area=0, max_area=100, min_integrated_intensity=0, psf_size=1.6):
        Analyzes and refines detected spots.

    gmask_fit(self, pic, xy_input=None, fit=False, psf_size=1.6):
        Gaussian mask fitting for spot characteristics.

    local_background(self, pic, display=False):
        Calculates local background using a linear fit to border pixels.

    COORDINATES_TO_CIRCLE(self, coordinates, ImageForSpots, circ_radius=5):
        Draws circles at specified coordinates on an image.

    SPOTS_TO_BOUNDARY(self, final_spots):
        Converts a binary image of spots to highlighted boundaries.
    """
    def __init__(self,params_dict):
        """
        Initializes the ImageAnalyzer with configuration parameters.

        Parameters:
            gui_params (object): Configuration parameters for image analysis.
        """
        self.params_dict = params_dict
    
    def neuceli_segmenter(self, input_img, pixpermic=None):
        """
        Segments nuclei in an image using various methods based on GUI settings.

        Parameters:
            input_img (numpy.ndarray): Input image for nuclei segmentation.
            pixpermic (float, optional): Microns per pixel for physical size in the image. Default is None.

        Returns:
            tuple: A tuple containing:
                - boundary (numpy.ndarray): Image with marked nuclei boundaries.
                - mask (numpy.ndarray): Binary mask of segmented nuclei.

        Usage:
            boundary, mask = image_analyzer.neuceli_segmenter(input_image, pixpermic=0.5)
        """
        
        
        if self.params_dict['NucDetectMethod_currentText'] == "Int.-based":
            
            first_tresh = self.params_dict['NucSeparationSlider_value']*2.55
            second_thresh = 255-(self.params_dict['NucDetectionSlider_value']*2.55)
            Cell_Size = self.params_dict['NucleiAreaSlider_value']
            
            boundary, mask = self.segmenter_function(input_img, cell_size=Cell_Size, first_threshold=first_tresh, second_threshold=second_thresh)
            
        if self.params_dict['NucDetectMethod_currentText'] == "Marker Controlled":
            
            Cell_Size = self.params_dict['NucleiAreaSlider_value']
            max_range = np.sqrt(Cell_Size/3.14)*2/float(pixpermic)
            nuc_detect_sldr = self.params_dict['NucDetectionSlider_value']
            first_tresh = np.ceil((1-(nuc_detect_sldr/100))*max_range).astype(int)
            
            second_thresh = self.params_dict['NucSeparationSlider_value']
            
            boundary, mask = self.watershed_scikit(input_img, cell_size=Cell_Size, first_threshold=first_tresh, second_threshold=second_thresh)

        if self.params_dict['NucDetectMethod_currentText'] == "CellPose-GPU":

            Cell_Size = self.params_dict['NucleiAreaSlider_value']
            cell_diameter = np.sqrt(Cell_Size/(float(pixpermic)*float(pixpermic)))*2/3.14
            
            boundary, mask = self.cellpose_segmenter(input_img, use_GPU=1, cell_dia=cell_diameter)
            
        if self.params_dict['NucDetectMethod_currentText'] == "CellPose-CPU":
            
            Cell_Size = self.params_dict['NucleiAreaSlider_value']
            cell_diameter = np.sqrt(Cell_Size/(float(pixpermic)*float(pixpermic)))*2/3.14
            
            boundary, mask = self.cellpose_segmenter(input_img, use_GPU=0, cell_dia=cell_diameter)
            
        if self.params_dict['NucDetectMethod_currentText'] == "CellPose-Cyto":
            
            Cell_Size = self.params_dict['NucleiAreaSlider_value']
            cell_diameter = np.sqrt(Cell_Size/(float(pixpermic)*float(pixpermic)))*2/3.14
            
            boundary, mask = self.cellpose_segmenter(input_img, use_GPU=1, cell_dia=cell_diameter)
                
        if self.params_dict['NucDetectMethod_currentText'] == "DeepCell":
                        
            boundary, mask = self.deepcell_segmenter(input_img, mmp=float(pixpermic))
        
        
        return boundary, mask   


    def deepcell_segmenter(self, input_img, mmp=None):
        """
        Performs nuclear segmentation on an input image using DeepCell's NuclearSegmentation application.

        Parameters:
            - input_img (numpy.ndarray): The input image for segmentation, expected to be a 2D array representing a grayscale image.
            - mmp (float, optional): Microns per pixel value for the input image. Adjusts the model's internal scaling for accurate size representation. Default is None.

        Returns:
            - boundary (numpy.ndarray): Image with boundaries of nuclei marked as white lines on a black background.
            - mask (numpy.ndarray): Binary mask image with filled areas representing segmented nuclei.

        The function initializes the NuclearSegmentation model, processes the input image, and uses the model to predict nuclear masks.
        It then finds and processes boundaries to create clear nuclear boundary and mask images.

        Usage Example:
            boundary, mask = deepcell_segmenter(input_image, mmp=0.5)

        Note:
            Ensure the deepcell library is installed and properly set up. The input image should be in grayscale. The mmp parameter is important for physical size representation in medical imaging.
        """
        from deepcell.applications import NuclearSegmentation
        app = NuclearSegmentation()
        im = np.expand_dims(input_img, axis=-1)
        im = np.expand_dims(im, axis=0)
        
        masks1 = app.predict(im, image_mpp=mmp)
        masks = np.squeeze(masks1)
        boundary = find_boundaries(masks, connectivity=1, mode='thick', background=0)
        boundary_img = (255*boundary).astype('uint8')
        resized_bound = boundary_img
        filled1 = ndimage.binary_fill_holes(resized_bound)
        mask= (255*filled1).astype('uint8')-resized_bound
        boundary= resized_bound.astype('uint8')

        return boundary, mask
        
    def cellpose_segmenter(self, input_img, use_GPU, cell_dia=None):
        """
        Performs cell segmentation on an input image using the Cellpose model.

        Parameters:
            - input_img (numpy.ndarray): The input image for segmentation.
            - use_GPU (bool): Flag to indicate whether to use GPU acceleration.
            - cell_dia (float, optional): Estimated diameter of the cells in pixels. Default is None.

        Returns:
            - boundary (numpy.ndarray): Image with boundaries of cells marked.
            - mask (numpy.ndarray): Binary mask image with filled areas representing segmented cells.

        This function uses the Cellpose model for segmentation. The user can choose between the 'cyto' and 'nuclei' model types
        based on the `NucDetectMethod_currentText` parameter from `gui_params`.
        The function applies pre-processing and post-processing steps to enhance the segmentation results, 
        including optional boundary removal and morphological operations for cleaning the mask.

        Usage Example:
            boundary, mask = cellpose_segmenter(input_image, True, cell_dia=100)

        Note:
            Requires the Cellpose and OpenCV libraries. The `use_GPU` parameter should be set according to the available hardware.
            The function adjusts the processing steps based on the `NucRemoveBoundaryCheckBox_isChecked` parameter from `gui_params`.
        """
        
        if self.params_dict['NucRemoveBoundaryCheckBox_isChecked'] == True:
            img_uint8 = input_img
        else:
            img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)

        if self.params_dict['NucDetectMethod_currentText'] == "CellPose-Cyto":
            model = models.Cellpose(gpu=use_GPU, model_type='cyto')
        else:
            model = models.Cellpose(gpu=use_GPU, model_type='nuclei')
        masks, flows, styles, diams = model.eval(img_uint8, diameter=cell_dia, flow_threshold=None)
                
        boundary = find_boundaries(masks, connectivity=1, mode='thick', background=0)

        if self.params_dict['NucRemoveBoundaryCheckBox_isChecked'] == True:

            boundary_img = (255*boundary).astype('uint8')
            filled1 = ndimage.binary_fill_holes(boundary_img)
            mask1= (255*filled1).astype('uint8')-boundary_img
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask1.astype('uint8'), cv2.MORPH_OPEN, kernel)
            
            boundary_img = find_boundaries(mask, connectivity=1, mode='thick', background=0)
            resized_bound = cv2.resize((255*boundary_img).astype('uint8'),(input_img.shape[1],input_img.shape[0]))
            filled1 = ndimage.binary_fill_holes(resized_bound)
        else:
            boundary_img = (255*boundary[3:boundary.shape[0]-3,3:boundary.shape[1]-3]).astype('uint8')
            resized_bound = cv2.resize(boundary_img,(input_img.shape[1],input_img.shape[0]))
            filled1 = ndimage.binary_fill_holes(resized_bound)
        
        mask1= (255*filled1).astype('uint8')-resized_bound
        kernel = np.ones((11,11), np.uint8)
        mask = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
        mask = 255*(mask==255).astype('uint8')
        mask[mask>0]=255
        boundary= resized_bound.astype('uint8')

        return boundary, mask

    def segmenter_function(self, input_img, cell_size=None, first_threshold=None, second_threshold=None):
        """
        Performs advanced image segmentation using a combination of blurring, thresholding, and the watershed algorithm.

        Parameters:
            - input_img (numpy.ndarray): The input image for segmentation.
            - cell_size (int, optional): Size of the kernel for median blurring. If even, it is incremented by 1. Default is None.
            - first_threshold (int, optional): Threshold value for the first round of thresholding in the watershed algorithm. Default is None.
            - second_threshold (int, optional): Threshold value for the second round of thresholding in the watershed algorithm. Default is None.

        Returns:
            - boundary (numpy.ndarray): Image showing the boundaries of segmented regions.
            - mask (numpy.ndarray): Binary mask image with segmented regions filled.

        The function applies a series of image processing steps including median and Gaussian blurring, binary thresholding, hole filling, 
        distance transformation, and watershed transformation for accurate segmentation. It returns the boundaries of segmented regions 
        and a binary mask of these regions.

        Usage Example:
            boundary, mask = segmenter_function(input_image, cell_size=5, first_threshold=50, second_threshold=150)

        Note:
            Requires OpenCV and SciPy libraries. The cell_size, first_threshold, and second_threshold parameters should be chosen based on 
            the specific requirements of the image analysis task.
        """
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        
        ## First blurring round
        if (cell_size %2)==0:
            cell_size = cell_size + 1
        median_img = cv2.medianBlur(img_uint8,7)
        gaussian_blurred = cv2.GaussianBlur(median_img,(5,5),0)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(gaussian_blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        bin_img = (1-thresh/255).astype('bool')
        ## Binary image filling
        filled = ndimage.binary_fill_holes(bin_img)
        filled_int= (filled*255).astype('uint8')
        ## Gray2RGB to feed the watershed algorithm
        img_rgb  = cv2.cvtColor(img_uint8,cv2.COLOR_GRAY2RGB)
        boundary = img_rgb
        boundary = boundary - img_rgb
        ## Distance trasform and thresholing to set the watershed markers
        dt = cv2.distanceTransform(filled.astype(np.uint8), 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        _, dt = cv2.threshold(dt, first_threshold, 255, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
        lbl = lbl.astype(np.int32)
        ## First round of Watershed transform
        cv2.watershed(img_rgb, lbl)
        ## Correcting image boundaries
        boundary[lbl == -1] = [255,255,255]
        boundary[0,:] = 0
        boundary[-1,:] = 0
        boundary[:,0] = 0
        boundary[:, -1] = 0
        b_gray = cv2.cvtColor(boundary,cv2.COLOR_BGR2GRAY)
        diff = filled_int-b_gray

        kernel = np.ones((11,11), np.uint8)
        first_pass = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

        ## Second round of marker generation and watershed 
        kernel = np.ones((5,5),np.uint8)
        aa = first_pass.astype('uint8')
        erosion = cv2.erode(aa,kernel,iterations = 1)
        kernel = np.ones((11,11), np.uint8)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        blur = cv2.GaussianBlur(opening,(11,11),50)
        ret2, thresh2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        dt = cv2.distanceTransform(255-thresh2, 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        _, dt = cv2.threshold(dt, second_threshold, 255, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
        lbl = lbl.astype(np.int32)
        cv2.watershed(img_rgb, lbl)
        ########
        boundary = img_rgb
        boundary = boundary - img_rgb

        boundary[lbl == -1] = [255,255,255]
        boundary_img = boundary[3:boundary.shape[0]-3,3:boundary.shape[1]-3]
        bound_gray = cv2.cvtColor(boundary_img,cv2.COLOR_BGR2GRAY)
        resized_bound = cv2.resize(bound_gray,(input_img.shape[1],input_img.shape[0]))

        kernel = np.ones((3,3),np.uint8)
        boundary = cv2.dilate(resized_bound,kernel,iterations = 1)
        filled1 = ndimage.binary_fill_holes(boundary)
        fin= 255*filled1-boundary
        mask = ndimage.binary_fill_holes(fin)
        mask = (255*mask).astype(np.uint8)

        return boundary, mask
            
    def watershed_scikit(self, input_img, cell_size=None, first_threshold=None, second_threshold=None):
        """
        Performs image segmentation using the watershed algorithm implemented in scikit-image.

        Parameters:
            - input_img (numpy.ndarray): The input image for segmentation.
            - cell_size (int, optional): Not used in the current implementation, but can be included for future use. Default is None.
            - first_threshold (int, optional): Threshold value for the first distance transform in watershed segmentation. Default is None.
            - second_threshold (int, optional): Threshold value for the second distance transform in watershed segmentation. Default is None.

        Returns:
            - boundary (numpy.ndarray): Image showing the boundaries of segmented regions.
            - mask (numpy.ndarray): Binary mask image with segmented regions filled.

        The function applies median filtering, thresholding, hole filling, and distance transformation, followed by the watershed algorithm for segmentation.
        It identifies and labels the regions of interest in the input image and returns the boundaries and binary mask of these regions.

        Usage Example:
            boundary, mask = watershed_scikit(input_image, first_threshold=0.1, second_threshold=0.2)

        Note:
            Requires scikit-image, OpenCV, and SciPy libraries. The first_threshold and second_threshold parameters are critical for the watershed segmentation process and should be chosen based on the specific requirements of the image analysis task.
        """
            
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        
        med_scikit = median(img_uint8, disk(1))
        thresh = threshold_li(med_scikit)
        binary = med_scikit > thresh
        filled = ndimage.binary_fill_holes(binary)
        filled_blurred = gaussian(filled, 1)
        filled_int= (filled_blurred*255).astype('uint8')
        
        thresh = threshold_li(filled_int)
        binary = filled_int > thresh
        filled = ndimage.binary_fill_holes(binary)
        filled_int = binary_opening(filled, disk(5))
        filled_int = ndimage.binary_fill_holes(filled_int)
        distance = ndimage.distance_transform_edt(filled_int)
        binary1 = distance > first_threshold
        distance1 = ndimage.distance_transform_edt(binary1)
        binary2 = distance1 > second_threshold

        labeled_spots, num_features = label(binary2)
        spot_labels = np.unique(labeled_spots)    

        spot_locations = ndimage.measurements.center_of_mass(binary2, labeled_spots, spot_labels[spot_labels>0])

        mask = np.zeros(distance.shape, dtype=bool)
        if spot_locations:
            mask[np.ceil(np.array(spot_locations)[:,0]).astype(int), np.ceil(np.array(spot_locations)[:,1]).astype(int)] = True
        markers, _ = ndimage.label(mask)
        labels = watershed(-distance, markers, mask=binary, compactness=0.5, watershed_line=True)
        boundary = find_boundaries(labels, connectivity=1, mode='thick', background=0)
        boundary_img = (255*boundary[3:boundary.shape[0]-3,3:boundary.shape[1]-3]).astype('uint8')
        resized_bound = cv2.resize(boundary_img,(input_img.shape[1],input_img.shape[0]))
        filled1 = ndimage.binary_fill_holes(resized_bound)
        
        mask= (255*filled1).astype('uint8')-resized_bound
        boundary= resized_bound.astype('uint8')
        
        return boundary, mask
    
    def max_z_project(self, image_stack):
        """
        Performs a maximum intensity projection (max-z projection) on a stack of images.

        Parameters:
            - image_stack :pandas.DataFrame
            A DataFrame where each row represents an image. 
            It must contain the columns 'ImageName' and 'Type'. 'ImageName' should be the path to the image file 
            or 'dask_array' to indicate the image is in the 'Type' column.

        Returns:
            - max_project (numpy.ndarray): A 2D array representing the maximum intensity projection 
              of the input image stack.

        The function iterates through the image stack, reads each image (either from a file or directly if it's a dask array), 
        and compiles them into a 3D numpy array. It then computes the maximum intensity projection along the z-axis (axis=2) 
        of this stack, effectively condensing the image stack into a single 2D image highlighting the most intense pixels 
        across the stack.

        Usage Example:
            max_projection_image = max_z_project(image_dataframe)

        Note:
            The input `image_stack` DataFrame must be properly formatted with 'ImageName' and 'Type' columns. 
            The function requires 'pandas' for DataFrame handling and 'PIL.Image' for image processing.
        """
        z_imglist=[]
        
        for index, row in image_stack.iterrows():
            if row['ImageName']=="dask_array":
                im = row["Type"]
            else: 
                im = Image.open(row['ImageName'])
            z_imglist.append( np.asarray(im))
        z_stack = np.stack(z_imglist, axis=2)
        max_project = z_stack.max(axis=2)
        
        return max_project
    
    def SpotDetector(self, **kwargs):
        """
        Performs spot detection in an input image using various methods such as Laplacian of Gaussian, Gaussian filtering, 
        intensity thresholding, and Enhanced LOG.

        Parameters:
            - kwargs (dict): Keyword arguments containing parameters for the spot detection process.
            The function expects the following key-value pairs:
            - 'input_image_raw' (numpy.ndarray): Raw input image for spot detection.
            - 'nuclei_image' (numpy.ndarray): Mask of nuclei, used in preprocessing.
            - 'spot_detection_method' (str): Method for spot detection ('Laplacian of Gaussian', 'Gaussian', 'Intensity Threshold', 'Enhanced LOG').
            - 'threshold_method' (str): Method for thresholding ('Auto' or 'Manual').
            - 'threshold_value' (float): Threshold value for 'Manual' method.
            - 'kernel_size' (int): Kernel size for filtering.
            - 'spot_location_coords' (str): Method to calculate spot locations ('CenterOfMass', 'MaxIntensity', 'Centroid').
            - 'remove_bright_junk' (bool): Flag to remove bright artifacts.
            - 'resize_factor' (float): Factor to resize the image.
            - 'min_area' (int): Minimum area for spots.
            - 'max_area' (int): Maximum area for spots.
            - 'min_integrated_intensity' (int): Minimum integrated intensity for spots.
            - 'psf_size' (float): Point spread function size.
            - 'gaussian_fit' (bool): Flag to perform Gaussian fitting.

        Returns:
            - spot_locations (list): List of coordinates for detected spots.
            - spots_df (pandas.DataFrame): DataFrame containing information about detected spots.

        This function applies various image processing techniques based on the specified spot detection method. 
        It handles pre-processing, spot detection, thresholding, and calculates spot locations based on the specified method.
        The function returns the locations of detected spots along with a DataFrame containing detailed information about these spots.

        Usage Example:
            spot_locations, spots_info = SpotDetector(input_image_raw=img, nuclei_image=nuclei_img, 
                                                      spot_detection_method="Laplacian of Gaussian",
                                                      threshold_method="Auto", kernel_size=3)

        Note:
            Requires cv2, numpy, scipy, and pandas libraries. The parameters should be carefully chosen based on the characteristics of the input image and the specific requirements of the spot detection task.
        """
        
        # Extract parameters from kwargs
        input_image_raw = kwargs.get('input_image_raw', None)
        nuc_mask = kwargs.get('nuc_mask', None)
        spot_detection_method = kwargs.get('spot_detection_method', "Laplacian of Gaussian")
        threshold_method = kwargs.get('threshold_method', "Auto")
        threshold_value = kwargs.get('threshold_value', 0)
        kernel_size = kwargs.get('kernel_size', 3)
        spot_location_coords = kwargs.get('spot_location_coords', "CenterOfMass")
        remove_bright_junk = kwargs.get('remove_bright_junk', False)
        resize_factor = kwargs.get('resize_factor', 1)
        min_area = kwargs.get('min_area', 0)
        max_area = kwargs.get('max_area', 99)
        min_integrated_intensity = kwargs.get('min_integrated_intensity', 99)
        psf_size = kwargs.get('psf_size', 1.6)
        gaussian_fit = kwargs.get('gaussian_fit', False)

        
        
        uint8_max_val = 255
        input_image = cv2.normalize(input_image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        input_image1 = input_image
        filled = nuc_mask
        if filled is None:
            filled = np.ones(input_image_raw.shape)
        #### this part is for removing bright junk in the image################
        if remove_bright_junk == True:
            labeled_nuc, num_features_nuc = label(filled)
            props = regionprops_table(labeled_nuc, input_image, properties=('label', 'area', 'max_intensity', 'mean_intensity'))
            props_df = pd.DataFrame(props)
            mean_intensity_ave=props_df['mean_intensity'].mean()
            max_intensity_max=props_df['max_intensity'].max()
            for ind, row in props_df.iterrows():

                if row['mean_intensity'] > 2*mean_intensity_ave:

                    input_image1[labeled_nuc==row['label']]=0
            input_image1[input_image1>max_intensity_max]=0 
        if resize_factor>1:
            input_image1 = rescale(input_image1.copy(), resize_factor, anti_aliasing=False, preserve_range=True)

        input_image1 = cv2.normalize(input_image1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)    
########################################
        # Spot Detection Logic
        if spot_detection_method == "Laplacian of Gaussian":
            log_result = ndimage.gaussian_laplace(input_image1, sigma=kernel_size)
            if resize_factor>1:
                
                log_result =  resize(log_result, input_image_raw.shape, anti_aliasing=False, preserve_range=True)

            if threshold_method == "Auto":

                ret_log, thresh_log = cv2.threshold(log_result.astype("uint8"),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                bin_img_log = (1-thresh_log/255).astype('bool')
                
                if resize_factor>1:
                    struct = ndimage.generate_binary_structure(2, 2)
                    bin_img_log = ndimage.binary_dilation(bin_img_log, structure=struct).astype(filled.dtype)

            if threshold_method == "Manual":
                
                manual_threshold = np.ceil(threshold_value*2.55).astype(int)
                thresh_log = log_result.astype("uint8") > manual_threshold
                bin_img_log = thresh_log
                if resize_factor>1:
                    struct = ndimage.generate_binary_structure(2, 2)
                    bin_img_log = ndimage.binary_dilation(bin_img_log, structure=struct).astype(filled.dtype)
            
            spots_img_log = (bin_img_log*255).astype('uint8')
            kernel = np.ones((3,3), np.uint8)
            spot_openned_log = cv2.morphologyEx(spots_img_log, cv2.MORPH_OPEN, kernel)
            final_spots = np.multiply(spot_openned_log,filled)
            spots_df, bin_img_log, labeled_spots = self.spots_information(final_spots, input_image_raw, gaussian_fit=gaussian_fit,  min_area=min_area, max_area=max_area,
                                                                          min_integrated_intensity=min_integrated_intensity, psf_size=psf_size)
            
        elif spot_detection_method == "Gaussian":
            result_gaussian = ndimage.gaussian_filter(input_image1, sigma=kernel_size)
            if resize_factor>1:
                result_gaussian =  resize(result_gaussian, input_image_raw.shape, anti_aliasing=False, preserve_range=True)
                
            if threshold_method == "Auto":
                
                ret_log, thresh_log = cv2.threshold(result_gaussian.astype("uint8"),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                bin_img_g = (1-thresh_log/255).astype('bool')
                
            if threshold_method == "Manual":
                
                manual_threshold = np.ceil(threshold_value*2.55).astype(int)
                
                thresh_log = result_gaussian > manual_threshold
                bin_img_g = thresh_log

            spots_img_g = ((bin_img_g>0)*255).astype('uint8') 
            kernel = np.ones((3,3), np.uint8)
            spot_openned_g = cv2.morphologyEx(spots_img_g, cv2.MORPH_OPEN, kernel)
            final_spots = np.multiply(spot_openned_g,filled)
            
            spots_df, bin_img_g, labeled_spots = self.spots_information(final_spots, input_image_raw, gaussian_fit=gaussian_fit,  min_area=min_area, max_area=max_area,
                                                                          min_integrated_intensity=min_integrated_intensity, psf_size=psf_size)
            
        if spot_detection_method == "Intensity Threshold":
            if resize_factor>1:
                input_image =  resize(input_image, input_image_raw.shape, anti_aliasing=False, preserve_range=True)
            if threshold_method == "Auto":
                
                ret_log, thresh_log = cv2.threshold(input_image.astype("uint8"),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                bin_img_g = (1-thresh_log/255).astype('bool')
                
            if threshold_method == "Manual":
                
                manual_threshold = np.ceil(threshold_value*2.55).astype(int)
                
                thresh_log = input_image > manual_threshold
                bin_img_g = thresh_log
            
            
            spots_img_g = (bin_img_g*255).astype('uint8')
            kernel = np.ones((3,3), np.uint8)
            spot_openned_g = cv2.morphologyEx(spots_img_g, cv2.MORPH_OPEN, kernel)
            
            final_spots = np.multiply(spot_openned_g,filled)
            spots_df, bin_img_g, labeled_spots = self.spots_information(final_spots, input_image_raw, 
                                                                        gaussian_fit=gaussian_fit,  min_area=min_area, max_area=max_area,
                                                                        min_integrated_intensity=min_integrated_intensity, psf_size=psf_size)

        if spot_detection_method == "Enhanced LOG":
            input_image2  = ndimage.gaussian_filter(input_image1, sigma=kernel_size/2)
            log_result = ndimage.gaussian_laplace(input_image2, sigma=kernel_size)
            if resize_factor>1:
                
                log_result =  resize(log_result, input_image_raw.shape, anti_aliasing=False, preserve_range=True)

            if threshold_method == "Auto":

                ret_log, thresh_log = cv2.threshold(log_result.astype("uint8"),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                bin_img_log = (1-thresh_log/255).astype('bool')
                
                if resize_factor>1:
                    struct = ndimage.generate_binary_structure(2, 2)
                    bin_img_log = ndimage.binary_dilation(bin_img_log, structure=struct).astype(filled.dtype)
                    
            spots_img_log = (bin_img_log*255).astype('uint8')
            kernel = np.ones((3,3), np.uint8)
            spot_openned_log = cv2.morphologyEx(spots_img_log, cv2.MORPH_OPEN, kernel)
            final_spots = np.multiply(spot_openned_log,filled)
            spots_df, bin_img_log, labeled_spots = self.spots_information(final_spots, input_image_raw, gaussian_fit=gaussian_fit,  min_area=min_area, max_area=max_area,
                                                                          min_integrated_intensity=min_integrated_intensity, psf_size=psf_size)   
        # Location calculation logic
        if spot_location_coords == "CenterOfMass":
            try:
                spot_locations = list(spots_df['center_of_mass_coords'].to_numpy())
            except:
                spot_locations = []
                
        elif spot_location_coords == "MaxIntensity":
            try:
                spot_locations = list(spots_df['max_intensity_coords'].to_numpy())
            except:
                spot_locations = []
        elif spot_location_coords == "Centroid":
            
            labeled_spots, num_features = label(final_spots)
            spot_labels = np.unique(labeled_spots)
            try:
                spot_locations = ndimage.measurements.center_of_mass(final_spots, labeled_spots, spot_labels[spot_labels>0])
            except:
                spot_locations = []
        # print(spots_df)
        spots_df = spots_df.reset_index(drop=True)

        return spot_locations, spots_df   
    
    
    def spots_information(self, bin_img_log, max_project, gaussian_fit= False,  
                          min_area=0, max_area=100, min_integrated_intensity=0, psf_size=1.6):
        """
        Extracts and analyzes information about spots detected in an image, 
        performing additional processing and calculations to refine the spot detection results.

        Parameters:
            - bin_img_log (numpy.ndarray): Binary image where spots are identified.
            - max_project (numpy.ndarray): Maximum projection image used for intensity measurements.
            - gaussian_fit (bool, optional): Flag to perform Gaussian fitting on spots. Default is False.
            - min_area (int, optional): Minimum area threshold for spots to be considered. Default is 0.
            - max_area (int, optional): Maximum area threshold for spots. Default is 100.
            - min_integrated_intensity (int, optional): Minimum integrated intensity threshold for spots. Default is 0.
            - psf_size (float, optional): Point spread function size used in Gaussian fitting. Default is 1.6.

        Returns:
            - props_df (pandas.DataFrame): DataFrame containing properties of each spot, including area, intensity metrics, and location.
            - new_bin_img_log (numpy.ndarray): Refined binary image of spots after processing.
            - labeled_spots (numpy.ndarray): Image with labeled spots.

        The function performs several operations including labeling of spots, filtering based on area and intensity, 
        and optional Gaussian fitting. It calculates various properties of the spots, such as area, intensity, and location coordinates, 
        and returns a DataFrame with these properties along with refined binary and labeled images of the spots.

        Usage Example:
            spots_props, refined_spots, labeled_spots = spots_information(binary_image, max_projection, gaussian_fit=True, min_area=5)

        Note:
            Requires scipy, numpy, pandas, and optionally cv2 for Gaussian fitting. The parameters for area and intensity thresholds should be 
            chosen based on the specific requirements of the image analysis task.
        """
        
        labeled_spots, num_features = label(bin_img_log)
        if num_features>0:
            props = regionprops_table(labeled_spots, max_project,  properties=( 'label',  'area', 'solidity','coords'))
            props_df = pd.DataFrame(props)
            new_bin_img_log = np.zeros(bin_img_log.shape)
            for spot_no in range(len(props_df)):
                spot_patch=max_project[props_df['coords'].iloc[spot_no][:,0].min():props_df['coords'].iloc[spot_no][:,0].max(),
                                       props_df['coords'].iloc[spot_no][:,1].min():props_df['coords'].iloc[spot_no][:,1].max()]
                if (spot_patch.shape[0]>0)&(spot_patch.shape[1]>0):
                    spot_fwhm = np.multiply(spot_patch>spot_patch.max()/2,spot_patch)
                    new_bin_img_log[props_df['coords'].iloc[spot_no][:,0].min():props_df['coords'].iloc[spot_no][:,0].max(),
                                    props_df['coords'].iloc[spot_no][:,1].min():props_df['coords'].iloc[spot_no][:,1].max()] = spot_fwhm>0
            
            kernel = np.ones((3,3), np.uint8)
            new_bin_img_log = cv2.morphologyEx(new_bin_img_log, cv2.MORPH_CLOSE, kernel)
            
            labeled_spots, num_features = label(new_bin_img_log)
            if num_features>0:

                props = regionprops_table(labeled_spots, max_project, properties=( 'area', 'max_intensity', 'min_intensity', 'mean_intensity',
                                                        'perimeter', 'solidity','coords', 'bbox'))
                props_df = pd.DataFrame(props)
                masked_spots = np.multiply(max_project,new_bin_img_log)
                spot_labels = np.unique(labeled_spots)            
                props_df['center_of_mass_coords'] = ndimage.measurements.center_of_mass(masked_spots, labeled_spots, spot_labels[spot_labels>0])
                props_df['max_intensity_coords'] = props_df.apply(lambda row : tuple(row['coords'][max_project[row['coords'][:,0], 
                                                                                             row['coords'][:,1]].argmax(),:]), axis = 1)
        
                nospot_max_project = np.multiply(max_project,~(new_bin_img_log>0))
                r=2
                for i in range(len(props_df)):

                    y0 = max(props_df.loc[i, "bbox-0"]-r,0)
                    y1 = min(props_df.loc[i, "bbox-2"]+r, max_project.shape[0])
                    x0 = max(props_df.loc[i, "bbox-1"]-r,0)
                    x1 = min(props_df.loc[i, "bbox-3"]+r, max_project.shape[1])
                    
                    
                    spot_patch = max_project[y0:y1,x0:x1]
                    fit_results = self.gmask_fit(spot_patch, xy_input=np.array(props_df.loc[i,'center_of_mass_coords'])-np.array([y0,x0]), 
                                                 fit=gaussian_fit, psf_size=psf_size)
                        
                    spot_area_patch = nospot_max_project[y0:y1,x0:x1]
                    nonzero_patch = spot_area_patch[np.nonzero(spot_area_patch)]
                    
                    
                    props_df.loc[i,"integrated_intensity"] = fit_results[2]
                    props_df.loc[i,"spot_area_mean"]= nonzero_patch.mean()
                    props_df.loc[i,"spot_area_std"]= nonzero_patch.std()
                    props_df.loc[i,"spot_area_median"]=np.median(nonzero_patch)
                    
                    props_df.loc[i,"spot_max_to_min"]=props_df.loc[i, "max_intensity"]/props_df.loc[i, "min_intensity"]
                    props_df.loc[i,"spot_to_area_mean"]=props_df.loc[i, "mean_intensity"]/props_df.loc[i, "spot_area_mean"]
                    props_df.loc[i,"spot_max_to_area_mean"]=props_df.loc[i, "max_intensity"]/props_df.loc[i, "spot_area_mean"]
                # props_df = props_df.loc[(props_df["spot_to_area_mean"]>1.2)& (props_df["spot_max_to_area_mean"]>1.2)]
                props_df = props_df.loc[(props_df["spot_max_to_area_mean"]>1.2)]
                props_df = props_df.reset_index(drop=True)
                
                props_df = props_df.loc[(props_df["area"]>min_area) & (props_df["area"]<max_area) & (props_df["integrated_intensity"]>min_integrated_intensity)]
                
        else:
            
            props_df = pd.DataFrame(columns=['area', 'max_intensity', 'min_intensity', 'mean_intensity', 'center_of_mass_coords',
                                             'max_intensity_coords', 'perimeter', 'solidity','coords', 'bbox', 'spot_area_mean',
                                             'spot_area_std', 'spot_area_median', 'integrated_intensity'])
            new_bin_img_log = np.zeros(bin_img_log.shape)
        props_df = props_df.drop(columns=['coords'])

        return props_df,new_bin_img_log, labeled_spots
    
    def gmask_fit(self, pic, xy_input=None, fit=False, psf_size=1.6):
        """
        Performs Gaussian mask fitting on an image to determine the centroid and photon number of a spot.

        Parameters:
            - pic (numpy.ndarray): The image (or a patch of the image) containing the spot to be analyzed.
            - xy_input (tuple, optional): Initial guess for the centroid coordinates (x0, y0). Required if 'fit' is False.
            - fit (bool, optional): If True, performs iterative fitting to find the centroid. If False, uses 'xy_input' as the centroid. Default is False.
            - psf_size (float, optional): Point spread function size, used in the Gaussian mask. Default is 1.6.

        Returns:
            - results (numpy.ndarray): A numpy array containing the fitted centroid coordinates (x0, y0) and the calculated photon number.

        This function applies a Gaussian mask fitting method to an image to determine the centroid coordinates and the number of photons 
        in a spot. It either uses an iterative fitting process or a fixed position based on the provided initial guess. The function 
        subtracts the local background before the fitting process and uses the error function (erf) to create the Gaussian mask.

        Usage Example:
            fitted_results = gmask_fit(image_patch, xy_input=(10, 10), fit=True, psf_size=1.5)

        Note:
            Requires numpy and scipy (for the error function). The 'pic' should be a cropped image or a patch containing the spot for accurate fitting. 
            The 'fit' parameter determines whether to perform iterative fitting or use a fixed centroid position.
        """
        s = pic.shape
        x_dim = s[0]
        y_dim = s[1]
        x0 = 0.0  # center of mass coordinates
        y0 = 0.0
        
        
        F = 1.0 / (np.sqrt(2.0) * psf_size)  # This factor shows up repeatedly in GMASK
        gauss_mask = np.zeros((x_dim, y_dim))  # gaussian mask for centroid fitting
        error = 0.0  # rms distance between the 'true' spot and the centroid spot
        results = np.zeros(3)  # this array hold the returned results of the function: x0, y0, number of photons in spot

        blksubtract = pic - self.local_background(pic)
        image = (blksubtract > 0) * blksubtract
        # tvscl, image
        # boundary condition.  border is set to zero
        image[0, :] = 0.0
        image[x_dim - 1, :] = 0.0
        image[:, 0] = 0.0
        image[:, y_dim - 1] = 0.0
        # iterative centroid calculation with gaussian mask
        h = 1.0e-3  # 5.0e-16       ; tolerance
        diff_x = 0.0
        diff_y = 0.0
        repeat_index = 0

        x_dim = int(x_dim)
        y_dim = int(y_dim)
        array = np.arange(x_dim * y_dim)
        xarr = array % x_dim
        yarr = array // x_dim

        if fit==False:
            x0 = xy_input[0]
            y0 = xy_input[1]
            a = F * (yarr - 0.5 - y0)
            b = F * (yarr + 0.5 - y0)
            c = F * (xarr - 0.5 - x0)
            d = F * (xarr + 0.5 - x0)
            gauss_mask = 0.25 * (erf(a) - erf(b)) * (erf(c) - erf(d))
        else:

            while True:
                x0 = x0 + diff_x / 2.0
                y0 = y0 + diff_y / 2.0
                # print, x0, y0
                a = F * (yarr - 0.5 - y0)
                b = F * (yarr + 0.5 - y0)
                c = F * (xarr - 0.5 - x0)
                d = F * (xarr + 0.5 - x0)
                gauss_mask = 0.25 * (erf(a) - erf(b)) * (erf(c) - erf(d))
                sum_ = np.sum(image.ravel() * gauss_mask)
                trial_x0 = np.sum(xarr * image.ravel() * gauss_mask)
                trial_y0 = np.sum(yarr * image.ravel() * gauss_mask)
                # print, x0, trial_x0, sum, trial_x0/sum
                diff_x = trial_x0 / sum_ - x0
                diff_y = trial_y0 / sum_ - y0
                repeat_index = repeat_index + 1
                if ((np.abs(diff_x) < h) and (np.abs(diff_y) < h)) or (repeat_index > 300):
                    break

        if (repeat_index > 300):
            # print, "GMASK ITERATION MAXED OUT (number of iterations=", repeat_index, ")"
            results[2] = 0.0
            return results

        # photon number calc
        sum_ = np.sum(gauss_mask * gauss_mask)
        N = np.sum(image.ravel() * gauss_mask)

        photon_number=N/sum_
        results[0]=x0
        results[1]=y0
        results[2]=photon_number

        return results
    def local_background(self, pic, display=False):
        """
        Calculates the local background of an image using a linear fit to the border pixels.

        Parameters:
            - pic (numpy.ndarray): The input image for which the local background needs to be calculated.
            - display (bool, optional): Flag to display intermediate calculation steps. Default is False.

        Returns:
            - plane (numpy.ndarray): An array of the same shape as 'pic', containing the calculated local background plane.

        This function calculates the local background by fitting a linear plane to the border pixels of the input image. 
        It uses a method similar to that described in Bevington's book (p. 96) to calculate the fit parameters and then applies 
        these parameters to define a background plane. This background can be subtracted from the original image to correct 
        for varying background levels.

        Usage Example:
            background_plane = local_background(image)

        Note:
            Designed for use with 2D numpy arrays representing images. The 'display' parameter can be set to True for debugging purposes 
            to observe the intermediate steps of the calculation.
        """
        
        # get the dimensions of the input and extract border coordinates
        pic_dim = np.shape(pic)
        x_dim = pic_dim[0]
        y_dim = pic_dim[1]

        x_border = np.zeros(2 * x_dim)
        y_border = np.zeros(2 * y_dim)

        x_border[0:x_dim] = pic[:, 0]
        x_border[x_dim:2*x_dim] = pic[:, y_dim-1]

        x = np.concatenate((np.arange(x_dim), np.arange(x_dim)))

        y_border[0:y_dim] = pic[0, :]
        y_border[y_dim:2*y_dim] = pic[x_dim-1, :]

        y = np.concatenate((np.arange(y_dim), np.arange(y_dim)))

        # following the method of Bevington, p. 96
        delta_x = 2 * x_dim * np.sum(x**2) - np.sum(x)**2
        a = (1. / delta_x) * (np.sum(x**2) * np.sum(x_border) - np.sum(x) * np.sum(x * x_border))
        b = (1. / delta_x) * (2 * x_dim * np.sum(x * x_border) - np.sum(x) * np.sum(x_border))

        delta_y = 2 * y_dim * np.sum(y**2) - np.sum(y)**2
        c = (1. / delta_y) * (np.sum(y**2) * np.sum(y_border) - np.sum(y) * np.sum(y * y_border))
        d = (1. / delta_y) * (2 * y_dim * np.sum(y * y_border) - np.sum(y) * np.sum(y_border))

        # The offset which is returned is averaged over each edge in x, and each edge in y.
        # The actual offset needs to be corrected for the tilt of the plane.
        # Then, the 2 offsets are averaged together to give a single offset.
        offset = (a - d * (y_dim - 1) / 2.0 + c - b * (x_dim - 1) / 2.0) / 2.0

        # now define the background plane in terms of the fit parameters
        plane = np.zeros((x_dim, y_dim))
        for i in range(x_dim):
            for j in range(y_dim):
                plane[i, j] = offset + b * float(i) + d * float(j)

        return plane

    def COORDINATES_TO_CIRCLE(self, coordinates,ImageForSpots, circ_radius =5):
        """
        Creates an image with circles drawn at specified coordinates.

        Parameters:
            - coordinates (numpy.ndarray): An array of coordinates where each row represents a point (y, x).
            - ImageForSpots (numpy.ndarray): The base image used to define the shape of the output image.
            - circ_radius (int, optional): The radius of the circles to be drawn. Default is 5.

        Returns:
            - circles (numpy.ndarray): An image of the same size as 'ImageForSpots' with circles drawn at the specified coordinates.

        This function takes a set of coordinates and draws circles of a given radius at these coordinates on an image. 
        The output is a binary image where the circles are marked in white (255) on a black (0) background.

        Usage Example:
            circle_image = COORDINATES_TO_CIRCLE(coordinates_array, base_image, circ_radius=10)

        Note:
            Requires numpy and skimage.draw (for circle_perimeter). Ensure that the coordinates are within the bounds of 'ImageForSpots'.
        """
        circles = np.zeros((ImageForSpots.shape), dtype=np.uint8)

        if coordinates.any():
            
            for center_y, center_x in zip(coordinates[:,0], coordinates[:,1]):
                    circy, circx = circle_perimeter(center_y, center_x, circ_radius, shape=ImageForSpots.shape)
                    circles[circy, circx] = 255

        return circles
    
    def SPOTS_TO_BOUNDARY(self, final_spots):
        """
        Converts a binary image of spots into an image where the boundaries of these spots are highlighted.

        Parameters:
            - final_spots (numpy.ndarray): A binary image where spots are represented by non-zero pixels.

        Returns:
            - spot_boundary (numpy.ndarray): An image where the boundaries of the spots in 'final_spots' are marked.

        This function identifies the boundaries of spots in a binary image and creates a new image where these boundaries are highlighted. 
        The output is a binary image with boundaries marked in white (255) on a black (0) background.

        Usage Example:
            boundary_image = SPOTS_TO_BOUNDARY(spots_image)

        Note:
            Requires numpy and skimage.segmentation (for find_boundaries). The input should be a binary image with spots marked.
        """
        labeled_spots, num_features = label(final_spots)
        boundary = find_boundaries(labeled_spots, connectivity=1, mode='thick', background=0)
        spot_boundary = (255*boundary).astype('uint8')
        
        return spot_boundary
