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
from PyQt5.QtWidgets import QComboBox

# from GUI_parameters import Gui_Params
# from dask.distributed import Client, LocalCluster

class ImageAnalyzer(object):
    

    def __init__(self,analysisgui, inout_resource_gui):
        self.AnalysisGui = analysisgui
        self.inout_resource_gui = inout_resource_gui
        spot_params_dict =self.INITIALIZE_SPOT_ANALYSIS_PARAMS()
    
    def neuceli_segmenter(self, input_img, pixpermic=None):
        
        if self.AnalysisGui.NucDetectMethod.currentText() == "Int.-based":
            
            first_tresh = self.AnalysisGui.NucSeparationSlider.value()*2.55
            second_thresh = 255-(self.AnalysisGui.NucDetectionSlider.value()*2.55)
            Cell_Size = self.AnalysisGui.NucleiAreaSlider.value()
            
            boundary, mask = self.segmenter_function(input_img, cell_size=Cell_Size, 
                                                     first_threshold=first_tresh, second_threshold=second_thresh)
            
        if self.AnalysisGui.NucDetectMethod.currentText() == "Marker Controlled":
            
          
            Cell_Size = self.AnalysisGui.NucleiAreaSlider.value()
            max_range = np.sqrt(Cell_Size/3.14)*2/float(pixpermic)
            nuc_detect_sldr = self.AnalysisGui.NucDetectionSlider.value()
            first_tresh = np.ceil((1-(nuc_detect_sldr/100))*max_range).astype(int)
            
            second_thresh = self.AnalysisGui.NucSeparationSlider.value()
            
            boundary, mask = self.watershed_scikit(input_img, cell_size=Cell_Size, 
                                                     first_threshold=first_tresh, second_threshold=second_thresh)

        if self.AnalysisGui.NucDetectMethod.currentText() == "CellPose-GPU":
#             pixpermic = 0.1
            Cell_Size = self.AnalysisGui.NucleiAreaSlider.value()
            cell_diameter = np.sqrt(Cell_Size/(float(pixpermic)*float(pixpermic)))*2/3.14
            
            boundary, mask = self.cellpose_segmenter(input_img, use_GPU=1, cell_dia=cell_diameter)
            
        if self.AnalysisGui.NucDetectMethod.currentText() == "CellPose-CPU":
            
            Cell_Size = self.AnalysisGui.NucleiAreaSlider.value()
            cell_diameter = np.sqrt(Cell_Size/(float(pixpermic)*float(pixpermic)))*2/3.14
            
            boundary, mask = self.cellpose_segmenter(input_img, use_GPU=0, cell_dia=cell_diameter)
            
        if self.AnalysisGui.NucDetectMethod.currentText() == "CellPose-Cyto":
            
            Cell_Size = self.AnalysisGui.NucleiAreaSlider.value()
            cell_diameter = np.sqrt(Cell_Size/(float(pixpermic)*float(pixpermic)))*2/3.14
            
            boundary, mask = self.cellpose_segmenter(input_img, use_GPU=1, cell_dia=cell_diameter)
                
        if self.AnalysisGui.NucDetectMethod.currentText() == "DeepCell":
            
            # Cell_Size = self.AnalysisGui.NucleiAreaSlider.value()
            # cell_diameter = Cell_Size/100
            
            boundary, mask = self.deepcell_segmenter(input_img, mmp=float(pixpermic))
        
#         kernel = np.ones((7,7), np.uint8)
#         first_pass = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         boundary = find_boundaries(first_pass, connectivity=1, mode='thick', background=0)
#         mask= (255*first_pass).astype('uint8')
#         boundary= (255*boundary).astype('uint8')
        
        return boundary, mask   


    def deepcell_segmenter(self, input_img, mmp=None):
        
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
        
        if self.AnalysisGui.NucRemoveBoundaryCheckBox.isChecked() == True:
            img_uint8 = input_img
        else:
            img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)

        if self.AnalysisGui.NucDetectMethod.currentText() == "CellPose-Cyto":
            model = models.Cellpose(gpu=use_GPU, model_type='cyto')
        else:
            model = models.Cellpose(gpu=use_GPU, model_type='nuclei')
        masks, flows, styles, diams = model.eval(img_uint8, diameter=cell_dia, flow_threshold=None)
                
        boundary = find_boundaries(masks, connectivity=1, mode='thick', background=0)

        if self.AnalysisGui.NucRemoveBoundaryCheckBox.isChecked() == True:

            boundary_img = (255*boundary).astype('uint8')
            filled1 = ndimage.binary_fill_holes(boundary_img)
            mask1= (255*filled1).astype('uint8')-boundary_img
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask1.astype('uint8'), cv2.MORPH_OPEN, kernel)
#             kernel = np.ones((1,1),np.uint8)
#             erosion = cv2.erode(mask,kernel,iterations = 1)

            
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
    
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        
        ## First blurring round
        if (cell_size %2)==0:
            cell_size = cell_size + 1
        median_img = cv2.medianBlur(img_uint8,cell_size)
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
        
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        
        med_scikit = median(img_uint8, disk(1))
        thresh = threshold_li(med_scikit)
        binary = med_scikit > thresh
        filled = ndimage.binary_fill_holes(binary)
        filled_blurred = gaussian(filled, 1)
        filled_int= (filled_blurred*255).astype('uint8')
        
# #         edge_sobel = sobel(img_uint8)
# #         enhanced = 50*edge_sobel/edge_sobel.max() + img_uint8
# #         enhanced.astype('uint8')
# #         med_scikit = median(img_uint8, disk(5))
        thresh = threshold_li(filled_int)
        binary = filled_int > thresh
        filled = ndimage.binary_fill_holes(binary)
        filled_int = binary_opening(filled, disk(5))
        filled_int = ndimage.binary_fill_holes(filled_int)
#         filled_blurred = gaussian(openeed, 3)
        
#         thresh = threshold_li(filled_int)
#         binary = filled_int > thresh
        #binary = binary_erosion(filled_int, disk(5))
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
        
        z_imglist=[]
        
        for index, row in image_stack.iterrows():
            if row['ImageName']=="dask_array":
                # Create a LocalCluster with 4 workers
#                 cluster = LocalCluster(n_workers=1)

#                 # Create a Client to connect to the LocalCluster
#                 client = Client(cluster)

                # im = row["Type"].compute()
                im = row["Type"]
                # Close the Client and LocalCluster
                # client.close()
                # cluster.close()
            else: 
                im = Image.open(row['ImageName'])
            z_imglist.append( np.asarray(im))
        z_stack = np.stack(z_imglist, axis=2)
        max_project = z_stack.max(axis=2)
        
        return max_project
    
   
    
    def SpotDetector(self, input_image_raw, AnalysisGui, nuclei_image, spot_channel):
        
        self.UPDATE_SPOT_ANALYSIS_PARAMS()
        if AnalysisGui.spotchannelselect.currentText()=='All':
            
            params_to_pass= self.spot_params_dict['Ch1']
        else:
            params_to_pass= self.spot_params_dict[spot_channel]
        
        uint8_max_val = 255
#         w,h = input_image.shape
#         dim = (2*w,2*h)
#         input_image1 = cv2.resize(input_image, dim, interpolation = cv2.INTER_AREA)
        
#         noise = np.random.normal(1*input_image_raw.mean(), 1*input_image_raw.std(), input_image_raw.shape)
#         input_image_raw = input_image_raw +noise
        
        input_image = cv2.normalize(input_image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        input_image1 = input_image
        ## First blurring round
        median_img = cv2.medianBlur(nuclei_image,11)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(median_img,0,uint8_max_val,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        bin_img = (1-thresh/uint8_max_val).astype('bool')
        ## Binary image filling
        filled = ndimage.binary_fill_holes(bin_img)
        struct = ndimage.generate_binary_structure(2, 2)
        filled = ndimage.binary_dilation(filled, structure=struct).astype(filled.dtype)
        filled = ndimage.binary_dilation(filled, structure=struct).astype(filled.dtype)
        boundary, filled = self.neuceli_segmenter(nuclei_image, 0.2)
        #### this part is for removing bright junk in the image################
        if self.AnalysisGui.RemoveBrightJunk.isChecked() == True:
            labeled_nuc, num_features_nuc = label(filled)
            props = regionprops_table(labeled_nuc, input_image, properties=('label', 'area', 'max_intensity', 'mean_intensity'))
            props_df = pd.DataFrame(props)
            mean_intensity_ave=props_df['mean_intensity'].mean()
            max_intensity_max=props_df['max_intensity'].max()
            for ind, row in props_df.iterrows():

                if row['mean_intensity'] > 2*mean_intensity_ave:

                    input_image1[labeled_nuc==row['label']]=0
            input_image1[input_image1>max_intensity_max]=0 
        if AnalysisGui.SpotPerChSpinBox.value()>1:
            input_image1 = rescale(input_image1.copy(), AnalysisGui.SpotPerChSpinBox.value(), anti_aliasing=False, preserve_range=True)

        input_image1 = cv2.normalize(input_image1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)    
########################################
        sig=params_to_pass[3]
        if params_to_pass[0] == 0:

            log_result = ndimage.gaussian_laplace(input_image1, sigma=sig)
            if AnalysisGui.SpotPerChSpinBox.value()>1:
                
                log_result =  resize(log_result, input_image_raw.shape, anti_aliasing=False, preserve_range=True)

            if params_to_pass[1] == 0:

                ret_log, thresh_log = cv2.threshold(log_result.astype("uint8"),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                bin_img_log = (1-thresh_log/255).astype('bool')
                
                if AnalysisGui.SpotPerChSpinBox.value()>1:
                    struct = ndimage.generate_binary_structure(2, 2)
                    bin_img_log = ndimage.binary_dilation(bin_img_log, structure=struct).astype(filled.dtype)

            if params_to_pass[1] == 1:
                
                manual_threshold = np.ceil(params_to_pass[2]*2.55).astype(int)
                thresh_log = log_result.astype("uint8") > manual_threshold
                bin_img_log = thresh_log
                if AnalysisGui.SpotPerChSpinBox.value()>1:
                    struct = ndimage.generate_binary_structure(2, 2)
                    bin_img_log = ndimage.binary_dilation(bin_img_log, structure=struct).astype(filled.dtype)
            
            spots_img_log = (bin_img_log*255).astype('uint8')
            kernel = np.ones((3,3), np.uint8)
            spot_openned_log = cv2.morphologyEx(spots_img_log, cv2.MORPH_OPEN, kernel)
            final_spots = np.multiply(spot_openned_log,filled)
            spots_df, bin_img_log, labeled_spots = self.spots_information(final_spots, input_image_raw, AnalysisGui)   
        if str(params_to_pass[0]) == '1':
            
            result_gaussian = ndimage.gaussian_filter(input_image1, sigma=sig)
            if AnalysisGui.SpotPerChSpinBox.value()>1:
                result_gaussian =  resize(result_gaussian, input_image_raw.shape, anti_aliasing=False, preserve_range=True)
                
            if str(params_to_pass[1]) == '0':
                
                ret_log, thresh_log = cv2.threshold(result_gaussian.astype("uint8"),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                bin_img_g = (1-thresh_log/255).astype('bool')
                
            if str(params_to_pass[1]) == '1':
                
                manual_threshold = np.ceil(params_to_pass[2]*2.55).astype(int)
                
                thresh_log = result_gaussian > manual_threshold
                bin_img_g = thresh_log

            # kernel = np.ones((3,3), np.uint8)
            # spot_openned_g = cv2.morphologyEx(bin_img_g, cv2.MORPH_OPEN, kernel)
            # final_spots = np.multiply(bin_img_g,filled)
            
#             final_spots = ((bin_img_g>0)*255).astype('uint8') 
            spots_img_g = ((bin_img_g>0)*255).astype('uint8') 
            kernel = np.ones((3,3), np.uint8)
            spot_openned_g = cv2.morphologyEx(spots_img_g, cv2.MORPH_OPEN, kernel)
            final_spots = np.multiply(spot_openned_g,filled)
            
            spots_df, bin_img_g, labeled_spots = self.spots_information(final_spots, input_image_raw, AnalysisGui)
        
        if str(params_to_pass[0]) == '2':
            if AnalysisGui.SpotPerChSpinBox.value()>1:
                input_image =  resize(input_image, input_image_raw.shape, anti_aliasing=False, preserve_range=True)
            if str(params_to_pass[1]) == '0':
                
                ret_log, thresh_log = cv2.threshold(input_image.astype("uint8"),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                bin_img_g = (1-thresh_log/255).astype('bool')
                
            if str(params_to_pass[1]) == '1':
                
                manual_threshold = np.ceil(params_to_pass[2]*2.55).astype(int)
                
                thresh_log = input_image > manual_threshold
                bin_img_g = thresh_log
            
            
            spots_img_g = (bin_img_g*255).astype('uint8')
            kernel = np.ones((3,3), np.uint8)
            spot_openned_g = cv2.morphologyEx(spots_img_g, cv2.MORPH_OPEN, kernel)
            
            final_spots = np.multiply(spot_openned_g,filled)
            spots_df, bin_img_g, labeled_spots = self.spots_information(final_spots, input_image_raw, AnalysisGui)
            
        if str(params_to_pass[0]) == '3':
            input_image2  = ndimage.gaussian_filter(input_image1, sigma=sig/2)
            log_result = ndimage.gaussian_laplace(input_image2, sigma=sig)
            if AnalysisGui.SpotPerChSpinBox.value()>1:
                
                log_result =  resize(log_result, input_image_raw.shape, anti_aliasing=False, preserve_range=True)

            if params_to_pass[1] == 0:

                ret_log, thresh_log = cv2.threshold(log_result.astype("uint8"),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                bin_img_log = (1-thresh_log/255).astype('bool')
                
                if AnalysisGui.SpotPerChSpinBox.value()>1:
                    struct = ndimage.generate_binary_structure(2, 2)
                    bin_img_log = ndimage.binary_dilation(bin_img_log, structure=struct).astype(filled.dtype)
                    
            spots_img_log = (bin_img_log*255).astype('uint8')
            kernel = np.ones((3,3), np.uint8)
            spot_openned_log = cv2.morphologyEx(spots_img_log, cv2.MORPH_OPEN, kernel)
            final_spots = np.multiply(spot_openned_log,filled)
            spots_df, bin_img_log, labeled_spots = self.spots_information(final_spots, input_image_raw, AnalysisGui)   
            
        ### center of mass calculation
        if str(AnalysisGui.SpotLocationCbox.currentIndex()) == '0':
            
            try:
                spot_locations = list(spots_df['center_of_mass_coords'].to_numpy())
            except:
                spot_locations = []
            ###### Brightest spot calculation
        if str(AnalysisGui.SpotLocationCbox.currentIndex()) == '1':
            try:
                spot_locations = list(spots_df['max_intensity_coords'].to_numpy())
            except:
                spot_locations = []
            ##### Centroid calculation
        if str(AnalysisGui.SpotLocationCbox.currentIndex()) == '2':
            
            labeled_spots, num_features = label(final_spots)
            spot_labels = np.unique(labeled_spots)
            try:
                spot_locations = ndimage.measurements.center_of_mass(final_spots, labeled_spots, spot_labels[spot_labels>0])
            except:
                spot_locations = []
        # print(spots_df)
        spots_df = spots_df.reset_index(drop=True)
        return spot_locations, spots_df
    
    def spots_information(self, bin_img_log, max_project, AnalysisGui ):
        
        labeled_spots, num_features = label(bin_img_log)
        if num_features>0:
            props = regionprops_table(labeled_spots, max_project,  properties=( 'label',  'area', 'solidity','coords'))
            props_df = pd.DataFrame(props)
#             props_df = props_df.loc[(props_df["area"]<25) & (props_df["area"]>5) & (props_df["solidity"]>0.98)]
            # props_df = props_df.loc[props_df['area']>4]
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
                    IntegratedIntensity_fitStatus = self.AnalysisGui.IntegratedIntensityCbox.currentIndex() > 0 
                    fit_results = self.gmask_fit(spot_patch, xy_input=np.array(props_df.loc[i,'center_of_mass_coords'])-np.array([y0,x0]), fit=IntegratedIntensity_fitStatus)
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
                
                min_area = AnalysisGui.SpotareaminSpinBox.value()
                max_area = AnalysisGui.SpotareamaxSpinBox.value()
                spot_integrated_intensity = AnalysisGui.SpotIntegratedIntensitySpinBox.value()
                props_df = props_df.loc[(props_df["area"]>min_area) & (props_df["area"]<max_area) & (props_df["integrated_intensity"]>spot_integrated_intensity)]
                
#                 print("before filtering...")
#                 print(props_df)
#                 true_ind = self.model1.predict(props_df[['max_intensity', 'min_intensity', 'mean_intensity','spot_area_mean', 'spot_area_std', 'spot_area_median']].to_numpy())
#                 props_df = props_df.loc[np.where(true_ind)[0]].reset_index(drop=True)
#                 print("after filtering...")
#                 print(props_df)
        else:
            
            props_df = pd.DataFrame(columns=['area', 'max_intensity', 'min_intensity', 'mean_intensity', 'center_of_mass_coords',
                                             'max_intensity_coords', 'perimeter', 'solidity','coords', 'bbox', 'spot_area_mean',
                                             'spot_area_std', 'spot_area_median', 'integrated_intensity'])
            new_bin_img_log = np.zeros(bin_img_log.shape)
        props_df = props_df.drop(columns=['coords'])

        return props_df,new_bin_img_log, labeled_spots
    
    def gmask_fit(self, pic, xy_input=None, fit=False):
        s = pic.shape
        x_dim = s[0]
        y_dim = s[1]
        x0 = 0.0  # center of mass coordinates
        y0 = 0.0
        
        
        F = 1.0 / (np.sqrt(2.0) * self.AnalysisGui.PSFsizeSpinBox.value())  # This factor shows up repeatedly in GMASK
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
        
        circles = np.zeros((ImageForSpots.shape), dtype=np.uint8)

        if coordinates.any():
            
            for center_y, center_x in zip(coordinates[:,0], coordinates[:,1]):
                    circy, circx = circle_perimeter(center_y, center_x, circ_radius, shape=ImageForSpots.shape)
                    circles[circy, circx] = 255

        return circles
    
    def SPOTS_TO_BOUNDARY(self, final_spots):
        
        labeled_spots, num_features = label(final_spots)
        boundary = find_boundaries(labeled_spots, connectivity=1, mode='thick', background=0)
        spot_boundary = (255*boundary).astype('uint8')
        
        return spot_boundary
    

    def INITIALIZE_SPOT_ANALYSIS_PARAMS(self):

        # Define a function to get the parameters for a channel
        def get_channel_params():
            return np.array([
                self.AnalysisGui.spotanalysismethod.currentIndex(),
                self.AnalysisGui.thresholdmethod.currentIndex(),
                self.AnalysisGui.ThresholdSlider.value(), 
                self.AnalysisGui.SensitivitySpinBox.value(),
                self.AnalysisGui.SpotPerChSpinBox.value(),
                self.AnalysisGui.SpotareaminSpinBox.value(),
                self.AnalysisGui.SpotareamaxSpinBox.value(),
                self.AnalysisGui.SpotIntegratedIntensitySpinBox.value()
            ])

        # Populate the dictionary using a loop for each channel
        self.spot_params_dict = {
            f"Ch{i}": get_channel_params() for i in range(1, 6)
        }

        return self.spot_params_dict
    
    def UPDATE_SPOT_ANALYSIS_PARAMS(self):

        # Define a function to get the parameters for a channel
        def get_channel_params():
            return np.array([
                self.AnalysisGui.spotanalysismethod.currentIndex(),
                self.AnalysisGui.thresholdmethod.currentIndex(),
                self.AnalysisGui.ThresholdSlider.value(), 
                self.AnalysisGui.SensitivitySpinBox.value(),
                self.AnalysisGui.SpotPerChSpinBox.value(),
                self.AnalysisGui.SpotareaminSpinBox.value(),
                self.AnalysisGui.SpotareamaxSpinBox.value(),
                self.AnalysisGui.SpotIntegratedIntensitySpinBox.value()
            ])

        current_channel = self.AnalysisGui.spotchannelselect.currentText()

        if current_channel == 'All':
            self.spot_params_dict = {
                f"Ch{i}": get_channel_params() for i in range(1, 6)
            }
        elif current_channel in ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']:
            self.spot_params_dict[current_channel] = get_channel_params()

            
    def UPDATE_SPOT_ANALYSIS_GUI_PARAMS(self):

        current_channel = self.AnalysisGui.spotchannelselect.currentText()

        if current_channel in ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']:

            gui_elements = [
                self.AnalysisGui.spotanalysismethod,  # QComboBox
                self.AnalysisGui.thresholdmethod,     # QComboBox
                self.AnalysisGui.ThresholdSlider,     # QSlider or QSpinBox
                self.AnalysisGui.SensitivitySpinBox,  # QSpinBox
                self.AnalysisGui.SpotPerChSpinBox,    # QSpinBox
                self.AnalysisGui.SpotareaminSpinBox,  # QSpinBox
                self.AnalysisGui.SpotareamaxSpinBox,  # QSpinBox
                self.AnalysisGui.SpotIntegratedIntensitySpinBox  # QSpinBox
            ]

            for i, gui_element in enumerate(gui_elements):
                value = np.array(self.spot_params_dict[current_channel][i]).astype(int)
                if isinstance(gui_element, QComboBox):
                    gui_element.setCurrentIndex(value)
                else:
                    gui_element.setValue(value)
