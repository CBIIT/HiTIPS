import numpy as np
import cv2
import math
import os
import time
import sys
import shutil
import stat
from skimage.measure import regionprops, regionprops_table
from skimage.io import imread
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from PIL import Image, ImageQt
from scipy.ndimage import label, distance_transform_edt
import multiprocessing
from multiprocessing import Pool, Process, Manager, freeze_support
import btrack
import imageio
from tifffile import imwrite
from skimage.transform import rotate, warp_polar, rescale
from skimage.color import label2rgb, gray2rgb
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from deepcell.applications import CellTracking
from skimage.feature import peak_local_max
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import adapted_rand_error, contingency_table, mean_squared_error, peak_signal_noise_ratio, variation_of_information
from skimage.filters import threshold_otsu, median
from skimage.morphology import disk, binary_closing, skeletonize, binary_opening, binary_erosion, white_tophat
from skimage.registration import phase_cross_correlation
from skimage.util import img_as_float
from scipy.cluster.hierarchy import linkage, fcluster
from hmmlearn import hmm
from scipy.spatial.distance import pdist, cdist, squareform
import spatial_efd
from skimage.segmentation import watershed, find_boundaries
from .Cell_Spot_Tracker import Tracking
import pkg_resources
from .logging_decorator import log_errors
import logging
from .Analysis import ImageAnalyzer

WELL_PLATE_ROWS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

class BatchAnalysis(object):
    """
    A class used to conduct batch analysis of biological image data, focusing on the analysis of cells and spots.

    Attributes
    ----------
        Various shared data lists and attributes for storing analysis results and configuration details.

    Methods
    -------
    __init__(self, Gui_Params, image_analyzer):
        Initializes the BatchAnalysis object with required parameters and data structures.

    ON_APPLYBUTTON(self, Meta_Data_df):
        Begins the batch analysis process, setting paths, reading metadata, and initiating various analyses.

    SAVE_NUCLEI_INFORMATION(self, cell_df, columns, rows):
        Saves analyzed information about nuclei into designated output files.

    SAVE_SPOT_INFO(self, spot_df, coordinates_method, columns, rows, channel_name):
        Saves the analyzed spot information for specific channels and methods.

    process_channel(self, channel_spot_df_list, columns, rows, channel_name):
        Processes each channel and saves spot information if necessary.

    PROCESS_ALL_SPOT_CHANNELS(self):
        Processes all spot channels and compiles their analyzed information.

    update_cell_index_in_channel(self, channel_attr, col, row, fov, time_point, previous_index, new_index):
        Updates cell indices in a specified channel's DataFrame.

    update_cell_index_in_all_spot_channels(self, col, row, fov, time_point, previous_index, new_index):
        Updates cell indices across all spot channels.

    normalize_stack(self, images):
        Normalizes a stack of images.

    cell_level_file_name(self, top_folder, folder_name, name_prefix, name_extention, col, row, fov, cell_ind, spot_ch=None):
        Generates file names for saving cell level data.

    spot_level_file_name(self, top_folder, folder_name, name_prefix, name_extention, col, row, fov, cell_ind, spot_ch, spot_ind):
        Generates file names for saving spot level data.

    BATCH_ANALYZER(self, col, row, fov, t): 
        Analyzes a batch of images for nuclei and spots, updating the results lists.

    Calculate_Spot_Distances(self, row, col):
        Calculates the distances between spots for given cells.

    DISTANCE_calculator(self, spot_pd_dict, key1, key2, select_cell, row, col):
        A helper function for calculating distances between spots.

    IMG_FOR_NUC_MASK(self):
        Loads the image used for creating nuclei masks.

    RAW_IMAGE_LOADER(self, maskchannel):
        Loads raw images for a specified channel.

    Z_STACK_NUC_SEGMENTER(self, ImageForNucMask):
        Segments nuclei from a z-stack of images.

    Z_STACK_NUC_LABLER(self, ImageForLabel):
        Labels nuclei in a z-stack of images.

    IMAGE_FOR_SPOT_DETECTION(self, ImageForNucMask):
        Prepares images for spot detection and retrieves coordinates.

    XYZ_SPOT_COORDINATES(self, images_pd_df, ImageForNucMask, spot_channel):
        Retrieves XYZ coordinates for spots in a specified channel.

    RADIAL_DIST_CALC(self, xyz_round, spot_nuc_labels, radial_dist_df, dist_img):
        Calculates radial distances for spots.
    """
    
    
    def __init__(self, params_dict = None):
        """
        Initializes the BatchAnalysis object with necessary parameters and structures for analysis.

        Parameters:

        Gui_Params : [type]
            Parameters from the GUI for controlling analysis settings.
        image_analyzer : [type]
            The image analyzer object to use for processing the images.

        Returns:

            None
        """
        
        self.ch1_spot_df, self.ch2_spot_df, self.ch3_spot_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.ch4_spot_df, self.ch5_spot_df, self.cell_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.df_checker = pd.DataFrame()
        self.Meta_Data_df = pd.DataFrame()
        self.spot_distances = {}
        self.experiment_name = []
        self.output_prefix = []
        self.output_folder = []
        self.params_dict = params_dict
        self.spot_cell_index = False


    def _process_jobs_nt(self, func_args):
        # This is the new helper method for processing jobs on Windows.
        args_length, _ = func_args.shape
        for ind1 in range(args_length):
            self.BATCH_ANALYZER(func_args[ind1, 0], func_args[ind1, 1], func_args[ind1, 2], func_args[ind1, 3])

    def _process_jobs_posix(self, func_args, jobs_number):
        # This is the new helper method for processing jobs on POSIX systems.
        arg_len, _ = func_args.shape
        start_pos = np.arange(0, arg_len, jobs_number)
        for st in start_pos:
            if st + jobs_number - 1 >= arg_len:
                data_ind = np.arange(st, arg_len)
            else:
                data_ind = np.arange(st, st + jobs_number)

            processes = []
            for ind1 in data_ind:
                process_args = np.array(func_args[ind1, :], dtype=int)
                processes.append(Process(target=self.BATCH_ANALYZER, args=(process_args[0], process_args[1], process_args[2], process_args[3])))
            
            for process in processes:
                process.start()
            for process in processes:
                process.join()

    def update_params_dict(self, params_dict):
        self.params_dict=params_dict
        
    @log_errors(logging.getLogger(__name__))            
    def ON_APPLYBUTTON(self, Meta_Data_df):
        """
        Begins the batch analysis process, including setting paths, reading metadata, and initiating analyses.

        Parameters:

        Meta_Data_df : DataFrame
            Metadata DataFrame containing information about the images to be analyzed.

        Returns:
            None
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.ERROR)
        
        seconds1 = time.time()
        
        # while self.params_dict["Output_dir"] ==[]:
                
        #     self.gui_params.OUTPUT_FOLDER_LOADBTN()
        self.Meta_Data_df = Meta_Data_df
        path_list = os.path.split(self.Meta_Data_df["ImageName"][0])[0].split(r'/')
        self.experiment_name = path_list[path_list.__len__()-2]
        self.output_prefix  = path_list[path_list.__len__()-1]
        self.output_folder = os.path.join(self.params_dict["Output_dir"],self.experiment_name)
        if os.path.isdir(self.output_folder) == False:
            os.mkdir(self.output_folder) 

        self.temp_dir =  os.path.join(self.output_folder,"temp")
        if os.path.isdir(self.temp_dir) == False:
            os.mkdir(self.temp_dir) 
        
        csv_config_folder = os.path.join(self.output_folder, 'configuration_files')
        if os.path.isdir(csv_config_folder) == False:
            os.mkdir(csv_config_folder) 
        self.config_file = os.path.join(csv_config_folder, 'analysis_configuration.csv')
        self.SAVE_CONFIGURATION(self.config_file, self.params_dict)
        
        ### error logs
        
        log_file = os.path.join(csv_config_folder, 'error_log.log')
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        ### save package info
        package_list = ['hitips', 'numpy', 'scikit-image', 'pandas', 'btrack', 'imageio', 'tifffile', 'aicsimageio', 'hmmlearn', 'scipy', 'spatial_efd', 'opencv-python-headless']
        versions = self.get_package_versions(package_list)
        versions['python'] = sys.version
        versions_df = pd.DataFrame(list(versions.items()), columns=['Package', 'Version'])
        versions_log_file = os.path.join(csv_config_folder, 'dependencies_versions.csv')
        versions_df.to_csv(versions_log_file, index=False)
        #####################################################################
        
        columns = np.unique(np.asarray(self.Meta_Data_df['column'], dtype=int))
        rows = np.unique(np.asarray(self.Meta_Data_df['row'], dtype=int))
        fovs = np.unique(np.asarray(self.Meta_Data_df['field_index'], dtype=int))

        time_points = np.unique(np.asarray(self.Meta_Data_df['time_point'], dtype=int))
        actionindices = np.unique(np.asarray(self.Meta_Data_df['action_index'], dtype=int))
        
        jobs_number=self.params_dict['NumCPUsSpinBox_value']

        func_args=np.zeros((0,4),dtype=int)
        for t in time_points:
            for fov in fovs:
                for row in rows:
                    for col in columns:
                        df_parallel = self.Meta_Data_df.loc[(self.Meta_Data_df['column'] == str(col)) & 
                                                              (self.Meta_Data_df['row'] == str(row)) & 
                                                              (self.Meta_Data_df['field_index'] == str(fov)) & 
                                                              (self.Meta_Data_df['time_point'] == str(t))]
                        if df_parallel.empty == False:

                            func_args=np.append(func_args,np.array([col,row,fov,t]).reshape(1,4),axis=0)
        if (os.name == 'nt'):
            # Windows does not fork, so it's safe to use the multiprocessing directly.
            self._process_jobs_posix(func_args, jobs_number)
        else:
            # For POSIX systems, ensure safe multiprocessing context
            self._process_jobs_posix(func_args, jobs_number)


        if self.params_dict['NucInfoChkBox_check_status'] == True:

            pickle_files = [f for f in os.listdir(self.temp_dir) if 'temp_cell_df' in f and f.endswith('.pickle')]
            
            dfs = []
            for file in pickle_files:
                  dfs.append(pd.read_pickle(os.path.join(self.temp_dir, file)))
            self.cell_df = pd.concat(dfs, ignore_index=True)
            self.SAVE_NUCLEI_INFORMATION(self.cell_df, columns, rows)
            
        self.PROCESS_ALL_SPOT_CHANNELS()
            # Calculate Spot Distances
    
        if self.params_dict['SpotsDistance_check_status'] == True:
            columns = np.unique(np.asarray(self.cell_df['column'], dtype=int))
            rows = np.unique(np.asarray(self.cell_df['row'], dtype=int))
            Parallel(n_jobs=jobs_number, prefer="threads")(delayed(self.Calculate_Spot_Distances)(row, col) for row in rows for col in columns)
     
        if self.params_dict['Cell_Tracking_check_status'] == True:
            self.ImageAnalyzer = ImageAnalyzer(self.params_dict)
            xlsx_name = ['Nuclei_Information.csv']
            xlsx_full_name = os.path.join(os.path.join(self.output_folder,"whole_plate_resutls"), xlsx_name[0])
            self.cell_df = pd.read_csv(xlsx_full_name).drop(["Unnamed: 0"], axis=1)
            for spot_file in os.listdir(os.path.join(self.output_folder,"whole_plate_resutls")):
                for i in range(5):
                    if 'Ch' + str(i+1) + '_Spot_Locations_' in spot_file:
                        setattr(self, 'ch' + str(i+1) + '_spot_df', 
                                pd.read_csv(os.path.join(self.output_folder,"whole_plate_resutls",spot_file)).drop(["Unnamed: 0"], axis=1)) 
                        
            cell_tracking_folder = os.path.join(self.output_folder, 'cell_tracking')
            if os.path.isdir(cell_tracking_folder) == False:
                os.mkdir(cell_tracking_folder)

            #### get all the unique channels containing spots and load all the images related to the spots
            columns = np.unique(np.asarray(self.cell_df['column'], dtype=int))
            rows = np.unique(np.asarray(self.cell_df['row'], dtype=int))
            cr_arr=[]
            for row in rows:
                for col in columns:
                    subcell_df=self.cell_df.loc[(self.cell_df['column']==col)&(self.cell_df['row']==row)]
                    if subcell_df.empty==False:
                        cr_arr.append([row,col])
                        
            fovs = np.unique(np.asarray(self.Meta_Data_df['field_index'], dtype=int))
            time_points = np.unique(np.asarray(self.Meta_Data_df['time_point'], dtype=int))
            actionindices = np.unique(np.asarray(self.Meta_Data_df['action_index'], dtype=int))
            
            for cr_elem in cr_arr:
                    row=int(cr_elem[0])
                    col=int(cr_elem[1])
                    #### load and concatenate all the spot information csv files for all the wells and fields. 
                    #### For every field the csv file for spots in each channel will be loaded and concatenated 
                    spots_loc = pd.DataFrame(columns=["column", "row", "field_index",'time_point', "channel", 'x_location', 'y_location'])
                    spot_loc_substring = '_well_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                    well_spots_output_folder = os.path.join(self.output_folder, 'well_spots_locations')
                    if os.path.exists(well_spots_output_folder):
                        for fname in os.listdir(well_spots_output_folder):    # change directory as needed
                            if spot_loc_substring in fname:    # search for string
                                spots_loc = pd.concat([spots_loc,pd.read_csv(os.path.join(well_spots_output_folder,fname))])                        
                    ### get all the channels containing the spots
                    if spots_loc.empty:
                        spot_channels=np.array([])
                    else:
                        spot_channels = spots_loc['channel'].unique()
        
                    for fov in fovs:
                        
                        spot_images,spot_dict_stack = {}, {}
                        for chnl in spot_channels:
                            spot_images[str(int(chnl))] = []
                            
                        masks, lbl_imgs, nuc_imgs=[], [], []
                        for t in time_points:
                            self.df_checker = self.Meta_Data_df.loc[(self.Meta_Data_df['column'] == str(col)) & 
                                                                    (self.Meta_Data_df['row'] == str(row)) & 
                                                                    (self.Meta_Data_df['field_index'] == str(fov)) & 
                                                                    (self.Meta_Data_df['time_point'] == str(t))]
                            if self.df_checker.empty:
                                continue
        
                            ImageForNucMask = self.RAW_IMAGE_LOADER(str(self.params_dict['NucleiChannel'][-1]))
                            normalized_nuc_img = cv2.normalize(ImageForNucMask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            nuc_imgs.append(normalized_nuc_img)
                
                            for chnl in spot_channels:
                                spot_images[str(int(chnl))].append(self.RAW_IMAGE_LOADER(str(int(chnl))))
                            
                            mask_file_name = ['Nuclei_Mask_for_Col' + str(col) + r'_row' + str(row)+
                                          r'_Time' + str(t) + r'_Field' + str(fov) + r'.tif']
                            
                            nuc_mask_output_folder = os.path.join(self.output_folder, 'nuclei_masks')
                            mask_full_name = os.path.join(nuc_mask_output_folder, mask_file_name[0])
                            mask_image = imread(mask_full_name, 0)
                            lbl, ncc = label(mask_image)
                            masks.append(mask_image)
                            lbl_imgs.append(lbl)
                        label_stack=np.stack(lbl_imgs,axis=0)
                        masks_stack=np.stack(masks,axis=0)
                        t_stack_nuc=np.stack(nuc_imgs,axis=0)
                        for chnl in spot_channels:
                            spot_dict_stack[str(int(chnl))] = np.stack(spot_images[str(int(chnl))],axis=2)
                        
                        #### select spots related to the current field only 
                        selected_spots = spots_loc.loc[(spots_loc['field_index'] == fov)&(spots_loc['column'] == col)&(spots_loc['row'] == row)]
                        #### get all the unique channels contating spots 
                        
                        
                        if selected_spots.empty:
                            unique_ch=np.array([])
                        else:
                            unique_ch = selected_spots['channel'].unique()
                            
                        #### run nuclei tracking algorithm based the selected method
                        if self.params_dict['NucTrackingMethod_currentText'] == "Bayesian":
                            
                            tracks_pd = Tracking.RUN_BTRACK(label_stack, self.params_dict)
                        
                        if self.params_dict['NucTrackingMethod_currentText'] == "DeepCell":
                        
                            tracks_pd = Tracking.deepcell_tracking(t_stack_nuc,label_stack,self.params_dict)
                        ###
                        tracks_pd_copy = tracks_pd.copy()
                        tracks_pd_copy=tracks_pd_copy.rename(columns={"t": "time_point", "x": "centroid-1", "y": "centroid-0", "ID": "cell_index"})
                        tracks_pd_copy["time_point"] = tracks_pd_copy["time_point"] + 1
                        tracks_pd_copy = tracks_pd_copy.astype({'centroid-0':'int', 'centroid-1':'int'})
                        self.cell_df =  self.cell_df.astype({'centroid-0':'int', 'centroid-1':'int', 'time_point':'int', 'cell_index':'int'})
                        field_cell_df = self.cell_df.loc[(self.cell_df['field_index'] == fov)&(self.cell_df['column'] == col)&(self.cell_df['row'] == row)]
                        for trck_ind, trck_row in tracks_pd_copy.iterrows():

                            row_index_celldf = field_cell_df.loc[(field_cell_df["time_point"]==trck_row["time_point"])&(field_cell_df["centroid-0"]==trck_row["centroid-0"])&
                                                          (field_cell_df["centroid-1"]==trck_row["centroid-1"])].index[0]
                            
                            # Update 'cell_index' in self.cell_df
                            previous_index = self.cell_df.loc[row_index_celldf, 'cell_index']
                            self.cell_df.loc[row_index_celldf, 'cell_index'] = trck_row["cell_index"]
                            self.update_cell_index_in_all_spot_channels( col, row, fov, trck_row["time_point"], previous_index, trck_row["cell_index"])
                        
                            
                        ##########save whole field track images
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = int(1)
                        fontColor = (255,255,255)
                        lineType  = int(2)
                        t,ww,hh=t_stack_nuc.shape
                        rgb_t_stack = np.zeros((t,1,3,ww,hh), dtype=t_stack_nuc.dtype)
                        for i in tracks_pd['t'].unique():

                            All_Channels = gray2rgb(t_stack_nuc[i,:,:])
                            time_tracks = tracks_pd.loc[tracks_pd['t']==i]
                            for ind, pd_row in time_tracks.iterrows():

                                txt=str(pd_row["ID"])
                                bottomLeftCornerOfText = (int(round(pd_row["x"])), int(round(pd_row["y"])))
                                cv2.putText(All_Channels,txt, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

                            rgb_t_stack[i,:,:,:,:] = np.rollaxis(All_Channels, 2, 0)  
                        
                        Raw_file_name = 'track_image_col' + str(col) + r'_row' + str(row)+ r'_field' + str(fov) + r'.ome.tif'
                        whole_field_track_folder = os.path.join(cell_tracking_folder, 'whole_field_track_images')
                        if os.path.isdir(whole_field_track_folder) == False:
                            os.mkdir(whole_field_track_folder)
                        whole_field_full_name = os.path.join(whole_field_track_folder, Raw_file_name)
                        writer = OmeTiffWriter()
                        writer.save(rgb_t_stack[:,0,0,:,:],whole_field_full_name, dimension_order="TYX")

#                         writer.save(spot_dict_stack[str(int(chnl))],whole_field_full_name, dimension_order="TYX")
                        print("field number: " , fov)
                        #####################
                        patch_size = (self.params_dict['patchsize_currentText'],self.params_dict['patchsize_currentText'])
                        IDs = tracks_pd['ID'].unique()

                        for id_ in IDs:
                            #### create a copy of each single track df containing all the nuclei informaion
                            single_track1 = tracks_pd.loc[tracks_pd['ID']==id_]
                            single_track = single_track1[single_track1['dummy']==False]
                            single_track_copy = single_track.copy().reset_index(drop=True)
                            
                            if len(single_track_copy)< self.params_dict['MintrackLengthSpinbox_current_value']:
                                 continue
                            
                            for chnl in unique_ch:
                                col_name1='ch'+str(int(chnl))+'_spots_number'
                                single_track_copy[col_name1]=np.zeros(len(single_track),dtype=int)
                                col_name2='ch'+str(int(chnl))+'_spots_locations'
                                single_track_copy[col_name2]=[[]]*len(single_track)
                                col_name3='ch'+str(int(chnl))+'_patch_spots_locations'
                                single_track_copy[col_name3]=[[]]*len(single_track)
                                col_name3='ch'+str(int(chnl))+'_transformed_spots_locations'
                                single_track_copy[col_name3]=[[]]*len(single_track)
                                col_name3='ch'+str(int(chnl))+'_integrated_intensity'
                                single_track_copy[col_name3]=[[]]*len(single_track)
                            
                            nuc_patches, rotated_nuc,mask_patches, rot_masks = [],[],[], []
                            spot_coor_dict, spot_patches, rotated_spot_patches = {},{},{}
                            transformed_spots ={}

                            spot_channels = selected_spots['channel'].unique().astype(int)
                            for chnl in spot_channels:
                                spot_patches[str(int(chnl))] = []
                                rotated_spot_patches[str(int(chnl))] = []
                            
                            for ind, pd_row in single_track_copy.iterrows():
                                time_spots = selected_spots[selected_spots['time_point']==pd_row['t']]
                                patch_spots = time_spots[(time_spots['x_location'] > int(pd_row['bbox-0'])) & 
                                                         (time_spots['x_location'] < int(pd_row['bbox-2'])) & 
                                                         (time_spots['y_location'] > int(pd_row['bbox-1'])) & 
                                                         (time_spots['y_location'] < int(pd_row['bbox-3'])) ]
                                pix_margin = 10
                                min_row = max(int(pd_row['bbox-0'])-pix_margin, 0)
                                max_row = min(int(pd_row['bbox-2'])+pix_margin, ww)
                                min_col = max(int(pd_row['bbox-1'])-pix_margin, 0)
                                max_col = min(int(pd_row['bbox-3'])+pix_margin, hh)
                                
                                small_img_patch = t_stack_nuc[int(pd_row['t']), min_row:max_row, min_col:max_col]
        
                                img_patch = Tracking.zero_pad_patch(small_img_patch, desired_size=patch_size)
                                
                                bin_img = Tracking.zero_pad_patch(masks_stack[int(pd_row['t']), min_row:max_row, min_col:max_col], desired_size=patch_size)
                                mask_patches.append(bin_img)
                                lbl_img, n_feat = label(bin_img)
                                label_center = int(self.params_dict['patchsize_currentText']/2)
                                lbl_img[lbl_img!=lbl_img[label_center,label_center]]=0
                                bin_img = lbl_img>0

                                bin_img = ndimage.binary_dilation(bin_img, structure=np.ones((12,12))>0).astype(bin_img.dtype)


                                img_patch[bin_img==0]=0
                                row_shift, col_shift = ndimage.center_of_mass(bin_img)
                                if (math.isnan(row_shift) or math.isnan(col_shift)):
                                    final_translation = np.array([0, 0])
                                else:
                                    final_translation = np.array([label_center-round(row_shift), label_center-round(col_shift)])

                                img_patch1 = ndimage.shift(img_patch, final_translation, order=0)
                                bin_img = ndimage.shift(bin_img, final_translation, order=0)

                                ##### this part finds the best rotation orientation for aligned nuclei to prevent flipping 
#                                 angle_1 = (-180*pd_row['orientation']/np.pi)+90

#                                 im1_rot = rotate(img_patch, angle_1, preserve_range=True)
                                nuc_patches.append(img_patch1)
                                if len(rotated_nuc)==0:
                                    
                                    rotated_nuc.append(img_patch1)
                                    rot_masks.append(bin_img)
                                    final_angle = 0
                                
                                elif len(rotated_nuc) > 0:
                                    if self.params_dict['Registrationmethod_currentText'] == "Phase Correlation":
                                        init_rotation = 15
                                        final_angle, rotated_img = Tracking.phase_correlation(rotated_nuc[-1], img_patch1, intial_rotation = 15, 
                                                                                          rescale_factor = 5, nuc_length = label_center-10)
                                        

                                    rotated_nuc.append(rotated_img)
                                
                                only_spots_patch={}
                                coordinates={}
                                for chnl in unique_ch:

                                    only_spots_patch[str(int(chnl))] = np.zeros(small_img_patch.shape)
                                    patch_ch_spots = patch_spots.loc[patch_spots['channel']==int(chnl)]
                                    col_name1='ch'+str(int(chnl))+'_spots_number'
                                    single_track_copy.loc[single_track_copy['t']==pd_row['t'], col_name1] = len(patch_ch_spots)
                                    col_name2='ch'+str(int(chnl))+'_spots_locations'
                                    if len(patch_ch_spots[['x_location','y_location','z_location']]<self.params_dict['maxspotspercellSpinbox_current_value']+1):
                                        coordinates[str(int(chnl))] = patch_ch_spots[['x_location','y_location','z_location']].to_numpy()
                                        single_track_copy.loc[single_track_copy['t']==pd_row['t'], col_name2] = [patch_ch_spots[['x_location','y_location','z_location']].to_numpy()]
                                        col_name3='ch'+str(int(chnl))+'_integrated_intensity'
                                        single_track_copy.loc[single_track_copy['t']==pd_row['t'], col_name3] = [[patch_ch_spots["integrated_intensity"].to_numpy()]]
                                    
                                    else:
                                        coordinates[str(int(chnl))] = np.zeros((0,3))
                                        single_track_copy.loc[single_track_copy['t']==pd_row['t'], col_name2] = [[np.array([])]]
                                        col_name3='ch'+str(int(chnl))+'_integrated_intensity'
                                        single_track_copy.loc[single_track_copy['t']==pd_row['t'], col_name3] = [[np.array([])]]
                                    
                                    patch_rel_coordinates={}
                                    if list(coordinates[str(int(chnl))]):
                                        patch_rel_coordinates[str(int(chnl))] = coordinates[str(int(chnl))][:,:2]-np.array([min_row, min_col])+final_translation


                                        coor_for_cir = np.round(patch_rel_coordinates[str(int(chnl))].astype(np.double)).astype(int)
                                        try:
                                            only_spots_patch[str(int(chnl))][coor_for_cir[:,0],coor_for_cir[:,1]] = 255
                                        except:
                                            pass


                                    only_spots_patch[str(int(chnl))] = Tracking.zero_pad_patch(only_spots_patch[str(int(chnl))], 
                                                                                           desired_size=patch_size)

                                    col_name3='ch'+str(int(chnl))+'_patch_spots_locations'
                                    single_track_copy.loc[single_track_copy['t']==pd_row['t'], col_name3]=[np.where(only_spots_patch[str(int(chnl))]>0)]

                                    spot_patch = Tracking.zero_pad_patch(spot_dict_stack[str(int(chnl))][ min_row:max_row, min_col:max_col, int(pd_row['t'])], 
                                                                     desired_size=patch_size)
                                    spot_patches[str(int(chnl))].append(spot_patch)

                                    temp_spots = []
                                    temp_transformed = []
                                    for i in range(len(np.where(only_spots_patch[str(int(chnl))]>0)[0])):

                                        temp_spots.append(np.array([np.where(only_spots_patch[str(int(chnl))]>0)[0][i], 
                                                                    np.where(only_spots_patch[str(int(chnl))]>0)[1][i]]))
                                        temp_transformed.append(Tracking.rotate_point((label_center-0.5,label_center-0.5),
                                                                             np.array([np.where(only_spots_patch[str(int(chnl))]>0)[0][i], 
                                                                                       np.where(only_spots_patch[str(int(chnl))]>0)[1][i]]), 
                                                                             (final_angle+15)*np.pi/180))

                                    if len(temp_spots)>0:
                                        transformed_spots[str(pd_row['t'])] = np.round(np.array(temp_transformed)).astype(int)
                                        spot_coor_dict[str(pd_row['t'])]= np.array(temp_spots)
                                        col_name4='ch'+str(int(chnl))+'_transformed_spots_locations'
                                        single_track_copy.loc[single_track_copy['t']==pd_row['t'], col_name4]=[transformed_spots[str(pd_row['t'])]]
            
                                    # only_spots_patch[str(int(chnl))] = Tracking.zero_pad_patch(only_spots_patch[str(int(chnl))])
                                    # spot_patch = cv2.normalize(spot_patch, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                    
                                    spot_patch = ndimage.shift(spot_patch, final_translation, order=0).astype(spot_dict_stack[str(int(chnl))].dtype)
                                    rot_spot_patch = rotate(rotate(spot_patch, 15, preserve_range=True, order=0), 
                                                            final_angle, preserve_range=True, order=0).astype(spot_dict_stack[str(int(chnl))].dtype)
                                    
                                    
                                    rotated_spot_patches[str(int(chnl))].append(rot_spot_patch)
                                    
                                     

                            large_rotcell_stack = np.stack(rotated_nuc, axis=2)
                            large_cell_stack = np.stack(nuc_patches, axis=2)
                            large_rot_spot_img_stack={}
                            for chnl in spot_channels:
                                large_rot_spot_img_stack[str(int(chnl))] = np.stack(rotated_spot_patches[str(int(chnl))], axis=2).astype(spot_dict_stack[str(int(chnl))].dtype)

                            for chnl in spot_channels:

                                anotated_large_rot_spot_img_stack = large_rot_spot_img_stack[str(int(chnl))].copy()
                                unanotated_large_rot_spot_img_stack = large_rot_spot_img_stack[str(int(chnl))].copy()
                                raw_large_rot_spot_img_stack = large_rot_spot_img_stack[str(int(chnl))].copy()
                                spot_df_list = single_track_copy.loc[single_track_copy["ch"+str(chnl)+"_spots_number"]>0]["ch"+str(chnl)+"_transformed_spots_locations"].tolist()
                                all_spots_for_gmm = np.round(np.array([item for sublist in spot_df_list for item in sublist])).astype(int)
                                if len(all_spots_for_gmm)>3:
                                    
                                    try:
                                        clusters  = Tracking.run_clustering(all_spots_for_gmm, outlier_threshold = 3, max_dist = self.params_dict['SpotSearchRadiusSpinbox_current_value'],
                                                                        min_burst_duration = self.params_dict['minburstdurationSpinbox_current_value'])
                                    except:

                                        clusters=[]

                                    for lbl1 in np.unique(clusters):
                                        if lbl1>0:
                                            single_track_copy['ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_locations"] = [[]]*len(single_track)
                                            single_track_copy['ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_integrated_intensity"] = 0
                                    if len(clusters) > 0:   
                                        for trk_ind in single_track_copy.index.tolist(): 
                                            timepoint_spots = list(single_track_copy.loc[trk_ind, "ch"+str(chnl)+"_transformed_spots_locations"])
                                            integrated_intensities = single_track_copy.loc[trk_ind, "ch"+str(chnl)+"_integrated_intensity"][0]
                                            if len(timepoint_spots)>0:

                                                for spot_loc in timepoint_spots:
                                                    if clusters[np.all(spot_loc == all_spots_for_gmm, axis=1)][0]>0:
                                                        col_name = 'ch'+str(int(chnl))+'_spot_no_'+str(clusters[np.all(spot_loc == all_spots_for_gmm, axis=1)][0])+"_locations"
                                                        single_track_copy.loc[trk_ind, col_name] = [[spot_loc]]
                                                        # col_name = 'ch'+str(int(chnl))+'_spot_no_'+str(clusters[np.all(spot_loc == all_spots_for_gmm, axis=1)][0])+"_integrated_intensity"
                                                        # single_track_copy.loc[trk_ind, col_name] = integrated_intensities[next((i for i, spot in enumerate(timepoint_spots) if np.array_equal(spot, spot_loc)), -1)]
                                    
                                    for lbl1 in np.unique(clusters)[np.unique(clusters)>0]:
                                        sp_bound= int(round(7*self.params_dict['PSFsizeSpinBox_value']/2))
                                        all_spot_patches, spot_patches_center_coords = Tracking.get_spot_patch(single_track_copy, chnl, lbl1, 
                                                                                                           rotated_spot_patches[str(int(chnl))], 
                                                                                                           spot_boundary = sp_bound)
                                        for jjj in single_track_copy.index.tolist():

                                            if jjj in list(all_spot_patches.keys()):
                                                spot_patch = all_spot_patches[jjj]
                                                spot_coords = spot_patches_center_coords[jjj]

                                                fit_results = self.ImageAnalyzer.gmask_fit(spot_patch, 
                                                                                           xy_input=np.array([np.where(spot_patch==spot_patch.max())[0][0],
                                                                                                              np.where(spot_patch==spot_patch.max())[1][0]]), 
                                                                                           fit=self.params_dict['IntegratedIntensity_fitStatus']>0)
                                                single_track_copy.loc[jjj,'ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_locations"] = [[np.array([spot_coords[0,0]-sp_bound+fit_results[0],
                                                                                                                                                spot_coords[0,1]-sp_bound+fit_results[1]])]]
                                                single_track_copy.loc[jjj,'ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_integrated_intensity"] = fit_results[2]
                                    
                            
                                    for lbl1 in np.unique(clusters)[np.unique(clusters)>0]: 
                                        row_col_name = 'ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_x"
                                        col_col_name = 'ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_y"
                                        try:
                                            single_track_copy[row_col_name]=single_track_copy['ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_locations"].apply(lambda coord_list: coord_list[0][0][0])
                                            single_track_copy[col_col_name]=single_track_copy['ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_locations"].apply(lambda coord_list: coord_list[0][0][1])
                                        except:
                                            single_track_copy[row_col_name]=single_track_copy['ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_locations"].apply(lambda coord_list: coord_list[0][0])
                                            single_track_copy[col_col_name]=single_track_copy['ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_locations"].apply(lambda coord_list: coord_list[0][1])
                                        single_track_copy["column"]=col
                                        single_track_copy["row"] = row
                                        single_track_copy["field_index"] = fov
                                        single_track_copy["channel"] = int(chnl)
                                        spot_signal_df = single_track_copy[["column", "row", "field_index", "channel", "t", row_col_name, 
                                                                            col_col_name, 'ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_integrated_intensity"]]


                                        rna_signal=single_track_copy['ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_integrated_intensity"].to_numpy().reshape(-1,1)
                                        n_samples = len(single_track_copy)
                                        signal = (rna_signal-rna_signal.mean())/rna_signal.std()
                                        
                                        if self.params_dict['FittingMethod_index']==0:
                                            model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
                                            # Initialize the means in ascending order
                                            sorted_idx = np.argsort([np.mean(signal[:n_samples // 2]), np.mean(signal[n_samples // 2:])])
                                            model.means_ = np.array([np.mean(signal[:n_samples // 2]), np.mean(signal[n_samples // 2:])])[sorted_idx].reshape(-1, 1)
                                            try:
                                            # Create and fit a Gaussian HMM with 2 states
                                                model.fit(signal)

                                                # Get the most likely hidden state sequence for the signal
                                                hidden_states = model.predict(signal)
                                            except:
                                                hidden_states = np.zeros(len(single_track_copy), dtype=float)
                                                
                                        elif self.params_dict['FittingMethod_index']==1:
                                            
                                            model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)

                                            # Divide the signal into three equal parts
                                            part1 = signal[:n_samples // 3]
                                            part2 = signal[n_samples // 3 : 2 * n_samples // 3]
                                            part3 = signal[2 * n_samples // 3 :]

                                            # Calculate the mean of each part and sort the indices
                                            means = np.array([np.mean(part1), np.mean(part2), np.mean(part3)])
                                            sorted_idx = np.argsort(means)

                                            # Initialize the means in ascending order
                                            model.means_ = means[sorted_idx].reshape(-1, 1)

                                            try:
                                                # Create and fit a Gaussian HMM with 3 states
                                                model.fit(signal)

                                                # Get the most likely hidden state sequence for the signal
                                                hidden_states = model.predict(signal)
                                            except:
                                                hidden_states = np.zeros(len(single_track_copy), dtype=float)
                                                
                                        single_track_copy.loc[:, 'ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_HMM_state"] = hidden_states
                                        
                                       
                                        
                                        
                                        spot_intensity_tables = os.path.join(cell_tracking_folder, 'spot_intensity_tables')
                                        if os.path.isdir(spot_intensity_tables) == False:
                                            os.mkdir(spot_intensity_tables)
                                        ######################## csv tables
                                        spot_signal_df.to_csv(self.spot_level_file_name(spot_intensity_tables, 'complete_tables', 'spot_intensity', '.csv', col, row, fov, id_,chnl, lbl1))
                                        ######################## max intensity tables
                                        sub_spot_df = single_track_copy[[row_col_name,
                                                                         col_col_name, 
                                                                         'ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_integrated_intensity", "t",
                                                                         'ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_HMM_state"]].copy()
                                        sub_spot_df.to_csv(self.spot_level_file_name(spot_intensity_tables, 'integrated_intensity_tables', 
                                                                                     'integrated_intensity', '.trk', col, row, fov, id_,chnl, lbl1),
                                                                                     sep='\t', index=False, header=False)
                                       
                                        ####################
                                        # Load the font
                                        current_dir = os.path.dirname(os.path.abspath(__file__))
                                        font_path = os.path.join(current_dir,'Roboto-Bold.ttf')

                                        font_size = 10  # You can adjust the font size based on your requirements
                                        font = ImageFont.truetype(font_path, font_size)
                                        anotated_large_rot_spot_img_stack = self.normalize_stack(anotated_large_rot_spot_img_stack)
                                        unanotated_large_rot_spot_img_stack = self.normalize_stack(unanotated_large_rot_spot_img_stack)
                                        
                                        for st_ind in range(anotated_large_rot_spot_img_stack.shape[2]):

                                            temp_img = anotated_large_rot_spot_img_stack[:, :,st_ind]
                                            temp_img_unannotated = unanotated_large_rot_spot_img_stack[:, :,st_ind]
                                            # Get the circle's coordinates
                                            circle_coordinates = spot_signal_df[[row_col_name, col_col_name]].to_numpy().astype(int)[st_ind].reshape(1,2)
                                            
                                            circ_for_draw = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(circle_coordinates,
                                                                                                     temp_img,circ_radius = 5)
                                            # Convert your numpy image to a PIL image so you can use ImageDraw
                                            pil_img = Image.fromarray(temp_img.astype('uint8'))
                                            draw = ImageDraw.Draw(pil_img)
                                            
                                            
                                            # Assuming circle_coordinates is in the format [[x, y]]
                                            circle_x, circle_y = circle_coordinates[0]

                                            # Adjust text position to be above the circle (may need fine-tuning)
                                            text_x = circle_x - 5  # Adjusting horizontally to roughly center the text, this might need some tweaking based on font size and circle size.
                                            text_y = circle_y - font_size - 5  # Position the text right above the circle.

                                            # Add text
                                            draw.text((text_y, text_x), str(lbl1), font=font, fill=255)  # You can adjust fill color if needed

                                            # Convert back to numpy
                                            temp_img = np.array(pil_img)
                                            
                                            temp_img_unannotated[circ_for_draw != 0] = 255
                                            temp_img[circ_for_draw != 0] = 255
                                            anotated_large_rot_spot_img_stack[:, :,st_ind]=temp_img
                                            unanotated_large_rot_spot_img_stack[:, :,st_ind]=temp_img_unannotated
                                    ########## save annotated and unannotated spot patches
                                    imwrite(self.cell_level_file_name(cell_tracking_folder, 'annotated_spot_image_patches', 'spot_img', '.tif', col, row, fov, id_,chnl), 
                                            np.rollaxis(anotated_large_rot_spot_img_stack, 2, 0), imagej=True, metadata={'axes': 'TYX'})    
                                    
                                    imwrite(self.cell_level_file_name(cell_tracking_folder, 'spot_image_patches', 'unannotated_spot_img', '.tif', col, row, fov, id_,chnl),
                                             np.rollaxis(unanotated_large_rot_spot_img_stack, 2, 0), imagej=True, metadata={'axes': 'TYX'})
                                    imwrite(self.cell_level_file_name(cell_tracking_folder, 'unannotated_spot_image_patches', 'raw_spot_img', '.tif', col, row, fov, id_,chnl),
                                            np.rollaxis(raw_large_rot_spot_img_stack, 2, 0))
                                    
                                    
                            large_cell_stack = self.normalize_stack(large_cell_stack)
                            imwrite(self.cell_level_file_name(cell_tracking_folder, 'single_track_images', 'nuclei_image', '.tif', col, row, fov, id_),
                                    np.rollaxis(large_cell_stack, 2, 0), imagej=True, metadata={'axes': 'TYX'})
                            large_rotcell_stack = self.normalize_stack(large_rotcell_stack)
                            imwrite(self.cell_level_file_name(cell_tracking_folder, 'single_track_images', 'aligned_nuclei_image', '.tif', col, row, fov, id_),
                                    np.rollaxis(large_rotcell_stack, 2, 0), imagej=True, metadata={'axes': 'TYX'})
                            
                            single_track_copy.to_csv(self.cell_level_file_name(cell_tracking_folder, 'single_track_tables', 'track_table', '.csv', col, row, fov, id_))  
                            print("saved track: "+ 'col' + str(col) + r'_row' + str(row) + r'_field' + str(fov) + r'_cell' + str(id_))
                            
        self.SAVE_NUCLEI_INFORMATION(self.cell_df, columns, rows)
        self.spot_cell_index = True
        self.PROCESS_ALL_SPOT_CHANNELS()
        try:
            shutil.rmtree(self.temp_dir, onerror=self.remove_readonly)
        except Exception as e:
            print(f"Error removing {self.temp_dir}: {e}")
        seconds2 = time.time()
        
        diff=seconds2-seconds1
        print('Total Processing Time (Minutes):',diff/60)
    
    def remove_readonly(self, func, path, _):
        "Clear the readonly bit and reattempt the removal"
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def get_package_versions(self, packages):
        versions = {}
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
            except pkg_resources.DistributionNotFound:
                version = 'Not Installed'
            versions[package] = version
        return versions
    def SAVE_NUCLEI_INFORMATION(self, cell_df, columns, rows):
        """
        Saves analyzed information about nuclei into designated output files.

        Parameters:
            cell_df : DataFrame
                DataFrame containing the analyzed nuclei information.
            columns : list
                List of column indices in the imaging plate.
            rows : list
                List of row indices in the imaging plate.

        Returns:
            None
        """
        xlsx_output_folder = os.path.join(self.output_folder, 'whole_plate_resutls')
        if os.path.isdir(xlsx_output_folder) == False:
            os.mkdir(xlsx_output_folder) 

        xlsx_name = ['Nuclei_Information.csv']
        xlsx_full_name = os.path.join(xlsx_output_folder, xlsx_name[0])
        cell_df.rename(columns={ "label":"cell_index"}, inplace = True)
        cell_df.to_csv(xlsx_full_name)

        well_nuc_folder = os.path.join(self.output_folder, 'well_nuclei_results')
        if os.path.isdir(well_nuc_folder) == False:
            os.mkdir(well_nuc_folder)
        for col in columns:
            for row in rows:

                well_nuc_df = cell_df.loc[(cell_df['column'] == col) & (cell_df['row'] == row)]
                if well_nuc_df.empty == False:
                    well_nuc_filename = self.output_prefix + '_nuclei_information_well_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                    nuc_well_csv_full_name = os.path.join(well_nuc_folder, well_nuc_filename)
                    well_nuc_df.to_csv(path_or_buf=nuc_well_csv_full_name, encoding='utf8')
        print("Nuclei Information Saved....")
        
    def SAVE_SPOT_INFO(self, spot_df, coordinates_method, columns, rows, channel_name ):
        
        """
        Saves the analyzed spot information for specific channels and methods.

        Parameters:
        -----------
        spot_df : DataFrame
            DataFrame containing the analyzed spot information.
        coordinates_method : str
            The method used for determining the coordinates of spots.
        columns : list
            List of column indices in the imaging plate.
        rows : list
            List of row indices in the imaging plate.
        channel_name : str
            Name of the channel for which the spot information is being saved.

        Returns:
        --------
        None
        """
        
        xlsx_output_folder = os.path.join(self.output_folder, 'whole_plate_resutls')
        if os.path.isdir(xlsx_output_folder) == False:
            os.mkdir(xlsx_output_folder)
        xlsx_name = channel_name + '_Spot_Locations_' + coordinates_method + r'.csv'
        xlsx_full_name = os.path.join(xlsx_output_folder, xlsx_name)
        spot_df.to_csv(xlsx_full_name)

        well_spot_loc_folder = os.path.join(self.output_folder, 'well_spots_locations')
        if os.path.isdir(well_spot_loc_folder) == False:
            os.mkdir(well_spot_loc_folder)
        for col in columns:
            for row in rows:
                spot_loc_df = spot_df.loc[(spot_df['column'] == col) & (spot_df['row'] == row)]
                if spot_loc_df.empty == False:
                    spot_loc_filename = self.output_prefix + '_' + channel_name +'_spots_locations_well_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                    spot_loc_well_csv_full_name = os.path.join(well_spot_loc_folder, spot_loc_filename)
                    spot_loc_df.to_csv(path_or_buf=spot_loc_well_csv_full_name, encoding='utf8')   
                    
    def process_channel(self, channel_spot_df_list, columns, rows, channel_name):
        """
        Processes each channel and saves spot information if necessary.

        Parameters:
        -----------
        channel_spot_df_list : list
            A list of DataFrames, each containing spot information for a specific channel.
        columns : list
            List of column indices in the imaging plate.
        rows : list
            List of row indices in the imaging plate.
        channel_name : str
            Name of the channel being processed.

        Returns:
        --------
        None
        """
        if len(channel_spot_df_list) > 0:
            if self.spot_cell_index:
                channel_spot_df = channel_spot_df_list
            else:
                channel_spot_df = pd.concat(channel_spot_df_list)
            if self.params_dict['SpotsLocation_check_status']:
                self.SAVE_SPOT_INFO(channel_spot_df, self.params_dict['SpotLocationCbox_currentText'], columns, rows, channel_name)
                
    def gather_spot_df_list(self,channel_name):
        substring = "_channel_" + str(channel_name)
        pickle_files = [f for f in os.listdir(self.temp_dir) if substring in f and f.endswith('.pickle')]
            
        dfs = []
        for file in pickle_files:
              dfs.append(pd.read_pickle(os.path.join(self.temp_dir, file)))
        return dfs
        
    def PROCESS_ALL_SPOT_CHANNELS(self):
        """
        Processes all spot channels and compiles their analyzed information.

        Returns:
        --------
        None
        """
        columns = np.unique(np.asarray(self.cell_df['column'], dtype=int))
        rows = np.unique(np.asarray(self.cell_df['row'], dtype=int))
        if self.spot_cell_index:
            self.process_channel(self.ch1_spot_df, columns, rows, 'Ch1')
            self.process_channel(self.ch2_spot_df, columns, rows, 'Ch2')
            self.process_channel(self.ch3_spot_df, columns, rows, 'Ch3')
            self.process_channel(self.ch4_spot_df, columns, rows, 'Ch4')
            self.process_channel(self.ch5_spot_df, columns, rows, 'Ch5')

            
        else:
            self.process_channel(self.gather_spot_df_list("1"), columns, rows, 'Ch1')
            self.process_channel(self.gather_spot_df_list("2"), columns, rows, 'Ch2')
            self.process_channel(self.gather_spot_df_list("3"), columns, rows, 'Ch3')
            self.process_channel(self.gather_spot_df_list("4"), columns, rows, 'Ch4')
            self.process_channel(self.gather_spot_df_list("5"), columns, rows, 'Ch5')

    def update_cell_index_in_channel(self, channel_attr, col, row, fov, time_point, previous_index, new_index):
        """
        Updates cell indices in a specified channel's DataFrame.

        Parameters:
        -----------
        channel_attr : str
            The attribute name of the channel DataFrame to be updated.
        col : int
            Column index of the well.
        row : int
            Row index of the well.
        fov : int
            Field of view index.
        time_point : int
            Time point index.
        previous_index : int
            The previous cell index.
        new_index : int
            The new cell index to replace the previous one.

        Returns:
        --------
        None
        """
        
        channel_df = getattr(self, channel_attr)
        if not channel_df.empty:
            row_index_spotdf = channel_df.loc[
                (channel_df['field_index'] == fov) & (channel_df['column'] == col) & 
                (channel_df['row'] == row) & (channel_df["time_point"] == time_point) & 
                (channel_df["cell_index"] == previous_index)
            ].index
            # print(channel_df.loc[row_index_spotdf, 'cell_index'])
            channel_df.loc[row_index_spotdf, 'cell_index'] = new_index
            # print(channel_df.loc[row_index_spotdf, 'cell_index'])
            
        setattr(self, channel_attr, channel_df) # Update the class variable
    
    def update_cell_index_in_all_spot_channels(self, col, row, fov, time_point, previous_index, new_index):
        
        """
        Updates cell indices across all spot channels.

        Parameters:
        -----------
        col : int
            Column index of the well.
        row : int
            Row index of the well.
        fov : int
            Field of view index.
        time_point : int
            Time point index.
        previous_index : int
            The previous cell index.
        new_index : int
            The new cell index to replace the previous one.

        Returns:
        --------
        None
        """
        
        for channel in ['ch1_spot_df', 'ch2_spot_df', 'ch3_spot_df', 'ch4_spot_df', 'ch5_spot_df']:

            self.update_cell_index_in_channel(channel, col, row, fov, time_point, previous_index, new_index)
        
    def normalize_stack(self, images):
        """
        Normalizes a stack of images.

        Parameters:
        -----------
        images : ndarray
            A stack of images to be normalized.

        Returns:
        --------
        ndarray
            The stack of normalized images.
        """
        normalized_stack = []
        # import random
        # import string
        for i in range(images.shape[2]):
            image = images[:, :, i]
            
            # Apply min-max normalization to each image individually
            normalized_image = cv2.normalize(image, None, 0, 240, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            normalized_stack.append(normalized_image)

        return np.stack(normalized_stack, axis=2).astype('uint8') 
    
    def cell_level_file_name(self, top_folder, folder_name, name_prefix, name_extention, col, row, fov, cell_ind,spot_ch=None):
        """
        Generates file names for saving cell level data.

        Parameters:
        -----------
        top_folder : str
            The top level directory where the files will be saved.
        folder_name : str
            The name of the folder where the files will be saved.
        name_prefix : str
            The prefix for the file name.
        name_extention : str
            The file extension.
        col : int
            Column index of the well.
        row : int
            Row index of the well.
        fov : int
            Field of view index.
        cell_ind : int
            The index of the cell.
        spot_ch : int, optional
            The channel index of the spot.

        Returns:
        --------
        str
            The generated file path for saving the data.
        """
        if "spot" in name_prefix:
            file_name = name_prefix+ r'_for_col' + str(col) + r'_row' + str(row) + r'_field' + str(fov) + r'_cell' + str(cell_ind)+ r'_Ch' + str(int(spot_ch)) + name_extention
        else:
            file_name = name_prefix+ r'_for_col' + str(col) + r'_row' + str(row) + r'_field' + str(fov) + r'_cell' + str(cell_ind) + name_extention
        
        file_path = os.path.join(top_folder, folder_name)
        if os.path.isdir(file_path) == False:
            os.mkdir(file_path)
        
        return os.path.join(file_path, file_name)
    
    def spot_level_file_name(self, top_folder, folder_name, name_prefix, name_extention, col, row, fov, cell_ind, spot_ch, spot_ind):
        """
        Generates file names for saving spot level data.

        Parameters:
        -----------
        top_folder : str
            The top level directory where the files will be saved.
        folder_name : str
            The name of the folder where the files will be saved.
        name_prefix : str
            The prefix for the file name.
        name_extention : str
            The file extension.
        col : int
            Column index of the well.
        row : int
            Row index of the well.
        fov : int
            Field of view index.
        cell_ind : int
            The index of the cell.
        spot_ch : int
            The channel index of the spot.
        spot_ind : int
            The index of the spot.

        Returns:
        --------
        str
            The generated file path for saving the data.
        """
        file_name = name_prefix+ r'_for_col' + str(col) + r'_row' + str(row) + r'_field' + str(fov) + r'_cell' + str(cell_ind)+ r'_Ch' + str(int(spot_ch))+ r'_spot' + str(spot_ind) + name_extention
        
        file_path = os.path.join(top_folder, folder_name)
        if os.path.isdir(file_path) == False:
            os.mkdir(file_path)
        
        return os.path.join(file_path, file_name)
        
    @log_errors(logging.getLogger(__name__))
    def BATCH_ANALYZER(self, col,row,fov,t): 
        
        """
        Analyzes a batch of images for nuclei and spots, updating the results lists.

        Parameters:
        -----------
        col : int
            Column index of the well.
        row : int
            Row index of the well.
        fov : int
            Field of view index.
        t : int
            Time point index.

        Returns:
        --------
        None
        """
        self.ImageAnalyzer = ImageAnalyzer(self.params_dict)
        ai = 1
        
        self.df_checker = self.Meta_Data_df.loc[(self.Meta_Data_df['column'] == str(col)) & 
                                              (self.Meta_Data_df['row'] == str(row)) & 
                                              (self.Meta_Data_df['field_index'] == str(fov)) & 
                                              (self.Meta_Data_df['time_point'] == str(t))]
        if self.df_checker.empty == False:
        
            loadedimg_formask = self.IMG_FOR_NUC_MASK()

            if loadedimg_formask.ndim ==2:
                
                ImageForNucMask = cv2.normalize(loadedimg_formask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                nuc_bndry, nuc_mask = self.ImageAnalyzer.neuceli_segmenter(ImageForNucMask, self.Meta_Data_df["PixPerMic"].iloc[0])
                
                labeled_nuc, number_nuc = label(nuc_mask)
                nuc_labels = np.unique(labeled_nuc)
                if nuc_labels.max()>0:
                    
                    
                    dist_img = distance_transform_edt(nuc_mask)
                    dist_props = regionprops_table(labeled_nuc, dist_img, properties=('label', 'max_intensity'))
                    radial_dist_df = pd.DataFrame(dist_props)

                    
                    data = { "Experiment":[self.experiment_name]*number_nuc,
                             "column": [col]*number_nuc, "row": [row]*number_nuc, 
                             "time_point": [t]*number_nuc, "field_index": [fov]*number_nuc,
                             "z_slice": ["max_project"]*number_nuc, "action_index":[ai]*number_nuc}
                    df = pd.DataFrame(data)
                    regions = regionprops(labeled_nuc, loadedimg_formask)
                    props = regionprops_table(labeled_nuc, loadedimg_formask, properties=(
                                            'label', 'centroid', 'orientation', 'major_axis_length', 'minor_axis_length',
                                            'area', 'max_intensity', 'min_intensity', 'mean_intensity',
                                            'orientation', 'perimeter', 'solidity'))
                    pixpermicron = np.asarray(self.Meta_Data_df["PixPerMic"].iloc[0]).astype(float)
                    props_df = pd.DataFrame(props)
                    props_df['sum_intensity'] = props_df['area'] * props_df['mean_intensity']
                    props_df['major_axis_length'] = pixpermicron*props_df['major_axis_length']
                    props_df['minor_axis_length'] = pixpermicron*props_df['minor_axis_length']
                    props_df['area'] = pixpermicron*pixpermicron*props_df['area']
                    props_df['perimeter'] = pixpermicron*props_df['perimeter']
                    image_cells_df = pd.concat([df,props_df], axis=1)
                    
                    labeled_boundary = np.multiply(find_boundaries(labeled_nuc, mode='inner', background=0),labeled_nuc)
                    # List to store the coordinates of each nucleus boundary
                    efc_ratio = {}
                    # Iterate over each region to extract its boundary coordinates
                    for region in regionprops(labeled_boundary):
                        ecd_coeffs = spatial_efd.CalculateEFD(region.coords[:,0], region.coords[:,1], 15)
                        EFDcoeffs, rotation = spatial_efd.normalize_efd(ecd_coeffs, size_invariant = True)
                        MinorAxisArray = []
                        MajorAxisArray = []
                        for i in range(len(EFDcoeffs)):
                            MinorAxis = (EFDcoeffs[i][0]**2 + EFDcoeffs[i][2]**2)**(1/2)
                            MajorAxis = (EFDcoeffs[i][1]**2 + EFDcoeffs[i][3]**2)**(1/2)
                            MinorAxisArray.append(MinorAxis)
                            MajorAxisArray.append(MajorAxis)    
                    
                        efc_ratio[region.label] = (MinorAxisArray[0] + MajorAxisArray[0])/( sum(MinorAxisArray[1:]) + sum(MajorAxisArray[1:]))
                    image_cells_df['efc_ratio']=0
                    image_cells_df['efc_ratio'] = image_cells_df['label'].map(efc_ratio).fillna(image_cells_df['efc_ratio'])
                    
                    if (self.params_dict['SecChannel_current_index'] > 0) & (self.params_dict['SecArea_current_index'] > 0):
                        
                        secondary_image_df = self.df_checker.loc[(self.df_checker['channel'] == str(self.params_dict['SecChannel_current_index']))]
                        secondary_image = self.ImageAnalyzer.max_z_project(secondary_image_df)

                        sec_props = regionprops_table(labeled_nuc, secondary_image, properties=('label','area', 'mean_intensity'))
                        sec_props_df = pd.DataFrame(sec_props)
                        sec_props_df['sum_intensity'] = sec_props_df['mean_intensity']*sec_props_df['area']
                        
                        col1_name = "ch" + str(self.params_dict['SecChannel_current_index']) + "_mean_intensity"
                        image_cells_df[col1_name]=0
                        col2_name = "ch" + str(self.params_dict['SecChannel_current_index']) + "_sum_intensity"
                        image_cells_df[col2_name]=0
                        
                        mapping_dict = sec_props_df.set_index('label')['mean_intensity'].to_dict()
                        image_cells_df[col1_name] = image_cells_df['label'].map(mapping_dict).fillna(image_cells_df[col1_name])

                        mapping_dict = sec_props_df.set_index('label')['sum_intensity'].to_dict()
                        image_cells_df[col2_name] = image_cells_df['label'].map(mapping_dict).fillna(image_cells_df[col2_name])

                    cell_file_name = os.path.join(self.temp_dir , "temp_cell_df_column_"+str(col)+"_row_" + str(row) + "_time_point_" + str(t) + "_field_index_" +str(fov)+'.pickle')
                    image_cells_df.to_pickle(cell_file_name)
                    
                    # self.cell_pd_list.append(image_cells_df)

                if self.params_dict['NucMaskCheckBox_status_check'] == True:
                    nuc_mask_output_folder = os.path.join(self.output_folder, 'nuclei_masks')
                    if os.path.isdir(nuc_mask_output_folder) == False:
                        os.mkdir(nuc_mask_output_folder) 

                    mask_file_name = ['Nuclei_Mask_for_Col' + str(col) + r'_row' + str(row)+
                                      r'_Time' + str(t) + r'_Field' + str(fov) + r'.tif']
                    mask_full_name = os.path.join(nuc_mask_output_folder, mask_file_name[0])
#                         cv2.imwrite(mask_full_name,nuc_mask)

                    imwrite(mask_full_name,nuc_mask)


            else:

                nuc_bndry, nuc_mask = self.Z_STACK_NUC_SEGMENTER(ImageForNucMask)
                label_nuc_stack = self.Z_STACK_NUC_LABLER(ImageForLabel)

            ch1_xyz, ch1_xyz_3D, ch1_final_spots, ch2_xyz, ch2_xyz_3D, ch2_final_spots, ch3_xyz, ch3_xyz_3D, ch3_final_spots, ch4_xyz, ch4_xyz_3D, ch4_final_spots, ch5_xyz, ch5_xyz_3D, ch5_final_spots  = self.IMAGE_FOR_SPOT_DETECTION( ImageForNucMask, nuc_mask)

            if self.params_dict['NucMaxZprojectCheckBox_status_check'] == True:
                if self.params_dict['SpotMaxZProject_status_check'] == True:
                    # Assuming channels are numbered from 1 to 5
                    for channel in range(1, 6):
                        channel_key = f'SpotCh{channel}CheckBox_status_check'
                        
                        if self.params_dict[channel_key]:
                            xyz = locals()[f'ch{channel}_xyz']
                            
                            if xyz.size > 0:
                                xyz_round = np.round(np.asarray(xyz)).astype('int')
                                spot_nuc_labels = labeled_nuc[xyz_round[:, 0], xyz_round[:, 1]]
                                num_spots = len(xyz)
                                
                                if labeled_nuc.max() > 0:
                                    radial = self.RADIAL_DIST_CALC(xyz_round, spot_nuc_labels, radial_dist_df, dist_img)
                                else:
                                    radial = np.nan
                    
                                data = {
                                    "Experiment": [self.experiment_name] * num_spots,
                                    "column": [col] * num_spots,
                                    "row": [row] * num_spots,
                                    "time_point": [t] * num_spots,
                                    "field_index": [fov] * num_spots,
                                    "z_slice": ["max_project"] * num_spots,
                                    "channel": [channel] * num_spots,
                                    "action_index": [ai] * num_spots,
                                    "cell_index": spot_nuc_labels,
                                    "x_location": xyz[:, 0],
                                    "y_location": xyz[:, 1],
                                    "z_location": xyz[:, 2],
                                    "radial_distance": radial
                                }
                    
                                df_channel = pd.concat([pd.DataFrame(data), locals()[f'ch{channel}_final_spots']], axis=1)    
                                spot_file_name = os.path.join(self.temp_dir ,
                                                              "temp_spot_df_column_"+str(col)+"_row_" + str(row) + "_time_point_" + str(t) + "_field_index_" +str(fov)+"_channel_" +str(channel)+'.pickle')
                                df_channel.to_pickle(spot_file_name)



    def Calculate_Spot_Distances(self, row, col):
        
        """
        Calculates the distances between spots for given cells.

        Parameters:
        -----------
        row : int
            Row index of the well.
        col : int
            Column index of the well.

        Returns:
        --------
        None
        """
        
        select_cell = self.cell_df.loc[(self.cell_df['column'] == col) & (self.cell_df['row'] == row)]    

        spot_pd_dict = {}

        if self.ch1_spot_df.empty == False:

            cell_spots_ch1 = self.ch1_spot_df.loc[(self.ch1_spot_df['column'] == col) & (self.ch1_spot_df['row'] == row)] 

            spot_pd_dict['ch1'] = cell_spots_ch1

        if self.ch2_spot_df.empty == False:

            cell_spots_ch2 = self.ch2_spot_df.loc[(self.ch2_spot_df['column'] == col) & (self.ch2_spot_df['row'] == row)] 

            spot_pd_dict['ch2'] = cell_spots_ch2

        if self.ch3_spot_df.empty == False:

            cell_spots_ch3 = self.ch3_spot_df.loc[(self.ch3_spot_df['column'] == col) & (self.ch3_spot_df['row'] == row) ] 

            spot_pd_dict['ch3'] = cell_spots_ch3

        if self.ch4_spot_df.empty == False:

            cell_spots_ch4 = self.ch4_spot_df.loc[(self.ch4_spot_df['column'] == col) & (self.ch4_spot_df['row'] == row)] 

            spot_pd_dict['ch4'] = cell_spots_ch4

        if self.ch5_spot_df.empty == False:

            cell_spots_ch5 = self.ch5_spot_df.loc[(self.ch5_spot_df['column'] == col) & (self.ch5_spot_df['row'] == row)] 

            spot_pd_dict['ch5'] = cell_spots_ch5

        ch_distances = []

        for key1 in spot_pd_dict.keys():
            for key2 in spot_pd_dict.keys():

                ch_distance1 = key2 + r'_' + key1
                if ch_distance1 in ch_distances:
                    pass
                else:
                    ch_distance = key1 + r'_' + key2
                    ch_distances.append(ch_distance)

                    dist_pd = self.DISTANCE_calculator(spot_pd_dict,key1,key2,select_cell, row, col)
                    well_spot_dist_folder = os.path.join(self.output_folder, 'well_spots_distances')
                    if os.path.isdir(well_spot_dist_folder) == False:
                        os.mkdir(well_spot_dist_folder)

                    spot_dist_filename = 'SpotDistances_' + key1 +'_' +key2 +'_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                    spot_dist_well_csv_full_name = os.path.join(well_spot_dist_folder, spot_dist_filename)
                    dist_pd.to_csv(path_or_buf=spot_dist_well_csv_full_name, encoding='utf8')

    def DISTANCE_calculator(self, spot_pd_dict, key1, key2,select_cell, row, col):
        
        """
        A helper function for calculating distances between spots.

        Parameters:
        -----------
        spot_pd_dict : dict
            Dictionary of spot DataFrames keyed by channel name.
        key1 : str
            The first channel key for comparison.
        key2 : str
            The second channel key for comparison.
        select_cell : DataFrame
            DataFrame of selected cells for distance calculation.
        row : int
            Row index of the well.
        col : int
            Column index of the well.

        Returns:
        --------
        DataFrame
            A DataFrame containing the calculated distances.
        """
        
        fovs = np.unique(np.asarray(select_cell['field_index'], dtype=int))
        time_points = np.unique(np.asarray(select_cell['time_point'], dtype=int))
        actionindices = np.unique(np.asarray(select_cell['action_index'], dtype=int))
        z_slices = np.unique(np.asarray(select_cell['z_slice']))
        dist_pd = pd.DataFrame()
        for f in fovs:
            for t in time_points:
                for a in actionindices:
                    for z in z_slices:

                        cells_in_field = select_cell.loc[ 
                                              (select_cell['time_point'] == t) & 
                                              (select_cell['field_index'] == f)& 
                                              (select_cell['action_index'] == a)&
                                              (select_cell['z_slice'] == z)] 
                        cells_in_field = np.unique(np.asarray(select_cell['label'], dtype=int))

                        for c in cells_in_field:
                            loci1 = spot_pd_dict[key1].loc[ 
                                                  (spot_pd_dict[key1]['time_point'] == t) & 
                                                  (spot_pd_dict[key1]['field_index'] == f)& 
                                                  (spot_pd_dict[key1]['action_index'] == a)&
                                                  (spot_pd_dict[key1]['z_slice'] == z) &
                                                  (spot_pd_dict[key1]['cell_index'] == c)] 
                            loci2 = spot_pd_dict[key2].loc[ 
                                                  (spot_pd_dict[key2]['time_point'] == t) & 
                                                  (spot_pd_dict[key2]['field_index'] == f)& 
                                                  (spot_pd_dict[key2]['action_index'] == a)&
                                                  (spot_pd_dict[key2]['z_slice'] == z) &
                                                  (spot_pd_dict[key2]['cell_index'] == c)] 

                            for ind1, locus1 in loci1.iterrows():
                                for ind2, locus2 in loci2.iterrows():
                                    if ind1!=ind2:
                                        dist_2d =  math.sqrt(
                                                            math.pow((locus1['x_location'] - locus2['x_location']),2) +
                                                            math.pow((locus1['y_location'] - locus2['y_location']),2) 
                                                            )

                                        dist_3d =  math.sqrt(
                                                            math.pow((locus1['x_location'] - locus2['x_location']),2) +
                                                            math.pow((locus1['y_location'] - locus2['y_location']),2) +
                                                            math.pow((locus1['z_location'] - locus2['z_location']),2)
                                                            )

                                        s1 = key1 + '_spot_index(1)'
                                        s2 = key2 + '_spot_index(2)'
                                        data = {"Experiment":[self.experiment_name],
                                                'column':col, 'row': row, 'time_point': [t], 'field_index': [f], 
                                                'action_index': [a], 'z_slice':[z], 'cell_index': [c],
                                                str(s1): [ind1], str(s2): [ind2], 
                                                'XY-Distance(pixels)': [dist_2d], 'XYZ-Distance(pixels)': [dist_3d],
                                                'XY-Distance(micron)': [dist_2d*np.asarray(self.Meta_Data_df["PixPerMic"].iloc[0]).astype(float)], 
                                                'XYZ-Distance(micron)':[dist_3d*np.asarray(self.Meta_Data_df["PixPerMic"].iloc[0]).astype(float)]}
                                        temp_df = pd.DataFrame(data)
                                        dist_pd = pd.concat([dist_pd, temp_df], ignore_index=True)

        return dist_pd

                                                        
    def IMG_FOR_NUC_MASK(self):
        """
        Loads the image used for creating nuclei masks.

        Returns:
        --------
        ndarray
            The loaded image for creating nuclei masks.
        """
        if self.df_checker.empty == False:

            if self.params_dict['NucMaxZprojectCheckBox_status_check'] == True:
                maskchannel = str(self.params_dict['NucleiChannel'][-1])
                imgformask = self.df_checker.loc[(self.df_checker['channel'] == maskchannel)]
                loadedimg_formask = self.ImageAnalyzer.max_z_project(imgformask)
                
            else:
                z_imglist=[]
                maskchannel = str(self.params_dict['NucleiChannel'][-1])
                imgformask = self.df_checker.loc[(self.df_checker['channel'] == maskchannel)]

                for index, row in imgformask.iterrows():
                    
                    if row['ImageName']=="dask_array":
                        # im = row["Type"].compute()
                        im = row["Type"]
                    else: 
                        # im = mpimg.imread(row['ImageName'])
                        im = imread(row['ImageName'])
                    im_uint8 = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    z_imglist.append( np.asarray(im_uint8))
                loadedimg_formask = np.stack(z_imglist, axis=2)
        
        return loadedimg_formask
    
    def RAW_IMAGE_LOADER(self, maskchannel):
        """
        Loads raw images for a specified channel.

        Parameters:
        -----------
        maskchannel : str
            The channel for which the raw images are to be loaded.

        Returns:
        --------
        ndarray
            The loaded raw image for the specified channel.
        """
        if self.df_checker.empty == False:

            if self.params_dict['NucMaxZprojectCheckBox_status_check'] == True:

                imgformask = self.df_checker.loc[(self.df_checker['channel'] == maskchannel)]
                loadedimg_formask = self.ImageAnalyzer.max_z_project(imgformask)
                loaded_image = loadedimg_formask

            else:
                z_imglist=[]
                imgformask = self.df_checker.loc[(self.df_checker['channel'] == maskchannel)]

                for index, row in imgformask.iterrows():
                    
                    if row['ImageName']=="dask_array":
                        # im = row["Type"].compute()
                        im = row["Type"]
                    else: 
                        # im = mpimg.imread(row['ImageName'])
                        im = imread(row['ImageName'])
                    z_imglist.append( np.asarray(im))
                loaded_image = np.stack(z_imglist, axis=2)
        
        return loaded_image

    def Z_STACK_NUC_SEGMENTER(self, ImageForNucMask):
        """
        Segments nuclei from a z-stack of images.

        Parameters:
        -----------
        ImageForNucMask : ndarray
            The z-stack image for nuclei segmentation.

        Returns:
        --------
        tuple
            A tuple containing the boundary and mask images for segmented nuclei.
        """
        nuc_bndry_list, nuc_mask_list = [], []
        w, h, z_plane = ImageForNucMask.shape

        for z in range(z_plane):

            single_z_img = ImageForNucMask[:,:,z]
            nuc_bndry_single, nuc_mask_single = self.ImageAnalyzer.neuceli_segmenter(single_z_img,
                                                                                     self.Meta_Data_df["PixPerMic"].iloc[0])

            nuc_bndry_list.append( np.asarray(nuc_bndry_single))
            nuc_mask_list.append( np.asarray(nuc_mask_single))
        nuc_bndry = np.stack(nuc_bndry_list, axis=2)
        nuc_mask = np.stack(nuc_mask_list, axis=2)
    
        return nuc_bndry, nuc_mask
    
    def Z_STACK_NUC_LABLER(self, ImageForLabel):
        """
        Labels nuclei in a z-stack of images.

        Parameters:
        -----------
        ImageForLabel : ndarray
            The z-stack image for nuclei labeling.

        Returns:
        --------
        ndarray
            The labeled nuclei image stack.
        """
        label_nuc_list = []
        w, h, z_plane = ImageForLabel.shape

        for z in range(z_plane):

            single_z_img = ImageForLabel[:,:,z]
            labeled_nuc, number_nuc = label(single_z_img)

            label_nuc_list.append( np.asarray(labeled_nuc))
            
        label_nuc_stack = np.stack(label_nuc_list, axis=2)
    
        return label_nuc_stack
    
    
    def IMAGE_FOR_SPOT_DETECTION(self, ImageForNucMask, nuc_mask=None):
        """
        Prepares images for spot detection and retrieves coordinates.

        Parameters:
        -----------
        nuc_mask : ndarray
            The nuclei mask to assist in spot detection.

        Returns:
        --------
        tuple
            A tuple containing the XYZ coordinates and final spots for different channels.
        """
        ch1_xyz, ch2_xyz, ch3_xyz, ch4_xyz, ch5_xyz = [],[],[],[],[]
        ch1_xyz_3D, ch2_xyz_3D, ch3_xyz_3D, ch4_xyz_3D, ch5_xyz_3D = [],[],[],[],[]
        ch1_final_spots, ch2_final_spots, ch3_final_spots, ch4_final_spots, ch5_final_spots = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        if self.params_dict['SpotCh1CheckBox_status_check'] == True:
            
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '1')]
            ch1_xyz, ch1_xyz_3D, ch1_final_spots = self.XYZ_SPOT_COORDINATES( imgforspot, nuc_mask, 'Ch1')
                
        if self.params_dict['SpotCh2CheckBox_status_check'] == True:
            
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '2')]
            ch2_xyz, ch2_xyz_3D, ch2_final_spots = self.XYZ_SPOT_COORDINATES( imgforspot, nuc_mask, 'Ch2')
                
        if self.params_dict['SpotCh3CheckBox_status_check'] == True:
            
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '3')]
            ch3_xyz, ch3_xyz_3D, ch3_final_spots = self.XYZ_SPOT_COORDINATES( imgforspot, nuc_mask, 'Ch3')
                    
        if self.params_dict['SpotCh4CheckBox_status_check'] == True:
             
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '4')]
            ch4_xyz, ch4_xyz_3D, ch4_final_spots = self.XYZ_SPOT_COORDINATES( imgforspot, nuc_mask, 'Ch4')
        
        if self.params_dict['SpotCh5CheckBox_status_check'] == True:
             
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '5')]
            ch5_xyz, ch5_xyz_3D, ch5_final_spots = self.XYZ_SPOT_COORDINATES( imgforspot, nuc_mask, 'Ch5')
            
        return ch1_xyz, ch1_xyz_3D, ch1_final_spots, ch2_xyz, ch2_xyz_3D, ch2_final_spots, ch3_xyz, ch3_xyz_3D, ch3_final_spots, ch4_xyz, ch4_xyz_3D, ch4_final_spots, ch5_xyz, ch5_xyz_3D, ch5_final_spots 

    
    def XYZ_SPOT_COORDINATES(self, images_pd_df, nuc_mask, spot_channel):
        """
        Calculates XYZ coordinates of spots in given images, processing each z-slice and generating a max projection.

        Parameters:
        -----------
        images_pd_df : DataFrame
            DataFrame containing image data, including image paths and z-slice information.
        ImageForNucMask : ndarray
            Nuclei image used as a mask for spot detection.
        spot_channel : str
            Channel identifier used for spot detection.

        Returns:
        --------
        tuple
            A tuple containing three elements:
            1. xyz_coordinates (ndarray): Array of XYZ coordinates of detected spots in the max projection.
            2. coordinates_stack (ndarray): Stack of XYZ coordinates of spots in each z-slice.
            3. final_spots (DataFrame): DataFrame containing the final processed spot data.
        """
        
        z_imglist = []
        coordinates_stack = np.ones((0,3), dtype='float')
        xyz_coordinates = []        
            
        for index, row in images_pd_df.iterrows():
            
            if row['ImageName']=="dask_array":
                # im = row["Type"].compute()
                im = row["Type"]
            else: 
                # im = mpimg.imread(row['ImageName'])
                im = imread(row['ImageName'])
            z_imglist.append( np.asarray(im))
            _z_coordinates1 = np.asarray(row['z_slice']).astype('float')
            
            
            detection_methods = ["Laplacian of Gaussian", "Gaussian", "Intensity Threshold", "Enhanced LOG"] 
            threshold_methods = ["Auto", "Manual"]
            
            
            params_to_pass= self.params_dict['spot_params_dict'][spot_channel]

            coordinates, final_spots = self.ImageAnalyzer.SpotDetector(
                                                                        input_image_raw=im, 
                                                                        nuc_mask=nuc_mask, 
                                                                        spot_detection_method= detection_methods[params_to_pass[0]],
                                                                        threshold_method= threshold_methods[params_to_pass[1]],
                                                                        threshold_value= params_to_pass[2],
                                                                        kernel_size= params_to_pass[3],
                                                                        spot_location_coords= self.params_dict['SpotLocationCbox_currentText'],
                                                                        remove_bright_junk= self.params_dict['RemoveBrightJunk_status_check'],
                                                                        resize_factor= params_to_pass[4],
                                                                        min_area= params_to_pass[5],
                                                                        max_area= params_to_pass[6],
                                                                        min_integrated_intensity= params_to_pass[7],
                                                                        psf_size= self.params_dict['PSFsizeSpinBox_value'],
                                                                        gaussian_fit= self.params_dict['IntegratedIntensity_fitStatus']>0,
                
                                                                    )

            
            if coordinates.__len__()>0:
                _z_coordinates = np.ones((coordinates.__len__(),1), dtype='float')*_z_coordinates1
            else:
                coordinates=np.ones((0,2), dtype='float')
                _z_coordinates=np.ones((0,1), dtype='float')
            
            xyz_3d_coordinates = np.append(np.asarray(coordinates).astype('float'), _z_coordinates, 1)
            coordinates_stack = np.append(coordinates_stack, xyz_3d_coordinates,0)
            

        if z_imglist.__len__()>0:
            print(row['ImageName'])
            image_stack = np.stack(z_imglist, axis=2)
            max_project = image_stack.max(axis=2)
            
            detection_methods = ["Laplacian of Gaussian", "Gaussian", "Intensity Threshold", "Enhanced LOG"] 
            threshold_methods = ["Auto", "Manual"]
            
            params_to_pass= self.params_dict['spot_params_dict'][spot_channel]

            coordinates_max_project, final_spots = self.ImageAnalyzer.SpotDetector(
                                                                        input_image_raw=im, 
                                                                        nuc_mask=nuc_mask, 
                                                                        spot_detection_method= detection_methods[params_to_pass[0]],
                                                                        threshold_method= threshold_methods[params_to_pass[1]],
                                                                        threshold_value= params_to_pass[2],
                                                                        kernel_size= params_to_pass[3],
                                                                        spot_location_coords= self.params_dict['SpotLocationCbox_currentText'],
                                                                        remove_bright_junk= self.params_dict['RemoveBrightJunk_status_check'],
                                                                        resize_factor= params_to_pass[4],
                                                                        min_area= params_to_pass[5],
                                                                        max_area= params_to_pass[6],
                                                                        min_integrated_intensity= params_to_pass[7],
                                                                        psf_size= self.params_dict['PSFsizeSpinBox_value'],
                                                                        gaussian_fit= self.params_dict['IntegratedIntensity_fitStatus']>0
                                                                    )
        
            if coordinates_max_project.__len__()>0:

                coordinates_max_project_round = np.round(np.asarray(coordinates_max_project)).astype('int')
                coordinates_max_project = np.array(coordinates_max_project)[coordinates_max_project_round.min(axis=1)>=0,:].tolist()
                coordinates_max_project_round = coordinates_max_project_round[coordinates_max_project_round.min(axis=1)>=0,:]
                spots_z_slices = np.argmax(image_stack[coordinates_max_project_round[:,0],coordinates_max_project_round[:,1],:], axis=1)
                spots_z_coordinates = np.zeros((spots_z_slices.__len__(),1), dtype='float')
                
                for i in range(spots_z_slices.__len__()):

                    spots_z_coordinates[i] = np.asarray(images_pd_df.loc[images_pd_df['z_slice']== str(spots_z_slices[i]+1)]
                                                     ['z_coordinate'].iloc[0]).astype('float')
                if coordinates_max_project==[]:
                    coordinates_max_project=np.ones((0,2), dtype='float')
                xyz_coordinates = np.append(np.asarray(coordinates_max_project).astype('float'), spots_z_coordinates, 1)

        return np.array(xyz_coordinates), np.array(coordinates_stack), final_spots
    
   
    def RADIAL_DIST_CALC(self, xyz_round, spot_nuc_labels, radial_dist_df, dist_img):
        
        """
        Calculates the radial distance of each spot from the center of its respective nucleus.

        Parameters:
        -----------
        xyz_round : ndarray
            Rounded XYZ coordinates of detected spots.
        spot_nuc_labels : ndarray
            Labels indicating to which nucleus each spot belongs.
        radial_dist_df : DataFrame
            DataFrame containing radial distance information for nuclei.
        dist_img : ndarray
            Distance transform image of nuclei.

        Returns:
        --------
        ndarray
            Array containing the radial distance of each spot from the center of its nucleus.
        """
        radial_dist=[]
        eps=0.000001
        for i in range(xyz_round.__len__()):
            
            sp_dist = dist_img[xyz_round[i,0], xyz_round[i,1]]
            spot_lbl =int(spot_nuc_labels[i])
            if spot_lbl>0:
                cell_max = radial_dist_df.loc[radial_dist_df['label']==spot_lbl]['max_intensity'].iloc[0]
                sp_radial_dist= (cell_max-sp_dist)/(cell_max-1+eps)
            else:
                sp_radial_dist = np.nan
            radial_dist.append(sp_radial_dist)
    
        return np.array(radial_dist).astype(float)

    def SAVE_CONFIGURATION(self, csv_filename, params_dict):
        det_method = ["Laplacian of Gaussian", "Gaussian", "Intensity Threshold", "Enhanced LOG"]
        thresh_method = ["Auto", "Manual"]
    
        # Function to retrieve spot parameters
        def get_spot_params(channel):
            params = params_dict['spot_params_dict'][channel]
            return [
                det_method[int(params[0])],
                thresh_method[int(params[1])],
                params[2],  # Threshold value
                params[3],  # Kernel size
                params[4],  # Spots/ch
                params[5],  # Spots area min
                params[6],  # Spots area max
                params[7]   # Spots integrated intensity
            ]
    
        # Initialize config_data with specific mappings from params_dict
        config_data = {
            "nuclei_channel": params_dict.get("NucleiChannel", ""),
            "nuclei_detection_method": params_dict.get("NucDetectMethod_currentText", ""),
            "nuclei_z_project": params_dict.get("NucMaxZprojectCheckBox_status_check", False),
            "remove_boundary_nuclei": params_dict.get("NucRemoveBoundaryCheckBox_isChecked", False),
            "nuclei_detection": params_dict.get("NucDetectionSlider_value", 0),
            "nuclei_separation": params_dict.get("NucSeparationSlider_value", 0),
            "nuclei_area": params_dict.get("NucleiAreaSlider_value", 0),
            "spot_coordinates": params_dict.get("SpotLocationCbox_currentText", ""),
            "spot_z_project": params_dict.get("SpotMaxZProject_status_check", False),
            "Nuclei_Info_CheckBox_status": params_dict.get("NucInfoChkBox_check_status", False),
            "Spots_Location_status": params_dict.get("SpotsLocation_check_status", False),
            "Spots_Tracking_status": params_dict.get("Spot_Tracking_check_status", False),
            "Nuclei_MaskCheckBox_status": params_dict.get("NucMaskCheckBox_status_check", False),
            "RemoveBrightJunk_status_check": params_dict.get("RemoveBrightJunk_status_check", False),
            "NumCPUsSpinBox_value": params_dict.get("NumCPUsSpinBox_value", 0),
            "Cell_Tracking_check_status": params_dict.get("Cell_Tracking_check_status", False),
            "NucTrackingMethod": params_dict.get("NucTrackingMethod_currentText", ""),
            "NucSearchRadius": params_dict.get("NucSearchRadiusSpinbox_current_value", 0),
            "SpotSearchRadius_value": params_dict.get("SpotSearchRadiusSpinbox_current_value", 0),
            "Sec_SpotSearchRadius_value": params_dict.get("Sec_SpotSearchRadiusSpinbox_current_value", 0),
            "Secondary_Area_index": params_dict.get("SecArea_current_index", 0),
            "MintrackLength_value": params_dict.get("MintrackLengthSpinbox_current_value", 0),
            "maxspotspercell_value": params_dict.get("maxspotspercellSpinbox_current_value", 0),
            "minburstduration_value": params_dict.get("minburstdurationSpinbox_current_value", 0),
            "FittingMethod_index": params_dict.get("FittingMethod_index", 0),
            "patchsize": params_dict.get("patchsize_currentText", 0),
            "IntegratedIntensity_fitStatus": params_dict.get("IntegratedIntensity_fitStatus", 0),
            "Registrationmethod": params_dict.get("Registrationmethod_currentText", ""),
            "PSFsize_value": params_dict.get("PSFsizeSpinBox_value", 0),
            "Seceondary_Channel_index": params_dict.get("SecChannel_current_index", 0),  # Assuming this is one of the missing parameters
            "IntegratedIntensity_Index": params_dict.get("IntegratedIntensityCbox_currentIndex", 0),  # Assuming this is one of the missing parameters
            "Nuclei_MaxZproject_CheckBox_status": params_dict.get("NucMaxZprojectCheckBox_status_check", False),  # Assuming this is one of the missing parameters
        }
    
        # Include spot status for each channel (chX_spot)
        for ch_num in range(1, 6):
            ch_key = f"SpotCh{ch_num}CheckBox_status_check"
            config_data[f"ch{ch_num}_spot"] = params_dict.get(ch_key, False)
    
        # Adding spot parameters for each channel
        for ch in ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']:
            ch_params = get_spot_params(ch)
            config_data.update({
                f"{ch.lower()}_spot_detection_method": ch_params[0],
                f"{ch.lower()}_spot_threshold_method": ch_params[1],
                f"{ch.lower()}_spot_threshold_value": ch_params[2],
                f"{ch.lower()}_kernel_size": ch_params[3],
                f"{ch.lower()}_spots/ch": ch_params[4],
                f"{ch.lower()}_spots_area_min": ch_params[5],
                f"{ch.lower()}_spots_area_max": ch_params[6],
                f"{ch.lower()}_spots_integrated_intensity": ch_params[7]
            })
    
        config_df = pd.DataFrame(list(config_data.items()), columns=['Parameter', 'Value'])
        config_df.set_index('Parameter', inplace=True)
        config_df.to_csv(csv_filename, index=True)



if __name__ == '__main__':
    freeze_support()