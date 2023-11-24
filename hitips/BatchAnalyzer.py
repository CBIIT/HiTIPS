import numpy as np
import cv2
import math
import os
import time
from skimage.measure import regionprops, regionprops_table
from skimage.io import imread
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image, ImageQt
from scipy.ndimage import label, distance_transform_edt
import multiprocessing
from multiprocessing import Pool, Process, Manager
from .GUI_parameters import Gui_Params
import btrack
from btrack.constants import BayesianUpdates
import imageio
from tifffile import imwrite
from skimage.transform import rotate, warp_polar, rescale
from matplotlib import pyplot as plt
from skimage.color import label2rgb, gray2rgb
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from deepcell.applications import CellTracking
from skimage.feature import peak_local_max
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import adapted_rand_error, contingency_table, mean_squared_error, peak_signal_noise_ratio, variation_of_information
from skimage.filters import threshold_otsu, median
from sklearn import mixture
from skimage.morphology import disk, binary_closing, skeletonize, binary_opening, binary_erosion, white_tophat
from skimage.registration import phase_cross_correlation
from skimage.util import img_as_float
from scipy.cluster.hierarchy import linkage, fcluster
from hmmlearn import hmm
from scipy.spatial.distance import pdist, cdist, squareform
import spatial_efd
from skimage.segmentation import watershed, find_boundaries

WELL_PLATE_ROWS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

class BatchAnalysis(object):
    output_folder = []
    manager = Manager()
    cell_pd_list, ch1_spot_df_list, ch2_spot_df_list = manager.list(), manager.list(), manager.list()
    ch3_spot_df_list, ch4_spot_df_list, ch5_spot_df_list = manager.list(), manager.list(), manager.list()
    ### make sure all the lists are clear
    cell_pd_list[:], ch1_spot_df_list[:], ch2_spot_df_list[:] = [], [], []
    ch3_spot_df_list[:], ch4_spot_df_list[:], ch5_spot_df_list[:] = [], [], []
    def __init__(self,analysisgui, image_analyzer, inout_resource_gui, displaygui, ImDisplay):

        self.inout_resource_gui = inout_resource_gui
        self.AnalysisGui = analysisgui
        self.displaygui = displaygui
        self.ImageAnalyzer = image_analyzer
        self.ImDisplay = ImDisplay
        self.ch1_spot_df, self.ch2_spot_df, self.ch3_spot_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.ch4_spot_df, self.ch5_spot_df, self.cell_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.df_checker = pd.DataFrame()
        self.Meta_Data_df = pd.DataFrame()
        self.spot_distances = {}
        self.experiment_name = []
        self.output_prefix = []
        self.output_folder = []
        self.gui_params = Gui_Params(self.AnalysisGui,self.inout_resource_gui)
        
    def SAVE_NUCLEI_INFORMATION(self, cell_df, columns, rows):
            
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
        if len(channel_spot_df_list) > 0:
            channel_spot_df = pd.concat(channel_spot_df_list)
            if self.gui_params.SpotsLocation_check_status:
                self.SAVE_SPOT_INFO(channel_spot_df, self.gui_params.SpotLocationCbox_currentText, columns, rows, channel_name)
    
    def PROCESS_ALL_SPOT_CHANNELS(self):

        columns = np.unique(np.asarray(self.cell_df['column'], dtype=int))
        rows = np.unique(np.asarray(self.cell_df['row'], dtype=int))
    
        # Process each channel
        self.process_channel(self.ch1_spot_df_list, columns, rows, 'Ch1')
        self.process_channel(self.ch2_spot_df_list, columns, rows, 'Ch2')
        self.process_channel(self.ch3_spot_df_list, columns, rows, 'Ch3')
        self.process_channel(self.ch4_spot_df_list, columns, rows, 'Ch4')
        self.process_channel(self.ch5_spot_df_list, columns, rows, 'Ch5')
        
    def update_cell_index_in_channel(self, channel_attr, col, row, fov, time_point, previous_index, new_index):
        channel_df = getattr(self, channel_attr)
        if not channel_df.empty:
            row_index_spotdf = channel_df.loc[
                (channel_df['field_index'] == fov) & (channel_df['column'] == col) & 
                (channel_df['row'] == row) & (channel_df["time_point"] == time_point) & 
                (channel_df["cell_index"] == previous_index)
            ].index
            channel_df.loc[row_index_spotdf, 'cell_index'] = new_index
        setattr(self, channel_attr, channel_df) # Update the class variable
    
    def update_cell_index_in_all_spot_channels(self, col, row, fov, time_point, previous_index, new_index):
        for channel in ['ch1_spot_df', 'ch2_spot_df', 'ch3_spot_df', 'ch4_spot_df', 'ch5_spot_df']:
            self.update_cell_index_in_channel(channel, col, row, fov, time_point, previous_index, new_index)

               
    def ON_APPLYBUTTON(self, Meta_Data_df):
        
        self.gui_params = Gui_Params(self.AnalysisGui,self.inout_resource_gui)
        seconds1 = time.time()
        
        while self.inout_resource_gui.Output_dir ==[]:
                
            self.inout_resource_gui.OUTPUT_FOLDER_LOADBTN()
        self.Meta_Data_df = Meta_Data_df
        path_list = os.path.split(self.Meta_Data_df["ImageName"][0])[0].split(r'/')
        self.experiment_name = path_list[path_list.__len__()-2]
        self.output_prefix  = path_list[path_list.__len__()-1]
        self.output_folder = os.path.join(self.inout_resource_gui.Output_dir,self.experiment_name)
        if os.path.isdir(self.output_folder) == False:
            os.mkdir(self.output_folder) 
            
        csv_config_folder = os.path.join(self.output_folder, 'configuration_files')
        if os.path.isdir(csv_config_folder) == False:
            os.mkdir(csv_config_folder) 
        self.config_file = os.path.join(csv_config_folder, 'analysis_configuration.csv')
        self.AnalysisGui.SAVE_CONFIGURATION(self.config_file, self.ImageAnalyzer)
        
        columns = np.unique(np.asarray(self.Meta_Data_df['column'], dtype=int))
        rows = np.unique(np.asarray(self.Meta_Data_df['row'], dtype=int))
        fovs = np.unique(np.asarray(self.Meta_Data_df['field_index'], dtype=int))

        time_points = np.unique(np.asarray(self.Meta_Data_df['time_point'], dtype=int))
        actionindices = np.unique(np.asarray(self.Meta_Data_df['action_index'], dtype=int))
        
        jobs_number=self.gui_params.NumCPUsSpinBox_value

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

        arg_len,_=func_args.shape
        start_pos=np.arange(0,arg_len,jobs_number)
        for st in start_pos:

            if st+jobs_number-1 >= arg_len:

                data_ind=np.arange(st, arg_len)
            else:
                data_ind=np.arange(st, st+jobs_number)

            processes=[]
            for ind1 in data_ind:
                process_args= np.array(func_args[ind1,:],dtype=int)
                processes.append(Process(target=self.BATCH_ANALYZER, args=process_args))
            # kick them off 
            for process in processes:
                process.start()
            # now wait for them to finish
            for process in processes:
                process.join()

        

        if self.gui_params.NucInfoChkBox_check_status == True:

            self.cell_df = pd.concat(self.cell_pd_list)
            self.SAVE_NUCLEI_INFORMATION(self.cell_df, columns, rows)
            

        # if self.ch1_spot_df_list.__len__() > 0:
        #     self.ch1_spot_df = pd.concat(self.ch1_spot_df_list)
        #     if self.gui_params.NucInfoChkBox_check_status == True:
        #         self.SAVE_SPOT_INFO(self.ch1_spot_df, self.gui_params.SpotLocationCbox_currentText, columns, rows, 'Ch1')
        
        # if self.ch2_spot_df_list.__len__() > 0:
        #     self.ch2_spot_df = pd.concat(self.ch2_spot_df_list)
        #     if self.gui_params.SpotsLocation_check_status == True:
        #         self.SAVE_SPOT_INFO(self.ch2_spot_df, self.gui_params.SpotLocationCbox_currentText, columns, rows, 'Ch2')

        # if self.ch3_spot_df_list.__len__() > 0:
        #     self.ch3_spot_df = pd.concat(self.ch3_spot_df_list)
        #     if self.gui_params.SpotsLocation_check_status == True:
        #         self.SAVE_SPOT_INFO(self.ch3_spot_df, self.gui_params.SpotLocationCbox_currentText, columns, rows, 'Ch3')

        # if self.ch4_spot_df_list.__len__() > 0:
        #     self.ch4_spot_df = pd.concat(self.ch4_spot_df_list)
        #     if self.gui_params.SpotsLocation_check_status == True:
        #         self.SAVE_SPOT_INFO(self.ch4_spot_df, self.gui_params.SpotLocationCbox_currentText, columns, rows, 'Ch4')

        # if self.ch5_spot_df_list.__len__() > 0:
        #     self.ch5_spot_df = pd.concat(self.ch5_spot_df_list)
        #     if self.gui_params.SpotsLocation_check_status == True:
        #         self.SAVE_SPOT_INFO(self.ch5_spot_df, self.gui_params.SpotLocationCbox_currentText, columns, rows, 'Ch5')
        self.PROCESS_ALL_SPOT_CHANNELS()
            # Calculate Spot Distances
    
        if self.gui_params.SpotsDistance_check_status == True:
            columns = np.unique(np.asarray(self.cell_df['column'], dtype=int))
            rows = np.unique(np.asarray(self.cell_df['row'], dtype=int))
            Parallel(n_jobs=jobs_number, prefer="threads")(delayed(self.Calculate_Spot_Distances)(row, col) for row in rows for col in columns)
     
        if self.gui_params.Cell_Tracking_check_status == True:
            
            xlsx_name = ['Nuclei_Information.csv']
            xlsx_full_name = os.path.join(os.path.join(self.output_folder,"whole_plate_resutls"), xlsx_name[0])
            self.cell_df = pd.read_csv(xlsx_full_name).drop(["Unnamed: 0"], axis=1)
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
#             fovs = np.array([2])
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
        
                            ImageForNucMask = self.RAW_IMAGE_LOADER(str(self.gui_params.NucleiChannel_index + 1))
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
                        if self.gui_params.NucTrackingMethod_currentText == "Bayesian":
                            
                            tracks_pd = self.RUN_BTRACK(label_stack, self.gui_params)
                        
                        if self.gui_params.NucTrackingMethod_currentText == "DeepCell":
                        
                            tracks_pd = self.deepcell_tracking(t_stack_nuc,label_stack,self.gui_params)
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
                            # if self.ch1_spot_df.empty==False:
                                
                            #     row_index_spotdf = self.ch1_spot_df.loc[(self.ch1_spot_df['field_index'] == fov)&(self.ch1_spot_df['column'] == col)&(self.ch1_spot_df['row'] == row)&
                            #                                             (self.ch1_spot_df["time_point"]==trck_row["time_point"])&(self.ch1_spot_df["cell_index"]==previous_index)]
                                
                            #     self.ch1_spot_df.loc[row_index_spotdf, 'cell_index'] = trck_row["cell_index"]
                            # Define the list of channel dataframes
                            # channel_dfs = [self.ch1_spot_df, self.ch2_spot_df, self.ch3_spot_df, self.ch4_spot_df, self.ch5_spot_df]
                            
                            # # Loop through each channel dataframe
                            # for channel_df in channel_dfs:
                            #     if not channel_df.empty:
                            #         row_index_spotdf = channel_df.loc[
                            #             (channel_df['field_index'] == fov) & (channel_df['column'] == col) & (channel_df['row'] == row) &
                            #             (channel_df["time_point"] == trck_row["time_point"]) & (channel_df["cell_index"] == previous_index)
                            #         ].index
                                    
                            #         channel_df.loc[row_index_spotdf, 'cell_index'] = trck_row["cell_index"]

                            
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
                        patch_size = (self.gui_params.patchsize_currentText,self.gui_params.patchsize_currentText)
                        IDs = tracks_pd['ID'].unique()

                        for id_ in IDs:
                            #### create a copy of each single track df containing all the nuclei informaion
                            single_track1 = tracks_pd.loc[tracks_pd['ID']==id_]
                            single_track = single_track1[single_track1['dummy']==False]
                            single_track_copy = single_track.copy().reset_index(drop=True)
                            
                            if len(single_track_copy)< self.gui_params.MintrackLengthSpinbox_current_value:
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
        
                                img_patch = self.zero_pad_patch(small_img_patch, desired_size=patch_size)
                                
                                bin_img = self.zero_pad_patch(masks_stack[int(pd_row['t']), min_row:max_row, min_col:max_col], desired_size=patch_size)
                                mask_patches.append(bin_img)
                                lbl_img, n_feat = label(bin_img)
                                label_center = int(self.gui_params.patchsize_currentText/2)
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
                                    if self.AnalysisGui.Registrationmethod.currentText() == "Phase Correlation":
                                        init_rotation = 15
                                        final_angle, rotated_img = self.phase_correlation(rotated_nuc[-1], img_patch1, intial_rotation = 15, 
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
                                    if len(patch_ch_spots[['x_location','y_location','z_location']]<self.gui_params.maxspotspercellSpinbox_current_value+1):
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


                                    only_spots_patch[str(int(chnl))] = self.zero_pad_patch(only_spots_patch[str(int(chnl))], 
                                                                                           desired_size=patch_size)

                                    col_name3='ch'+str(int(chnl))+'_patch_spots_locations'
                                    single_track_copy.loc[single_track_copy['t']==pd_row['t'], col_name3]=[np.where(only_spots_patch[str(int(chnl))]>0)]

                                    spot_patch = self.zero_pad_patch(spot_dict_stack[str(int(chnl))][ min_row:max_row, min_col:max_col, int(pd_row['t'])], 
                                                                     desired_size=patch_size)
                                    spot_patches[str(int(chnl))].append(spot_patch)

                                    temp_spots = []
                                    temp_transformed = []
                                    for i in range(len(np.where(only_spots_patch[str(int(chnl))]>0)[0])):

                                        temp_spots.append(np.array([np.where(only_spots_patch[str(int(chnl))]>0)[0][i], np.where(only_spots_patch[str(int(chnl))]>0)[1][i]]))
                                        temp_transformed.append(self.rotate_point((label_center-0.5,label_center-0.5),
                                                                             np.array([np.where(only_spots_patch[str(int(chnl))]>0)[0][i], 
                                                                                       np.where(only_spots_patch[str(int(chnl))]>0)[1][i]]), 
                                                                             (final_angle+15)*np.pi/180))

                                    if len(temp_spots)>0:
                                        transformed_spots[str(pd_row['t'])] = np.round(np.array(temp_transformed)).astype(int)
                                        spot_coor_dict[str(pd_row['t'])]= np.array(temp_spots)
                                        col_name4='ch'+str(int(chnl))+'_transformed_spots_locations'
                                        single_track_copy.loc[single_track_copy['t']==pd_row['t'], col_name4]=[transformed_spots[str(pd_row['t'])]]
            
                                    # only_spots_patch[str(int(chnl))] = self.zero_pad_patch(only_spots_patch[str(int(chnl))])
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
                                        clusters  = self.run_clustering(all_spots_for_gmm, outlier_threshold = 3, max_dist = self.gui_params.SpotSearchRadiusSpinbox_current_value)
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
                                        sp_bound= int(round(7*self.gui_params.PSFsizeSpinBox_current_value/2))
                                        all_spot_patches, spot_patches_center_coords = self.get_spot_patch(single_track_copy, chnl, lbl1, 
                                                                                                           rotated_spot_patches[str(int(chnl))], 
                                                                                                           spot_boundary = sp_bound)
                                        for jjj in single_track_copy.index.tolist():

                                            if jjj in list(all_spot_patches.keys()):
                                                spot_patch = all_spot_patches[jjj]
                                                spot_coords = spot_patches_center_coords[jjj]
                                                # print(spot_patch.shape)

                                                fit_results = self.ImageAnalyzer.gmask_fit(spot_patch, 
                                                                                           xy_input=np.array([np.where(spot_patch==spot_patch.max())[0][0],
                                                                                                              np.where(spot_patch==spot_patch.max())[1][0]]), 
                                                                                           fit=self.gui_params.IntegratedIntensity_fitStatus)
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
                                        
                                        if self.gui_params.FittingMethod_index==0:
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
                                                
                                        elif self.gui_params.FittingMethod_index==1:
                                            
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
        self.PROCESS_ALL_SPOT_CHANNELS()
        seconds2 = time.time()
        
        diff=seconds2-seconds1
        print('Total Processing Time (Minutes):',diff/60)
        
    def normalize_stack(self, images):

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
        
        if "spot" in name_prefix:
            file_name = name_prefix+ r'_for_col' + str(col) + r'_row' + str(row) + r'_field' + str(fov) + r'_cell' + str(cell_ind)+ r'_Ch' + str(int(spot_ch)) + name_extention
        else:
            file_name = name_prefix+ r'_for_col' + str(col) + r'_row' + str(row) + r'_field' + str(fov) + r'_cell' + str(cell_ind) + name_extention
        
        file_path = os.path.join(top_folder, folder_name)
        if os.path.isdir(file_path) == False:
            os.mkdir(file_path)
        
        return os.path.join(file_path, file_name)
    
    def spot_level_file_name(self, top_folder, folder_name, name_prefix, name_extention, col, row, fov, cell_ind, spot_ch, spot_ind):
        
        file_name = name_prefix+ r'_for_col' + str(col) + r'_row' + str(row) + r'_field' + str(fov) + r'_cell' + str(cell_ind)+ r'_Ch' + str(int(spot_ch))+ r'_spot' + str(spot_ind) + name_extention
        
        file_path = os.path.join(top_folder, folder_name)
        if os.path.isdir(file_path) == False:
            os.mkdir(file_path)
        
        return os.path.join(file_path, file_name)

    def BATCH_ANALYZER(self, col,row,fov,t): 
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
#                     self.cell_pd_list = pd.concat([self.cell_pd_list, image_cells_df], axis=0, ignore_index=True)
                    
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
                    
                    if (self.gui_params.SecChannel_current_index > 0) & (self.gui_params.SecArea_current_index > 0):
                        
                        secondary_image_df = self.df_checker.loc[(self.df_checker['channel'] == str(self.gui_params.SecChannel_current_index))]
                        secondary_image = self.ImageAnalyzer.max_z_project(secondary_image_df)

                        sec_props = regionprops_table(labeled_nuc, secondary_image, properties=('label','area', 'mean_intensity'))
                        sec_props_df = pd.DataFrame(sec_props)
                        sec_props_df['sum_intensity'] = sec_props_df['mean_intensity']*sec_props_df['area']
                        
                        col1_name = "ch" + str(self.gui_params.SecChannel_current_index) + "_mean_intensity"
                        image_cells_df[col1_name]=0
                        col2_name = "ch" + str(self.gui_params.SecChannel_current_index) + "_sum_intensity"
                        image_cells_df[col2_name]=0
                        
                        mapping_dict = sec_props_df.set_index('label')['mean_intensity'].to_dict()
                        image_cells_df[col1_name] = image_cells_df['label'].map(mapping_dict).fillna(image_cells_df[col1_name])

                        mapping_dict = sec_props_df.set_index('label')['sum_intensity'].to_dict()
                        image_cells_df[col2_name] = image_cells_df['label'].map(mapping_dict).fillna(image_cells_df[col2_name])
                        
                    self.cell_pd_list.append(image_cells_df)

                if self.gui_params.NucMaskCheckBox_status_check == True:
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

            ch1_xyz, ch1_xyz_3D, ch1_final_spots, ch2_xyz, ch2_xyz_3D, ch2_final_spots, ch3_xyz, ch3_xyz_3D, ch3_final_spots, ch4_xyz, ch4_xyz_3D, ch4_final_spots, ch5_xyz, ch5_xyz_3D, ch5_final_spots  = self.IMAGE_FOR_SPOT_DETECTION( ImageForNucMask)

            if self.gui_params.NucMaxZprojectCheckBox_status_check == True:

                if self.gui_params.SpotMaxZProject_status_check == True:

                        if self.gui_params.SpotCh1CheckBox_status_check == True:
                            if ch1_xyz.size > 0:
                                ch1_xyz_round = np.round(np.asarray(ch1_xyz)).astype('int')
                                ch1_spot_nuc_labels = labeled_nuc[ch1_xyz_round[:,0], ch1_xyz_round[:,1]]

                                ch1_num_spots = ch1_xyz.__len__()
                                if labeled_nuc.max()>0:
                                    ch1_radial = self.RADIAL_DIST_CALC(ch1_xyz_round,ch1_spot_nuc_labels,
                                                                       radial_dist_df, dist_img)
                                else:
                                    ch1_radial=np.nan
                                data = { "Experiment":[self.experiment_name]*ch1_num_spots,
                                         "column": [col]*ch1_num_spots, "row": [row]*ch1_num_spots, 
                                         "time_point": [t]*ch1_num_spots, "field_index": [fov]*ch1_num_spots,
                                         "z_slice": ["max_project"]*ch1_num_spots, "channel": [1]*ch1_num_spots,
                                         "action_index": [ai]*ch1_num_spots, "cell_index": ch1_spot_nuc_labels,
                                         "x_location": ch1_xyz[:,0], "y_location": ch1_xyz[:,1],
                                         "z_location": ch1_xyz[:,2], 
                                         "radial_distance":ch1_radial
                                    
                                       }
                                df_ch1 = pd.concat([pd.DataFrame(data), ch1_final_spots], axis=1)

                                self.ch1_spot_df_list.append(df_ch1)
    
                                    
                        if self.gui_params.SpotCh2CheckBox_status_check == True:
                            # if ch2_xyz!=[]:
                            if ch2_xyz.size > 0:
                                
                                ch2_xyz_round = np.round(np.asarray(ch2_xyz)).astype('int')
                                ch2_spot_nuc_labels = labeled_nuc[ch2_xyz_round[:,0], ch2_xyz_round[:,1]]
                                ch2_num_spots = ch2_xyz.__len__()
                                if labeled_nuc.max()>0:
                                    ch2_radial = self.RADIAL_DIST_CALC(ch2_xyz_round,ch2_spot_nuc_labels,
                                                                       radial_dist_df, dist_img)
                                else:
                                    ch2_radial=np.nan
                                data = { "Experiment":[self.experiment_name]*ch2_num_spots,
                                         "column": [col]*ch2_num_spots, "row": [row]*ch2_num_spots, 
                                         "time_point": [t]*ch2_num_spots, "field_index": [fov]*ch2_num_spots,
                                         "z_slice": ["max_project"]*ch2_num_spots, "channel": [2]*ch2_num_spots,
                                         "action_index": [ai]*ch2_num_spots, "cell_index": ch2_spot_nuc_labels,
                                         "x_location": ch2_xyz[:,0], "y_location": ch2_xyz[:,1],
                                         "z_location": ch2_xyz[:,2],
                                         "radial_distance":ch2_radial
                                       }
#                                 df_ch2 = pd.DataFrame(data)
                                df_ch2 = pd.concat([pd.DataFrame(data), ch2_final_spots], axis=1)
                                
                
                                self.ch2_spot_df_list.append(df_ch2)
    

                        if self.gui_params.SpotCh3CheckBox_status_check == True:
                            if ch3_xyz.size > 0:

                                ch3_xyz_round = np.round(np.asarray(ch3_xyz)).astype('int')
                                ch3_spot_nuc_labels = labeled_nuc[ch3_xyz_round[:,0], ch3_xyz_round[:,1]]
                                ch3_num_spots = ch3_xyz.__len__()
                                if labeled_nuc.max()>0:
                                    ch3_radial = self.RADIAL_DIST_CALC(ch3_xyz_round,ch3_spot_nuc_labels,
                                                                       radial_dist_df, dist_img)
                                else:
                                    ch3_radial=np.nan
                                data = { "Experiment":[self.experiment_name]*ch3_num_spots,
                                         "column": [col]*ch3_num_spots, "row": [row]*ch3_num_spots, 
                                         "time_point": [t]*ch3_num_spots, "field_index": [fov]*ch3_num_spots,
                                         "z_slice": ["max_project"]*ch3_num_spots, "channel": [3]*ch3_num_spots,
                                         "action_index": [ai]*ch3_num_spots, "cell_index": ch3_spot_nuc_labels,
                                         "x_location": ch3_xyz[:,0], "y_location": ch3_xyz[:,1],
                                         "z_location": ch3_xyz[:,2],
                                         "radial_distance": ch3_radial
                                       }

#                                 df_ch3 = pd.DataFrame(data)
                                df_ch3 = pd.concat([pd.DataFrame(data), ch3_final_spots], axis=1)
    
                                
                                self.ch3_spot_df_list.append(df_ch3)
    

                        if self.gui_params.SpotCh4CheckBox_status_check == True:
                            if ch4_xyz.size > 0:

                                ch4_xyz_round = np.round(np.asarray(ch4_xyz)).astype('int')
                                ch4_spot_nuc_labels = labeled_nuc[ch4_xyz_round[:,0], ch4_xyz_round[:,1]]
                                ch4_num_spots = ch4_xyz.__len__()
                                if labeled_nuc.max()>0:
                                    ch4_radial = self.RADIAL_DIST_CALC(ch4_xyz_round,ch4_spot_nuc_labels,
                                                                       radial_dist_df, dist_img)
                                else:
                                    ch4_radial=np.nan
                                data = { "Experiment":[self.experiment_name]*ch4_num_spots,
                                         "column": [col]*ch4_num_spots, "row": [row]*ch4_num_spots, 
                                         "time_point": [t]*ch4_num_spots, "field_index": [fov]*ch4_num_spots,
                                         "z_slice": ["max_project"]*ch4_num_spots, "channel": [4]*ch4_num_spots,
                                         "action_index": [ai]*ch4_num_spots, "cell_index": ch4_spot_nuc_labels,
                                         "x_location": ch4_xyz[:,0], "y_location": ch4_xyz[:,1],
                                         "z_location": ch4_xyz[:,2],
                                         "radial_distance":ch4_radial
                                       }

#                                 df_ch4 = pd.DataFrame(data)
                                df_ch4 = pd.concat([pd.DataFrame(data), ch4_final_spots], axis=1)

                               
                                self.ch4_spot_df_list.append(df_ch4)

                                    
                        if self.gui_params.SpotCh5CheckBox_status_check == True:
                            if ch5_xyz.size > 0:

                                ch5_xyz_round = np.round(np.asarray(ch5_xyz)).astype('int')
                                ch5_spot_nuc_labels = labeled_nuc[ch5_xyz_round[:,0], ch5_xyz_round[:,1]]
                                ch5_num_spots = ch5_xyz.__len__()
                                if labeled_nuc.max()>0:
                                    ch5_radial = self.RADIAL_DIST_CALC(ch5_xyz_round,ch5_spot_nuc_labels,
                                                                       radial_dist_df, dist_img)
                                else:
                                    ch5_radial=np.nan


                                data = { "Experiment":[self.experiment_name]*ch5_num_spots,
                                         "column": [col]*ch5_num_spots, "row": [row]*ch5_num_spots, 
                                         "time_point": [t]*ch5_num_spots, "field_index": [fov]*ch5_num_spots,
                                         "z_slice": ["max_project"]*ch5_num_spots, "channel": [5]*ch5_num_spots,
                                         "action_index": [ai]*ch5_num_spots, "cell_index": ch5_spot_nuc_labels,
                                         "x_location": ch5_xyz[:,0], "y_location": ch5_xyz[:,1],
                                         "z_location": ch5_xyz[:,2],
                                         "radial_distance": ch5_radial
                                       }

#                                 df_ch5 = pd.DataFrame(data)
                                df_ch5 = pd.concat([pd.DataFrame(data), ch5_final_spots], axis=1)

                               
                                self.ch5_spot_df_list.append(df_ch5)
                                

    def Calculate_Spot_Distances(self, row, col):
    
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
        
        if self.df_checker.empty == False:

            if self.gui_params.NucMaxZprojectCheckBox_status_check == True:
                maskchannel = str(self.gui_params.NucleiChannel_index + 1)
                imgformask = self.df_checker.loc[(self.df_checker['channel'] == maskchannel)]
                loadedimg_formask = self.ImageAnalyzer.max_z_project(imgformask)
                
            else:
                z_imglist=[]
                maskchannel = str(self.gui_params.NucleiChannel_index+1)
                imgformask = self.df_checker.loc[(self.df_checker['channel'] == maskchannel)]

                for index, row in imgformask.iterrows():
                    
                    if row['ImageName']=="dask_array":
                        # im = row["Type"].compute()
                        im = row["Type"]
                    else: 
                        im = mpimg.imread(row['ImageName'])
                    im_uint8 = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    z_imglist.append( np.asarray(im_uint8))
                loadedimg_formask = np.stack(z_imglist, axis=2)
        
        return loadedimg_formask
    
    def RAW_IMAGE_LOADER(self, maskchannel):
        
        if self.df_checker.empty == False:

            if self.gui_params.NucMaxZprojectCheckBox_status_check == True:

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
                        im = mpimg.imread(row['ImageName'])
                    z_imglist.append( np.asarray(im))
                loaded_image = np.stack(z_imglist, axis=2)
        
        return loaded_image

    def Z_STACK_NUC_SEGMENTER(self, ImageForNucMask):
        
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
        
        label_nuc_list = []
        w, h, z_plane = ImageForLabel.shape

        for z in range(z_plane):

            single_z_img = ImageForLabel[:,:,z]
            labeled_nuc, number_nuc = label(single_z_img)

            label_nuc_list.append( np.asarray(labeled_nuc))
            
        label_nuc_stack = np.stack(label_nuc_list, axis=2)
    
        return label_nuc_stack
    
    
    def IMAGE_FOR_SPOT_DETECTION(self, ImageForNucMask):
        
        ch1_xyz, ch2_xyz, ch3_xyz, ch4_xyz, ch5_xyz = [],[],[],[],[]
        ch1_xyz_3D, ch2_xyz_3D, ch3_xyz_3D, ch4_xyz_3D, ch5_xyz_3D = [],[],[],[],[]
        ch1_final_spots, ch2_final_spots, ch3_final_spots, ch4_final_spots, ch5_final_spots = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        if self.gui_params.SpotCh1CheckBox_status_check == True:
            
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '1')]
            ch1_xyz, ch1_xyz_3D, ch1_final_spots = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask, 'Ch1')
                
        if self.gui_params.SpotCh2CheckBox_status_check == True:
            
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '2')]
            ch2_xyz, ch2_xyz_3D, ch2_final_spots = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask, 'Ch2')
                
        if self.gui_params.SpotCh3CheckBox_status_check == True:
            
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '3')]
            ch3_xyz, ch3_xyz_3D, ch3_final_spots = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask, 'Ch3')
                    
        if self.gui_params.SpotCh4CheckBox_status_check == True:
             
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '4')]
            ch4_xyz, ch4_xyz_3D, ch4_final_spots = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask, 'Ch4')
        
        if self.gui_params.SpotCh5CheckBox_status_check == True:
             
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '5')]
            ch5_xyz, ch5_xyz_3D, ch5_final_spots = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask, 'Ch5')
            
        return ch1_xyz, ch1_xyz_3D, ch1_final_spots, ch2_xyz, ch2_xyz_3D, ch2_final_spots, ch3_xyz, ch3_xyz_3D, ch3_final_spots, ch4_xyz, ch4_xyz_3D, ch4_final_spots, ch5_xyz, ch5_xyz_3D, ch5_final_spots 

    
    def XYZ_SPOT_COORDINATES(self, images_pd_df, ImageForNucMask, spot_channel):
        
        z_imglist = []
        coordinates_stack = np.ones((0,3), dtype='float')
        xyz_coordinates = []        
            
        for index, row in images_pd_df.iterrows():
            
            if row['ImageName']=="dask_array":
                # im = row["Type"].compute()
                im = row["Type"]
            else: 
                im = mpimg.imread(row['ImageName'])
            z_imglist.append( np.asarray(im))
            _z_coordinates1 = np.asarray(row['z_slice']).astype('float')
            
            
            coordinates, final_spots = self.ImageAnalyzer.SpotDetector(im, self.AnalysisGui, ImageForNucMask, spot_channel)
            
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
            coordinates_max_project, final_spots = self.ImageAnalyzer.SpotDetector(max_project, self.AnalysisGui, ImageForNucMask, spot_channel)
        
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

    
    def RUN_BTRACK(self,label_stack, gui_params):
        
        obj_from_generator = btrack.utils.segmentation_to_objects(label_stack, properties = ('bbox','area', 'perimeter', 'major_axis_length','orientation',
                                                                                             'solidity','eccentricity' ))
        # initialise a tracker session using a context manager
        with btrack.BayesianTracker() as tracker:
            tracker = btrack.BayesianTracker()
                # configure the tracker using a config file
                
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # config_file = os.path.join(current_dir, 'BayesianTracker', 'models', 'cell_config.json')
            config_file = os.path.join(current_dir, 'cell_config.json')
            tracker.configure_from_file(config_file)
            if label_stack.shape[0] > 1250:
                tracker.update_method = BayesianUpdates.APPROXIMATE
            # tracker.configure_from_file('/data2/HiTIPS_hmm/HiTIPS/BayesianTracker/models/cell_config.json')
            tracker.max_search_radius = gui_params.NucSearchRadiusSpinbox_current_value

            # append the objects to be tracked
            tracker.append(obj_from_generator)

            # set the volume
        #     tracker.volume=((0, 1200), (0, 1200), (-1e5, 64.))

            # track them (in interactive mode)
            tracker.track_interactive(step_size=100)

            # generate hypotheses and run the global optimizer
            if label_stack.shape[0] < 1250:
                tracker.optimize()

#                             tracker.export(os.path.join('/data2/cell_tracking/','tracking.h5'), obj_type='obj_type_1')

#                             # get the tracks in a format for napari visualization
#                             data, properties, graph = tracker.to_napari(ndim=2)

            tracks = tracker.tracks

        tracks_pd = pd.DataFrame()
        for i in range(len(tracks)):
            tracks_pd = tracks_pd.append(pd.DataFrame(tracks[i].to_dict()))
        tracks_pd = tracks_pd[tracks_pd['dummy']==False]
        return tracks_pd
    
    def deepcell_tracking(self,t_stack_nuc,masks_stack, gui_params):
    
        tracker = CellTracking()
        tracked_data = tracker.track(np.copy(np.expand_dims(t_stack_nuc, axis=-1)), np.copy(np.expand_dims(masks_stack, axis=-1)))

        tracks_pd = pd.DataFrame()
        for i in range(len(tracked_data['y_tracked'])):
            labeled_nuc, number_nuc = label(np.squeeze(tracked_data['y_tracked'][i]))
            dist_props = regionprops_table(labeled_nuc, properties=('label','centroid', 'bbox','area', 'perimeter', 'major_axis_length','orientation', 'solidity','eccentricity'))
            single_image_df = pd.DataFrame(dist_props)
            single_image_df['t']=i*np.ones(len(single_image_df),dtype=int)
            single_image_df.rename(columns={ "label":"ID"}, inplace = True)
            single_image_df.rename(columns={ "centroid-0":"y"}, inplace = True)
            single_image_df.rename(columns={ "centroid-1":"x"}, inplace = True)
            single_image_df['z']=i*np.zeros(len(single_image_df),dtype=int)
            single_image_df['generation']=i*np.zeros(len(single_image_df),dtype=int)
            single_image_df['parent']=single_image_df['ID']
            single_image_df['root']=single_image_df['ID']
            new_columns=['ID','t','x','y','z','parent','root','generation', 'area', 'major_axis_length', 'solidity', 'bbox-1', 'bbox-2', 'bbox-3', 'eccentricity', 'bbox-0', 'orientation', 'perimeter']
            new_df = single_image_df[new_columns]
            tracks_pd = pd.concat([tracks_pd,new_df], axis=0)
        
        return tracks_pd
    
    def mutual_information(self, hgram):
     
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1) # marginal for x over y
        py = np.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    def zero_pad_patch(self, input_image, desired_size=(128,128)):
        
        padded_image1 = np.zeros(desired_size, dtype=float)
        w,h= input_image.shape
        if w>desired_size[0]:
            input_image = input_image[:desired_size[0],:]
        if h>desired_size[1]:
            input_image = input_image[:,:desired_size[1]]
        shift_w = int(desired_size[0]/2)
        shift_h = int(desired_size[1]/2)
        padded_image1[:w, :h] = input_image
        padded_image = np.roll(np.roll(padded_image1, int(shift_w-w/2),axis=0), int(shift_h-h/2), axis=1)    

        return padded_image
    
    def registration_features(self, ref_img, ref_mask, reg_img, reg_mask):
    
        csi1 = np.dot(np.ravel(ref_img),np.ravel(reg_img))/(np.linalg.norm(np.ravel(ref_img))*np.linalg.norm(np.ravel(reg_img)))

        hist_2d_1, x_edges, y_edges = np.histogram2d(reg_img.ravel(), ref_img.ravel(), bins=20)
        mi1 = self.mutual_information(hist_2d_1)

        ssim_ind1 = ssim(ref_img, reg_img, data_range=ref_img.max() - ref_img.min())

        mse = -mean_squared_error(ref_img, reg_img)

        # ct = contingency_table(ref_mask, reg_mask)
        # are = adapted_rand_error(image_true=ref_mask, image_test=reg_mask, table=ct)[0]
        # # vi1, vi2 = variation_of_information(image0=ref_mask, image1=reg_mask , table=ct)

        eps_are=0.000000001

        psnr = peak_signal_noise_ratio(ref_img.astype("uint8"),reg_img.astype("uint8"))

        return np.array([csi1, mi1, ssim_ind1, mse,  psnr])#,1/(are+eps_are), -vi1, -vi2])
    
    def rotation_register_img_prep(self, ref_img, reg_img, rotation_angle, median_disk_size=3):
    
        median_ref_img = median(ref_img*255/(ref_img.max()+0.0001), disk(median_disk_size))

        im2_rot = rotate(reg_img, rotation_angle, center=None, preserve_range=True)
        img2_rot = median(im2_rot*255/(im2_rot.max()+0.0001) ,disk(median_disk_size))

        im2_rot1 = self.zero_pad_patch(img2_rot, desired_size=patch_size)

        #### correct for translation
        rot2_translation = phase_cross_correlation(median_ref_img, im2_rot1, upsample_factor=1)[0]
        # rot2_translation = np.array([0,0])
        median_reg_img = ndimage.shift(im2_rot1, rot2_translation, order=0)

        return median_ref_img, median_reg_img

    def run_registration_rotation(self, angle):

        img1, im1_rot1 = self.rotation_register_img_prep(rotated_nuc[-1], img_patch1, angle, median_disk_size=3)

        return self.registration_features(img1, im1_rot1)

    def phase_correlation(self, ref_img, reg_img, intial_rotation = 15, rescale_factor = 5, nuc_length = 50):
        
        if ref_img.max()>0:    
            
            radius = nuc_length*rescale_factor
            ref_image = img_as_float(rescale(median(ref_img,disk(3)),rescale_factor))
            rotated = rotate(rescale(median(reg_img,disk(3)),rescale_factor), intial_rotation, preserve_range=True)
            image_polar = warp_polar(ref_image, radius=radius)
            rotated_polar = warp_polar(rotated, radius=radius)
            shifts, error, phasediff = phase_cross_correlation(image_polar, rotated_polar)
            final_angle = -shifts[0]
            rotated_img = rotate(rotate(reg_img, intial_rotation, order=0, preserve_range=True), final_angle, order=0, preserve_range=True)

        elif ref_img.max()==0:
            
            final_angle = 0
            rotated_img = rotate(reg_img, 0, order=0, preserve_range=True)
        
        return final_angle, rotated_img
    
    
#     def max_distance_condition(self, cluster, max_dist):
#         pairwise_distances = pdist(cluster)
#         return np.max(pairwise_distances) <= max_dist

#     def outlier_removal(self, points, threshold):
#         mean = np.mean(points, axis=0)
#         std_dev = np.std(points, axis=0)
#         not_outlier_index = np.all(np.abs(points - mean) <= threshold * std_dev, axis=1)
#         outlier_index =~not_outlier_index
#         return not_outlier_index, outlier_index

#     def run_clustering(self, points, outlier_threshold = 2, max_dist = 6):
#         # Remove outliers
#         not_outlier_index, outlier_index = self.outlier_removal(points, outlier_threshold)
#         filtered_points = points[not_outlier_index]

#         # Compute the distance matrix and perform hierarchical clustering
#         distance_matrix = pdist(filtered_points)
#         linkage_matrix = linkage(distance_matrix, method='single')
#         clusters = fcluster(linkage_matrix, max_dist, criterion='distance')
#         # Merge small clusters
#         clusters = self.merge_small_clusters(filtered_points, clusters)

#         return clusters, not_outlier_index, outlier_index 

#     def merge_small_clusters(self, points, labels):
        
#         if len(labels) < 2:
#             return labels

#         # Get cluster sizes
#         cluster_sizes = np.bincount(labels)

#         # Find the large clusters
#         large_clusters = np.where(cluster_sizes > 2)[0]

#         # Find the small clusters
#         small_clusters = np.where(cluster_sizes < 3)[0]

#         # Compute the centroids of the large clusters
#         large_cluster_centroids = np.array([points[labels == i].mean(axis=0) for i in large_clusters])

#         # For each small cluster, find the closest large cluster and merge them
#         for small_cluster in small_clusters:
#             # Get the indices of the data points in the small cluster
#             small_cluster_indices = np.where(labels == small_cluster)[0]

#             # Compute the centroid of the small cluster
#             small_cluster_centroid = points[small_cluster_indices].mean(axis=0)

#             # Calculate the distances from the small cluster centroid to the centroids of the large clusters
#             distances = cdist(small_cluster_centroid.reshape(1, -1), large_cluster_centroids)

#             # Find the closest large cluster
#             closest_large_cluster = large_clusters[np.argmin(distances)]

#             # Merge the small cluster with the closest large cluster
#             labels[small_cluster_indices] = closest_large_cluster

#         # Reassign cluster labels to maintain consecutive numbering
#         unique_labels, unique_inverse = np.unique(labels, return_inverse=True)
#         new_labels = np.arange(1, len(unique_labels) + 1)
#         label_mapping = dict(zip(unique_labels, new_labels))
#         return np.array([label_mapping[label] for label in labels])
    
    def rotate_point(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy
    
    def get_spot_patch(self, single_track_copy, chnl, lbl1, rot_spot_patches, spot_boundary = 4):
        small_spot_patches = {}
        spot_patches_center_coords = {}
        real_spots = single_track_copy[single_track_copy['ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_locations"].apply(lambda x: x != [])]['ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_locations"]
        real_spots_indices = np.array(real_spots.index)
        for i in np.array(single_track_copy.index):
            if i not in real_spots_indices:
                time_point = single_track_copy['t'].loc[i]
                ind1 = np.searchsorted(real_spots_indices, i)
                if ind1==0:
                    try:
                        row,col = real_spots.loc[real_spots_indices[0]][0][0]
                    except:
                        row,col = real_spots.loc[real_spots_indices[0]][0]
                        
                if ind1==len(real_spots_indices):
                    try:
                          row,col = real_spots.loc[real_spots_indices[-1]][0][0]            
                    except:
                          row,col = real_spots.loc[real_spots_indices[-1]][0]
                          
                else:
                    try: 
                          lower_row,lower_col = real_spots.loc[real_spots_indices[ind1-1]][0][0]
                    except:
                          lower_row,lower_col = real_spots.loc[real_spots_indices[ind1-1]][0]
                         
                    try:
                          uper_row,uper_col = real_spots.loc[real_spots_indices[ind1]][0][0]
                    except:
                          uper_row,uper_col = real_spots.loc[real_spots_indices[ind1]][0]
                          
                    row = int(np.round((lower_row+uper_row)/2))
                    col = int(np.round((lower_col+uper_col)/2))

            else:
                 
                try:
                    row,col = real_spots.loc[i][0][0]
                except:
                    row,col = real_spots.loc[i][0]
       
            small_spot_patches[i]=rot_spot_patches[int(i)][int(row-spot_boundary) : int(row+spot_boundary+1), int(col-spot_boundary) : int(col+spot_boundary+1)]
            spot_patches_center_coords[i] = np.array([row,col]).reshape((1,2))

        return small_spot_patches, spot_patches_center_coords
    
    def merge_small_clusters(self,points, labels, max_dist):
        if (len(labels) < 2)|(self.gui_params.minburstdurationSpinbox_current_value==1):
            return labels

        cluster_sizes = np.bincount(labels)

        large_clusters = np.where(cluster_sizes >= self.gui_params.minburstdurationSpinbox_current_value)[0]
        small_clusters = np.where((cluster_sizes > 0) & (cluster_sizes < self.gui_params.minburstdurationSpinbox_current_value))[0]

        large_cluster_points = points[np.isin(labels, large_clusters)]

        for small_cluster in small_clusters:
            small_cluster_indices = np.where(labels == small_cluster)[0]

            for idx in small_cluster_indices:
                distances = cdist(points[idx].reshape(1, -1), large_cluster_points)
                closest_large_cluster_point_index = np.argmin(distances)
                closest_large_cluster_point_distance = distances[0, closest_large_cluster_point_index]

                if closest_large_cluster_point_distance <= max_dist:
                    labels[idx] = labels[large_cluster_points[closest_large_cluster_point_index]]
                else:
                    labels[idx] = 0

        return labels


    def run_clustering(self, points, outlier_threshold=2, max_dist=6):
        distance_matrix = pdist(points)
        linkage_matrix = linkage(distance_matrix, method='single')
        initial_clusters = fcluster(linkage_matrix, max_dist, criterion='distance')

        unique_clusters = np.unique(initial_clusters)
        centroids = np.array([points[initial_clusters == cluster].mean(axis=0) for cluster in unique_clusters])

        cluster_std_devs = []
        for cluster, centroid in zip(unique_clusters, centroids):
            cluster_points = points[initial_clusters == cluster]
            distances = cdist(centroid.reshape(1, -1), cluster_points)
            cluster_std_devs.append(np.std(distances))

        outlier_index = np.zeros(points.shape[0], dtype=bool)
        for cluster, centroid, std_dev in zip(unique_clusters, centroids, cluster_std_devs):
            cluster_points = points[initial_clusters == cluster]
            distances = cdist(centroid.reshape(1, -1), cluster_points)
            is_outlier = distances > outlier_threshold * max_dist
            outlier_index[initial_clusters == cluster] = is_outlier.flatten()

        initial_clusters[outlier_index] = 0

        # Merge small clusters and outliers
        clusters = self.merge_small_clusters(points, initial_clusters, max_dist)


        return clusters