from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
import numpy as np
import time
import pandas as pd
import cv2
from PIL import Image
from skimage import exposure
from skimage.measure import regionprops, regionprops_table
from PIL import Image, ImageQt
import qimage2ndarray
from .Analysis import ImageAnalyzer
from .AnalysisGUI import analyzer
from .IO_ResourceGUI import InOut_resource
from scipy.ndimage import label
from skimage.color import label2rgb, gray2rgb
from skimage.io import imread
import time

class imagedisplayer(object):
    METADATA_DATAFRAME = pd.DataFrame()
    imgchannels = pd.DataFrame()
    grid_data = np.zeros(5, dtype = int)
    def __init__(self,centralwidget, gui_params,analysisgui):
        #super(self, analyzer).__init__(centralwidget)
        
        self._zoom = 0
        self._empty = True
        self.hist_max = {'Ch 1': 255, 'Ch 2': 255, 'Ch 3': 255, 'Ch 4': 255, 'Ch 5': 255}
        self.hist_min = {'Ch 1': 0, 'Ch 2': 0, 'Ch 3': 0, 'Ch 4': 0, 'Ch 5': 0}
        self.AnalysisGui = analysisgui
        self.gui_params = gui_params
        self.ImageAnalyzer = ImageAnalyzer(self.gui_params.params_dict)
        self.lookup_table_rgb = { "Fire": [255, 128, 0], "Grays": [128, 128, 128], "Ice": [128, 128, 255], "Red": [255, 0, 0], "Green": [0, 255, 0],
                                  "Blue": [0, 0, 255], "Cyan": [0, 255, 255], "Magenta": [255, 0, 255], "Yellow": [255, 255, 0], "Royal": [65, 105, 225],  
                                  "Orange": [255, 165, 0], "Spring": [0, 255, 127], "Violet": [238, 130, 238], "Pink": [255, 192, 203], "HotPink": [255, 105, 180],
                                  "Goldenrod": [218, 165, 32], "Rainbow": [127, 127, 127], "Ocean": [0, 127, 255], "Terrain": [139, 69, 19], "Neon": [255, 0, 102]}
        
        self.start_time = 0
        self.end_time = 0
    def display_initializer(self, out_df, displaygui, IO_GUI):

        # Initial setup
        channel_indices = [0, 1, 2, 3, 4]
        displaygui.setEnabled(True)
        unique_ch = np.unique(np.asarray(out_df['channel'], dtype=int))

        # Dictionary to map channels to their corresponding GUI elements
        channel_mapping = {
            1: ("Channel 1", "Ch1CheckBox", "Ch1maxproject", "SpotCh1CheckBox"),
            2: ("Channel 2", "Ch2CheckBox", "Ch2maxproject", "SpotCh2CheckBox"),
            3: ("Channel 3", "Ch3CheckBox", "Ch3maxproject", "SpotCh3CheckBox"),
            4: ("Channel 4", "Ch4CheckBox", "Ch4maxproject", "SpotCh4CheckBox"),
            5: ("Channel 5", "Ch5CheckBox", "Ch5maxproject", "SpotCh5CheckBox")
        }

        for ch, (text, checkbox, maxproject, spotcheckbox) in channel_mapping.items():
            if ch not in unique_ch:
                # Disable corresponding GUI elements
                getattr(displaygui, checkbox).setEnabled(False)
                getattr(displaygui, maxproject).setEnabled(False)
                index = self.AnalysisGui.NucleiChannel.findText(text)
                channel_indices.remove(ch - 1)
                self.AnalysisGui.NucleiChannel.model().item(index).setEnabled(False)
                displaygui.HistChannel.model().item(index).setEnabled(False)
                getattr(self.AnalysisGui, spotcheckbox).setEnabled(False)

        # Set current indices
        self.AnalysisGui.NucleiChannel.setCurrentIndex(int(channel_indices[0]))
        displaygui.HistChannel.setCurrentIndex(int(channel_indices[0]))

        # Set metadata dataframe
        self.METADATA_DATAFRAME = out_df

        displaybtn = IO_GUI.DisplayCheckBox
        if displaybtn.isChecked():
            # Image scroller and spinbox initialization
            numoffiles = len(np.asarray(out_df['column'], dtype=int))

            # Histogram Max Min initialization for slider and spinbox
            displaygui.MaxHistSlider.setRange(0, 255)
            displaygui.MaxHistSlider.setValue(255)

            displaygui.MinHistSlider.setRange(0, 255)
            displaygui.MinHistSlider.setValue(0)

    
    ##### Image histogram controller funcions            
      
    def MAX_HIST_SLIDER_UPDATE(self, displaygui):
            
            self.MaxSlider_ind = displaygui.MaxHistSlider.value()
            self.MinSlider_ind = displaygui.MinHistSlider.value()
            
            if self.MaxSlider_ind <= self.MinSlider_ind:
                
                displaygui.MinHistSlider.setValue(self.MaxSlider_ind)
            
            self.READ_IMAGE(displaygui, self.imgchannels)
            
           
    def MIN_HIST_SLIDER_UPDATE(self, displaygui):
            
            self.MinSlider_ind = displaygui.MinHistSlider.value()
            self.MaxSlider_ind = displaygui.MaxHistSlider.value()
            
            if self.MinSlider_ind >= self.MaxSlider_ind:
                
                displaygui.MaxHistSlider.setValue(self.MinSlider_ind)
                
            
            self.READ_IMAGE(displaygui, self.imgchannels)
            
    def GET_IMAGE_NAME(self,displaygui):

            self.start_time = time.time()
            self.imgchannels = self.METADATA_DATAFRAME.loc[
                                    (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                    (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                    (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                    (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) & 
                                    (self.METADATA_DATAFRAME['z_slice'] == str(self.grid_data[4]))
                                        ]
            self.READ_IMAGE(displaygui, self.imgchannels)            
    
    def READ_IMAGE(self, displaygui, image_channels):
        self.height = 0
        self.width = 0

        all_channels_dict = {}
        # Define a list of channels and their associated attributes
        channels = [
                    {"index": "1", "checkbox": displaygui.Ch1CheckBox, "maxproject": displaygui.Ch1maxproject, "color": "Ch1"},
                    {"index": "2", "checkbox": displaygui.Ch2CheckBox, "maxproject": displaygui.Ch2maxproject, "color": "Ch2"},
                    {"index": "3", "checkbox": displaygui.Ch3CheckBox, "maxproject": displaygui.Ch3maxproject, "color": "Ch3"},
                    {"index": "4", "checkbox": displaygui.Ch4CheckBox, "maxproject": displaygui.Ch4maxproject, "color": "Ch4"},
                    {"index": "5", "checkbox": displaygui.Ch5CheckBox, "maxproject": displaygui.Ch5maxproject, "color": "Ch5"},
                ]

        if not self.imgchannels.empty:
            for channel in channels:
                if channel["checkbox"].isChecked():
                    ch_name = self.imgchannels.loc[self.imgchannels['channel'] == channel["index"]]['ImageName'].iloc[0]

                    if ch_name:
                        if channel["maxproject"].isChecked():
                            zstack_attribute = f"ch{channel['index']}_zstack"
                            setattr(self, zstack_attribute, self.METADATA_DATAFRAME.loc[
                                (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) &
                                (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) &
                                (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) &
                                (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &
                                (self.METADATA_DATAFRAME['channel'] == channel["index"])
                            ])
                            ch_img = self.ImageAnalyzer.max_z_project(getattr(self, zstack_attribute))
                        else:
                            if ch_name == "dask_array":
                                ch_img = self.imgchannels.loc[self.imgchannels['channel'] == channel["index"]]["Type"].iloc[0]
                            else:
                                ch_img = imread(ch_name)

                        self.height, self.width = np.shape(ch_img)
                        ch_img_key = f"ch{channel['index']}_img"
                        all_channels_dict[ch_img_key] = self.apply_lut_color(displaygui, ch_img, displaygui.channel_colors[channel["color"]])

        if self.height or self.width:
            self.ADJUST_IMAGE_CONTRAST(displaygui, all_channels_dict)
            
    def ADJUST_IMAGE_CONTRAST(self, displaygui, all_channels_dict):
        self.lower = displaygui.MinHistSlider.value()
        self.upper = displaygui.MaxHistSlider.value()

        channels = [
            {"hist_channel_name": "Ch 1", "img_key": "ch1_img"},
            {"hist_channel_name": "Ch 2", "img_key": "ch2_img"},
            {"hist_channel_name": "Ch 3", "img_key": "ch3_img"},
            {"hist_channel_name": "Ch 4", "img_key": "ch4_img"},
            {"hist_channel_name": "Ch 5", "img_key": "ch5_img"},
        ]

        for channel in channels:
            if displaygui.HistChannel.currentText() == channel["hist_channel_name"]:
                self.hist_max[channel["hist_channel_name"]] = self.upper
                self.hist_min[channel["hist_channel_name"]] = self.lower

            if channel["img_key"] in all_channels_dict.keys():
                all_channels_dict[channel["img_key"]] = self.ON_ADJUST_INTENSITY(
                    all_channels_dict[channel["img_key"]],
                    self.hist_min[channel["hist_channel_name"]],
                    self.hist_max[channel["hist_channel_name"]]
                )

        self.MERGEIAMGES(displaygui, all_channels_dict)


    def MERGEIAMGES(self,displaygui, all_channels_dict):
        

        channel_keys = list(all_channels_dict.keys())
    
        # Start with the first image as the base
        All_Channels = all_channels_dict[channel_keys[0]]

        for key in channel_keys[1:]:
            # Use addWeighted to combine base_image and the current image
            All_Channels = cv2.addWeighted(All_Channels, 1, all_channels_dict[key], 1, 0)


        height, width, ch = np.shape(All_Channels)
        totalBytes = All_Channels.nbytes

        if displaygui.NuclMaskCheckBox.isChecked() == True:

            self.input_image = self.IMAGE_TO_BE_MASKED()
            self.gui_params.update_values()
            bound, filled_res = self.ImageAnalyzer.neuceli_segmenter(self.input_image,
                                                                     self.METADATA_DATAFRAME["PixPerMic"].iloc[0])

            if displaygui.NucPreviewMethod.currentText() == "Boundary":

                All_Channels[bound != 0] = displaygui.channel_colors['Nuclei']

            if displaygui.NucPreviewMethod.currentText() == "Area":

                labeled_array, num_features = label(filled_res)
                rgblabel = label2rgb(labeled_array,alpha=0.1, bg_label = 0)
                rgblabel = cv2.normalize(rgblabel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                image_input_stack = np.stack((self.input_image,)*3, axis=-1)
                All_Channels = cv2.addWeighted(rgblabel,0.2, All_Channels, 1, 0)
                ##############

            if displaygui.NucPreviewMethod.currentText() == "Nuc.Index":

                All_Channels[bound != 0] = [255,0,0]
                labeled_array, num_features = label(filled_res)
                props = regionprops_table(labeled_array, properties=('label', 'centroid','area' ))
                props_df = pd.DataFrame(props)
                props_df1=props_df[props_df['area']>5]

                font = cv2.FONT_HERSHEY_SIMPLEX

                fontScale              = int(1)
                fontColor              = (255,0,255)
                lineType               = int(2)
                if labeled_array.max()>0:
                    for row_ind, row in props_df1.iterrows():

                        txt=str(row["label"])
                        bottomLeftCornerOfText = (int(round(row["centroid-1"])), int(round(row["centroid-0"])))
                        cv2.putText(All_Channels,txt, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        if displaygui.SpotsCheckBox.isChecked() == True:

            self.input_image = self.IMAGE_TO_BE_MASKED()
            
            if 'filled_res' in locals() or 'filled_res' in globals():
                pass
            else:
                filled_res = None
                
            ch1_spots_img, ch2_spots_img, ch3_spots_img, ch4_spots_img, ch5_spots_img = self.IMAGE_FOR_SPOT_DETECTION(self.input_image, displaygui, filled_res)


            if ch1_spots_img.size != 0:

                All_Channels[ch1_spots_img != 0] = displaygui.channel_colors['Ch1_spot']

            if ch2_spots_img.size != 0:

                All_Channels[ch2_spots_img != 0] = displaygui.channel_colors['Ch2_spot']

            if ch3_spots_img.size != 0:

                All_Channels[ch3_spots_img != 0] = displaygui.channel_colors['Ch3_spot']

            if ch4_spots_img.size != 0:

                All_Channels[ch4_spots_img != 0] = displaygui.channel_colors['Ch4_spot']

            if ch5_spots_img.size != 0:

                All_Channels[ch5_spots_img != 0] = displaygui.channel_colors['Ch5_spot']

            if displaygui.NucPreviewMethod.currentText() == "Cross":

                pass


        self.SHOWIMAGE(displaygui, All_Channels, width, height, totalBytes)
            
    def apply_lut_color(self, displaygui, img, rgb_values):
        
        # color_name_string = displaygui.inverse_lookup.get(tuple(color_name))
        # rgb_values = displaygui.lookup_table_rgb[color_name]
        colored_img = gray2rgb(img)/img.max()
        colored_img = colored_img * np.array(rgb_values)
        
       
        return colored_img.astype(np.uint8)
        
    def SHOWIMAGE(self, displaygui, img, width, height, totalBytes):
            
                        
            displaygui.viewer.setPhoto(QtGui.QPixmap.fromImage(qimage2ndarray.array2qimage(img)))
            self.end_time = time.time()
            print("image was processed in " + str(self.end_time - self.start_time) + "   seconds...")
    def IMAGE_TO_BE_MASKED(self):
        
        if self.AnalysisGui.NucMaxZprojectCheckBox.isChecked() == True:
            self.gui_params.update_values()
            maskchannel = str(self.gui_params.NucleiChannel_index+1)
            self.imgformask = self.METADATA_DATAFRAME.loc[
                                    (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                    (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                    (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                    (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &  
                                    (self.METADATA_DATAFRAME['channel'] == maskchannel)
                                    ]
            loadedimg_formask = self.ImageAnalyzer.max_z_project(self.imgformask)
            ImageForNucMask = cv2.normalize(loadedimg_formask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # ImageForNucMask = (255*loadedimg_formask/loadedimg_formask.max()).astype("uint8")

        else:
            
            maskchannel = str(self.gui_params.NucleiChannel_index+1)
            
            mask_img_name = self.imgchannels.loc[self.imgchannels['channel']== maskchannel]['ImageName'].iloc[0]
            if mask_img_name=="dask_array":
                # loadedimg_formask=self.imgchannels.loc[self.imgchannels['channel']==maskchannel]["Type"].iloc[0].compute()
                loadedimg_formask=self.imgchannels.loc[self.imgchannels['channel']==maskchannel]["Type"].iloc[0]
            else:
            
                loadedimg_formask = imread(mask_img_name)
            
            ImageForNucMask = cv2.normalize(loadedimg_formask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return ImageForNucMask
       
    def IMAGE_FOR_SPOT_DETECTION(self, nuclei_image, displaygui, nuc_mask=None):
        # Store results in a dictionary for each channel
        spots_images = {'Ch1': np.array([]), 'Ch2': np.array([]), 'Ch3': np.array([]), 'Ch4': np.array([]), 'Ch5': np.array([])}

        for channel_num in range(1, 6):  # Loop over channels 1 through 5
            ch_str = f"Ch{channel_num}"
            checkbox = getattr(self.AnalysisGui, f"Spot{ch_str}CheckBox")

            if checkbox.isChecked():
                if self.AnalysisGui.SpotMaxZProject.isChecked():
                    self.imgforspot = self.METADATA_DATAFRAME.loc[
                        (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) &
                        (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) &
                        (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) &
                        (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &
                        (self.METADATA_DATAFRAME['channel'] == str(channel_num))
                    ]
                    loadedimg_forspot = self.ImageAnalyzer.max_z_project(self.imgforspot)
                else:
                    spot_img_name = self.imgchannels.loc[self.imgchannels['channel'] == str(channel_num)]['ImageName'].iloc[0]
                    if spot_img_name == "dask_array":
                        loadedimg_forspot = self.imgchannels.loc[self.imgchannels['channel'] == str(channel_num)]["Type"].iloc[0]
                    else:
                        loadedimg_forspot = imread(spot_img_name)

                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                
                detection_methods = ["Laplacian of Gaussian", "Gaussian", "Intensity Threshold", "Enhanced LOG"] 
                threshold_methods = ["Auto", "Manual"]
                
                self.gui_params.update_values()
                if self.gui_params.spotchannelselect_currentText =='All':
                    
                    params_to_pass= self.gui_params.spot_params_dict['Ch1']
                else:
                    params_to_pass= self.gui_params.spot_params_dict[str(ch_str)]
                
                
                coordinates, final_spots = self.ImageAnalyzer.SpotDetector(
                                                                            input_image_raw=loadedimg_forspot, 
                                                                            nuc_mask=nuc_mask, 
                                                                            spot_detection_method= detection_methods[params_to_pass[0]],
                                                                            threshold_method= threshold_methods[params_to_pass[1]],
                                                                            threshold_value= params_to_pass[2],
                                                                            kernel_size= params_to_pass[3],
                                                                            spot_location_coords= self.gui_params.SpotLocationCbox_currentText,
                                                                            remove_bright_junk= self.gui_params.RemoveBrightJunk_status_check,
                                                                            resize_factor= self.gui_params.Resize_Factor,
                                                                            min_area=self.gui_params.SpotareaminSpinBox_value,
                                                                            max_area= self.gui_params.SpotareamaxSpinBox_value,
                                                                            min_integrated_intensity= self.gui_params.SpotIntegratedIntensitySpinBox_value,
                                                                            psf_size= self.gui_params.PSFsizeSpinBox_value,
                                                                            gaussian_fit= self.gui_params.IntegratedIntensity_fitStatus
                                                                        )

                if displaygui.spotPreviewMethod.currentText() == "Circle":
                    spots_images[ch_str] = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'), ImageForSpots)
                else:
                    spots_images[ch_str] = self.ImageAnalyzer.SPOTS_TO_BOUNDARY(final_spots)

        return spots_images['Ch1'], spots_images['Ch2'], spots_images['Ch3'], spots_images['Ch4'], spots_images['Ch5']


    def ON_ADJUST_INTENSITY(self, input_img, min_range, max_range):
        eplsion = 0.005
        mid_img = np.zeros((input_img.shape),dtype='uint16')
        input_img[input_img < min_range] = 0
        mid_img = input_img.astype('uint16')
        
        scale_factor = 255/(max_range + eplsion)
        mid_img = np.round(mid_img * scale_factor).astype('uint16')
        
        mid_img[mid_img > 255] = 255
        output_img = mid_img.astype('uint8')
        
        return output_img
    