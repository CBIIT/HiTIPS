from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
import numpy as np
import time
import pandas as pd
import matplotlib.image as mpimg
import cv2
from PIL import Image
from skimage import exposure
from skimage.measure import regionprops, regionprops_table
from PIL import Image, ImageQt
import qimage2ndarray
from Analysis import ImageAnalyzer
from AnalysisGUI import analyzer
from IO_ResourceGUI import InOut_resource
from scipy.ndimage import label
from skimage.color import label2rgb, gray2rgb

class imagedisplayer(object):
    METADATA_DATAFRAME = pd.DataFrame()
    imgchannels = pd.DataFrame()
    grid_data = np.zeros(5, dtype = int)
    def __init__(self,centralwidget, inout_resource_gui,analysisgui):
        #super(self, analyzer).__init__(centralwidget)
        
        self._zoom = 0
        self._empty = True
        
        self.ch1_hist_max = 255
        self.ch1_hist_min = 0
        self.ch2_hist_max = 255
        self.ch2_hist_min = 0
        self.ch3_hist_max = 255
        self.ch3_hist_min = 0
        self.ch4_hist_max = 255
        self.ch4_hist_min = 0
        self.AnalysisGui = analysisgui
        self.inout_resource_gui = inout_resource_gui
        self.ImageAnalyzer = ImageAnalyzer(self.AnalysisGui, self.inout_resource_gui)
    def display_initializer(self, out_df, displaygui, IO_GUI):
            channel_indices = [0,1,2,3,4]
            
            displaygui.setEnabled(True)
            unique_ch = np.unique(np.asarray(out_df['channel'], dtype=int))
            if 1 not in unique_ch:
                displaygui.Ch1CheckBox.setEnabled(False)
                displaygui.Ch1maxproject.setEnabled(False)
                index = self.AnalysisGui.NucleiChannel.findText("Channel 1")  # find the index of text
                channel_indices.remove(0)
                self.AnalysisGui.NucleiChannel.model().item(index).setEnabled(False)
                displaygui.HistChannel.model().item(index).setEnabled(False)
                self.AnalysisGui.SpotCh1CheckBox.setEnabled(False)
            if 2 not in unique_ch:
                displaygui.Ch2CheckBox.setEnabled(False)
                displaygui.Ch2maxproject.setEnabled(False)
                index = self.AnalysisGui.NucleiChannel.findText("Channel 2")  # find the index of text
                channel_indices.remove(1)
                self.AnalysisGui.NucleiChannel.model().item(index).setEnabled(False)
                displaygui.HistChannel.model().item(index).setEnabled(False)
                self.AnalysisGui.SpotCh2CheckBox.setEnabled(False)
            if 3 not in unique_ch:
                displaygui.Ch3CheckBox.setEnabled(False)
                displaygui.Ch3maxproject.setEnabled(False)
                index = self.AnalysisGui.NucleiChannel.findText("Channel 3")  # find the index of text
                channel_indices.remove(2)
                self.AnalysisGui.NucleiChannel.model().item(index).setEnabled(False)
                displaygui.HistChannel.model().item(index).setEnabled(False)
                self.AnalysisGui.SpotCh3CheckBox.setEnabled(False)
            if 4 not in unique_ch:
                displaygui.Ch4CheckBox.setEnabled(False)
                displaygui.Ch4maxproject.setEnabled(False)
                index = self.AnalysisGui.NucleiChannel.findText("Channel 4")  # find the index of text
                channel_indices.remove(3)
                self.AnalysisGui.NucleiChannel.model().item(index).setEnabled(False)
                displaygui.HistChannel.model().item(index).setEnabled(False)
                self.AnalysisGui.SpotCh4CheckBox.setEnabled(False)
            if 5 not in unique_ch:
                displaygui.Ch5CheckBox.setEnabled(False)
                displaygui.Ch5maxproject.setEnabled(False)
                index = self.AnalysisGui.NucleiChannel.findText("Channel 5")  # find the index of text
                channel_indices.remove(4)
                self.AnalysisGui.NucleiChannel.model().item(index).setEnabled(False)
                displaygui.HistChannel.model().item(index).setEnabled(False)
                self.AnalysisGui.SpotCh5CheckBox.setEnabled(False)
                
            self.AnalysisGui.NucleiChannel.setCurrentIndex(int(channel_indices[0]))
            displaygui.HistChannel.setCurrentIndex(int(channel_indices[0]))
            self.METADATA_DATAFRAME = out_df
            
            displaybtn = IO_GUI.DisplayCheckBox
            if displaybtn.isChecked() == True:
            # Image scroller and spinbox initialization
                numoffiles = np.asarray(out_df['column'], dtype=int).__len__()
            
            
                # Histogram Max Min initialization for slider and spinbox

                displaygui.MaxHistSlider.setMaximum(255)
                displaygui.MaxHistSlider.setMinimum(0)
                displaygui.MaxHistSlider.setValue(255)

                displaygui.MinHistSlider.setMaximum(255)
                displaygui.MinHistSlider.setMinimum(0)
                displaygui.MinHistSlider.setValue(0)
                
            else:
                pass

    
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
            
            # print(self.grid_data)
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

            if 'ch1_img' in locals():
                del ch1_img
            if 'ch2_img' in locals():
                del ch2_img
            if 'ch3_img' in locals():
                del ch3_img
            if 'ch4_img' in locals():
                del ch4_img
            if 'ch5_img' in locals():
                del ch5_img
            if 'All_Channels' in locals():
                del All_Channels
                
            if self.imgchannels.empty == False:
                
                if displaygui.Ch1CheckBox.isChecked() == True:
                    ch1_name = self.imgchannels.loc[self.imgchannels['channel']=='1']['ImageName'].iloc[0]
                    if ch1_name:
                        if displaygui.Ch1maxproject.isChecked() == True:
                            self.ch1_zstack = self.METADATA_DATAFRAME.loc[
                                                    (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                                    (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                                    (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                                    (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &  
                                                    (self.METADATA_DATAFRAME['channel'] == '1')
                                                    ]
                            ch1_img = self.ImageAnalyzer.max_z_project(self.ch1_zstack)
                        else:
                            if ch1_name=="dask_array":
                                # ch1_img=self.imgchannels.loc[self.imgchannels['channel']=='1']["Type"].iloc[0].compute()
                                ch1_img=self.imgchannels.loc[self.imgchannels['channel']=='1']["Type"].iloc[0]
                            else:
                                ch1_img = mpimg.imread(ch1_name)
                        ch1_img = (ch1_img/(ch1_img.max()/250)).astype('uint8')
                        self.CH1_img = ch1_img
                        self.height, self.width = np.shape(ch1_img)

                if displaygui.Ch2CheckBox.isChecked() == True:

                    ch2_name =self.imgchannels.loc[self.imgchannels['channel']=='2']['ImageName'].iloc[0]
                    if ch2_name:
                        if displaygui.Ch2maxproject.isChecked() == True:
                            self.ch2_zstack = self.METADATA_DATAFRAME.loc[
                                                    (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                                    (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                                    (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                                    (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &  
                                                    (self.METADATA_DATAFRAME['channel'] == '2')
                                                    ]
                            ch2_img = self.ImageAnalyzer.max_z_project(self.ch2_zstack)
                        else:
                            if ch2_name=="dask_array":
                                # ch2_img=self.imgchannels.loc[self.imgchannels['channel']=='2']["Type"].iloc[0].compute()
                                ch2_img=self.imgchannels.loc[self.imgchannels['channel']=='2']["Type"].iloc[0]
                            else:
                                ch2_img = mpimg.imread(ch2_name)
                        ch2_img = (ch2_img/(ch2_img.max()/250)).astype('uint8')
                        self.height, self.width = np.shape(ch2_img)

                if displaygui.Ch3CheckBox.isChecked() == True:

                    ch3_name = self.imgchannels.loc[self.imgchannels['channel']=='3']['ImageName'].iloc[0]
                    if ch3_name:
                        if displaygui.Ch3maxproject.isChecked() == True:
                            self.ch3_zstack = self.METADATA_DATAFRAME.loc[
                                                    (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                                    (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                                    (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                                    (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &  
                                                    (self.METADATA_DATAFRAME['channel'] == '3')
                                                    ]
                            ch3_img = self.ImageAnalyzer.max_z_project(self.ch3_zstack)
                        else:
                            if ch3_name=="dask_array":
                                # ch3_img=self.imgchannels.loc[self.imgchannels['channel']=='3']["Type"].iloc[0].compute()
                                ch3_img=self.imgchannels.loc[self.imgchannels['channel']=='3']["Type"].iloc[0]
                            else:
                                ch3_img = mpimg.imread(ch3_name)
                        ch3_img = (ch3_img/(ch3_img.max()/250)).astype('uint8')
                        self.height, self.width = np.shape(ch3_img)

                if displaygui.Ch4CheckBox.isChecked() == True:
                    ch4_name = self.imgchannels.loc[self.imgchannels['channel']=='4']['ImageName'].iloc[0]
                    if ch4_name:
                        if displaygui.Ch4maxproject.isChecked() == True:
                            self.ch4_zstack = self.METADATA_DATAFRAME.loc[
                                                    (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                                    (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                                    (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                                    (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &  
                                                    (self.METADATA_DATAFRAME['channel'] == '4')
                                                    ]
                            ch4_img = self.ImageAnalyzer.max_z_project(self.ch4_zstack)
                        else:
                            if ch4_name=="dask_array":
                                # ch4_img=self.imgchannels.loc[self.imgchannels['channel']=='4']["Type"].iloc[0].compute()
                                ch4_img=self.imgchannels.loc[self.imgchannels['channel']=='4']["Type"].iloc[0]
                            else:
                                ch4_img = mpimg.imread(ch4_name)
                        ch4_img = (ch4_img/(ch4_img.max()/250)).astype('uint8')
                        self.height, self.width = np.shape(ch4_img)

                if displaygui.Ch5CheckBox.isChecked() == True:

                    ch5_name = self.imgchannels.loc[self.imgchannels['channel']=='5']['ImageName'].iloc[0]
                    if ch5_name:
                        if displaygui.Ch5maxproject.isChecked() == True:
                            self.ch5_zstack = self.METADATA_DATAFRAME.loc[
                                                    (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                                    (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                                    (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                                    (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &  
                                                    (self.METADATA_DATAFRAME['channel'] == '5')
                                                    ]
                            self.ch5_img = self.ImageAnalyzer.max_z_project(self.ch5_zstack)
                        else:
                            if ch5_name=="dask_array":
                                # self.ch5_img=self.imgchannels.loc[self.imgchannels['channel']=='5']["Type"].iloc[0].compute()
                                self.ch5_img=self.imgchannels.loc[self.imgchannels['channel']=='5']["Type"].iloc[0]
                            else:
                                self.ch5_img = mpimg.imread(ch5_name)
                        self.ch5_img = (self.ch5_img/(self.ch5_img.max()/250)).astype('uint8')
                        self.height, self.width = np.shape(self.ch5_img)
                else:
                    
                    self.ch5_img = np.zeros((self.height, self.width))
                if self.height or self.width:

                    self.RGB_channels = np.zeros((self.height, self.width, 3))
                    if 'ch1_img' not in locals():
                        self.CH1_img = np.zeros((self.height, self.width, 3))
                    if 'ch2_img' in locals():
                        self.RGB_channels[:,:,1] = ch2_img
                    if 'ch3_img' in locals():
                        self.RGB_channels[:,:,0] = ch3_img
                    if 'ch4_img' in locals():
                        self.RGB_channels[:,:,2] = ch4_img
#                     if 'ch5_img' in locals():
#                         self.RGB_channels[:,:,0] = ch5_img
#                         self.RGB_channels[:,:,1] = 0.647 * ch5_img
                    h,w,c = self.RGB_channels.shape
                    
                    self.ADJUST_IMAGE_CONTRAST(displaygui, self.CH1_img, self.RGB_channels,self.ch5_img )

            
    def ADJUST_IMAGE_CONTRAST(self, displaygui, CH1 , RGB_CHANNELS, CH5):
            
            RGB_Channels = RGB_CHANNELS.astype(np.uint8)
            self.lower = displaygui.MinHistSlider.value()
            self.upper = displaygui.MaxHistSlider.value()

            if displaygui.HistChannel.currentText() == "Ch 1":
                
                self.ch1_hist_max = self.upper
                self.ch1_hist_min = self.lower
                
            if 'RGB_CHANNELS' in locals():  
            
                
                if displaygui.HistChannel.currentText() == "Ch 2":
                    
                    self.ch2_hist_max = self.upper
                    self.ch2_hist_min = self.lower
                    
                if displaygui.HistChannel.currentText() == "Ch 3":
                    
                    self.ch3_hist_max = self.upper
                    self.ch3_hist_min = self.lower
                
                if displaygui.HistChannel.currentText() == "Ch 4":
                    
                    self.ch4_hist_max = self.upper
                    self.ch4_hist_min = self.lower
                    
            if displaygui.HistChannel.currentText() == "Ch 5":

                self.ch5_hist_max = self.upper
                self.ch5_hist_min = self.lower
                CH5 = self.ON_ADJUST_INTENSITY(CH5, self.ch5_hist_min, self.ch5_hist_max)    
            CH1 = self.ON_ADJUST_INTENSITY(CH1, self.ch1_hist_min, self.ch1_hist_max)
            RGB_Channels[:,:,1] = self.ON_ADJUST_INTENSITY(RGB_Channels[:,:,1], self.ch2_hist_min, self.ch2_hist_max)
            RGB_Channels[:,:,0] = self.ON_ADJUST_INTENSITY(RGB_Channels[:,:,0], self.ch3_hist_min, self.ch3_hist_max)
            RGB_Channels[:,:,2] = self.ON_ADJUST_INTENSITY(RGB_Channels[:,:,2], self.ch4_hist_min, self.ch4_hist_max)
                
            self.MERGEIAMGES(displaygui, CH1, RGB_Channels, CH5)
            
    def MERGEIAMGES(self,displaygui, CH1, RGB_Channels, CH5):
        
            if displaygui.Ch1CheckBox.isChecked() == True:
                ch1_rgb = np.stack((CH1,)*3, axis=-1)
            else:
                ch1_rgb = np.zeros(RGB_Channels.shape, dtype = np.uint8)
            All_Channels = cv2.addWeighted(ch1_rgb, 1, RGB_Channels, 1, 0)
            if displaygui.Ch5CheckBox.isChecked() == True:
                ch5_rgb = np.stack((CH5,)*3, axis=-1)
                ch5_rgb[:,:,1] = CH5*0.647
                ch5_rgb[:,:,2] = 0
                All_Channels = cv2.addWeighted(ch5_rgb, 1, All_Channels, 1, 0)
            height, width, ch = np.shape(All_Channels)
            totalBytes = All_Channels.nbytes

            if displaygui.NuclMaskCheckBox.isChecked() == True:
                
                self.input_image = self.IMAGE_TO_BE_MASKED()
                bound, filled_res = self.ImageAnalyzer.neuceli_segmenter(self.input_image,
                                                                         self.METADATA_DATAFRAME["PixPerMic"].iloc[0])

                if displaygui.NucPreviewMethod.currentText() == "Boundary":
                    
                    All_Channels[bound != 0] = [255,0,0]
                    
                if displaygui.NucPreviewMethod.currentText() == "Area":
                
                    labeled_array, num_features = label(filled_res)
                    rgblabel = label2rgb(labeled_array,alpha=0.1, bg_label = 0)
                    rgblabel = cv2.normalize(rgblabel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    image_input_stack = np.stack((self.input_image,)*3, axis=-1)
                    All_Channels = cv2.addWeighted(rgblabel,0.2, ch1_rgb, 1, 0)
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
                ch1_spots_img, ch2_spots_img, ch3_spots_img, ch4_spots_img, ch5_spots_img = self.IMAGE_FOR_SPOT_DETECTION(self.input_image, displaygui)

               
                    
                if ch1_spots_img!=[]:

                    All_Channels[ch1_spots_img != 0] = [255,255,255]

                if ch2_spots_img!=[]:

                    All_Channels[ch2_spots_img != 0] = [0,255,0]

                if ch3_spots_img!=[]:

                    All_Channels[ch3_spots_img != 0] = [255,0,0]

                if ch4_spots_img!=[]:

                    All_Channels[ch4_spots_img != 0] = [0,0,255]
                    
                if ch5_spots_img!=[]:

                    All_Channels[ch5_spots_img != 0] = [255,165,0]
                       
                if displaygui.NucPreviewMethod.currentText() == "Cross":
                    
                    pass
                

            self.SHOWIMAGE(displaygui, All_Channels, width, height, totalBytes)
            
    def SHOWIMAGE(self, displaygui, img, width, height, totalBytes):
            
                        
            displaygui.viewer.setPhoto(QtGui.QPixmap.fromImage(qimage2ndarray.array2qimage(img)))
            
    def IMAGE_TO_BE_MASKED(self):
        
        if self.AnalysisGui.NucMaxZprojectCheckBox.isChecked() == True:
            maskchannel = str(self.AnalysisGui.NucleiChannel.currentIndex()+1)
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
            
            maskchannel = str(self.AnalysisGui.NucleiChannel.currentIndex()+1)
            
            mask_img_name = self.imgchannels.loc[self.imgchannels['channel']== maskchannel]['ImageName'].iloc[0]
            if mask_img_name=="dask_array":
                # loadedimg_formask=self.imgchannels.loc[self.imgchannels['channel']==maskchannel]["Type"].iloc[0].compute()
                loadedimg_formask=self.imgchannels.loc[self.imgchannels['channel']==maskchannel]["Type"].iloc[0]
            else:
            
                loadedimg_formask = mpimg.imread(mask_img_name)
            
            ImageForNucMask = cv2.normalize(loadedimg_formask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return ImageForNucMask
       
    
    def IMAGE_FOR_SPOT_DETECTION(self, nuclei_image, displaygui):
        
        ch1_spots_img, ch2_spots_img, ch3_spots_img, ch4_spots_img , ch5_spots_img = [],[],[],[],[]
        
        if self.AnalysisGui.SpotCh1CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                        (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                        (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                        (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &  
                                        (self.METADATA_DATAFRAME['channel'] == '1')
                                        ]
                loadedimg_forspot = self.ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates, final_spots = self.ImageAnalyzer.SpotDetector(loadedimg_forspot, self.AnalysisGui, nuclei_image, 'Ch1')
                if displaygui.spotPreviewMethod.currentText() == "Circle":
                    ch1_spots_img = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                else:
                    ch1_spots_img = self.ImageAnalyzer.SPOTS_TO_BOUNDARY(final_spots)
                
            else:
                spot_img_name = self.imgchannels.loc[self.imgchannels['channel']== '1']['ImageName'].iloc[0]
                if spot_img_name=="dask_array":
                    # loadedimg_forspots=self.imgchannels.loc[self.imgchannels['channel']=='1']["Type"].iloc[0].compute()
                    loadedimg_forspots=self.imgchannels.loc[self.imgchannels['channel']=='1']["Type"].iloc[0]
                else:
                    loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['channel']== '1']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates,final_spots = self.ImageAnalyzer.SpotDetector(loadedimg_forspot, self.AnalysisGui, nuclei_image, 'Ch1')
                if displaygui.spotPreviewMethod.currentText() == "Circle":
                    ch1_spots_img = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                else:
                    ch1_spots_img = self.ImageAnalyzer.SPOTS_TO_BOUNDARY(final_spots)
        

        if self.AnalysisGui.SpotCh2CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                        (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                        (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                        (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &  
                                        (self.METADATA_DATAFRAME['channel'] == '2')
                                        ]
                loadedimg_forspot = self.ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates,final_spots = self.ImageAnalyzer.SpotDetector(loadedimg_forspot, self.AnalysisGui, nuclei_image, 'Ch2')
                if displaygui.spotPreviewMethod.currentText() == "Circle":
                    ch2_spots_img = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                else:
                    ch2_spots_img = self.ImageAnalyzer.SPOTS_TO_BOUNDARY(final_spots)
                
            else:
                spot_img_name = self.imgchannels.loc[self.imgchannels['channel']== '2']['ImageName'].iloc[0]
                if spot_img_name=="dask_array":
                    # loadedimg_forspots=self.imgchannels.loc[self.imgchannels['channel']=='2']["Type"].iloc[0].compute()
                    loadedimg_forspots=self.imgchannels.loc[self.imgchannels['channel']=='2']["Type"].iloc[0]
                else:
                    loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['channel']== '2']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates,final_spots = self.ImageAnalyzer.SpotDetector(loadedimg_forspot, self.AnalysisGui, nuclei_image, 'Ch2')
                if displaygui.spotPreviewMethod.currentText() == "Circle":
                    ch2_spots_img = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                else:
                    ch2_spots_img = self.ImageAnalyzer.SPOTS_TO_BOUNDARY(final_spots)
       
                
        if self.AnalysisGui.SpotCh3CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                        (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                        (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                        (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) & 
                                        (self.METADATA_DATAFRAME['channel'] == '3')
                                        ]
                loadedimg_forspot = self.ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates,final_spots = self.ImageAnalyzer.SpotDetector(loadedimg_forspot, self.AnalysisGui, nuclei_image, 'Ch3')
                if displaygui.spotPreviewMethod.currentText() == "Circle":
                    ch3_spots_img = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                else:
                    ch3_spots_img = self.ImageAnalyzer.SPOTS_TO_BOUNDARY(final_spots)
            else:

                spot_img_name = self.imgchannels.loc[self.imgchannels['channel']== '3']['ImageName'].iloc[0]
                if spot_img_name=="dask_array":
                    # loadedimg_forspots=self.imgchannels.loc[self.imgchannels['channel']=='3']["Type"].iloc[0].compute()
                    loadedimg_forspots=self.imgchannels.loc[self.imgchannels['channel']=='3']["Type"].iloc[0]
                else:
                    loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['channel']== '3']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates,final_spots = self.ImageAnalyzer.SpotDetector(loadedimg_forspot, self.AnalysisGui, nuclei_image, 'Ch3')
                if displaygui.spotPreviewMethod.currentText() == "Circle":
                    ch3_spots_img = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                else:
                    ch3_spots_img = self.ImageAnalyzer.SPOTS_TO_BOUNDARY(final_spots)
        
                
        if self.AnalysisGui.SpotCh4CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                        (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                        (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                        (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &  
                                        (self.METADATA_DATAFRAME['channel'] == '4')
                                        ]
                loadedimg_forspot = self.ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates,final_spots = self.ImageAnalyzer.SpotDetector(loadedimg_forspot, self.AnalysisGui, nuclei_image, 'Ch4')
                if displaygui.spotPreviewMethod.currentText() == "Circle":
                    ch4_spots_img = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                else:
                    ch4_spots_img = self.ImageAnalyzer.SPOTS_TO_BOUNDARY(final_spots)
                
            else:

                spot_img_name = self.imgchannels.loc[self.imgchannels['channel']== '4']['ImageName'].iloc[0]
                if spot_img_name=="dask_array":
                    # loadedimg_forspots=self.imgchannels.loc[self.imgchannels['channel']=='4']["Type"].iloc[0].compute()
                    loadedimg_forspots=self.imgchannels.loc[self.imgchannels['channel']=='4']["Type"].iloc[0]
                else:
                    loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['channel']== '4']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates,final_spots = self.ImageAnalyzer.SpotDetector(loadedimg_forspot, self.AnalysisGui, nuclei_image, 'Ch4')
                if displaygui.spotPreviewMethod.currentText() == "Circle":
                    ch4_spots_img = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                else:
                    ch4_spots_img = self.ImageAnalyzer.SPOTS_TO_BOUNDARY(final_spots)
                
        if self.AnalysisGui.SpotCh5CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['column'] == str(self.grid_data[0])) & 
                                        (self.METADATA_DATAFRAME['row'] == str(self.grid_data[1])) & 
                                        (self.METADATA_DATAFRAME['time_point'] == str(self.grid_data[2])) & 
                                        (self.METADATA_DATAFRAME['field_index'] == str(self.grid_data[3])) &  
                                        (self.METADATA_DATAFRAME['channel'] == '5')
                                        ]
                loadedimg_forspot = self.ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates,final_spots = self.ImageAnalyzer.SpotDetector(loadedimg_forspot, self.AnalysisGui, nuclei_image, 'Ch5')
                if displaygui.spotPreviewMethod.currentText() == "Circle":
                    ch5_spots_img = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                else:
                    ch5_spots_img = self.ImageAnalyzer.SPOTS_TO_BOUNDARY(final_spots)
            else:

                spot_img_name = self.imgchannels.loc[self.imgchannels['channel']=='5']['ImageName'].iloc[0]
                if spot_img_name=="dask_array":
                    # loadedimg_forspots=self.imgchannels.loc[self.imgchannels['channel']=='5']["Type"].iloc[0].compute()
                    loadedimg_forspots=self.imgchannels.loc[self.imgchannels['channel']=='5']["Type"].iloc[0]
                else:
                    loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['channel']== '5']['ImageName'].iloc[0])
                
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates,final_spots = self.ImageAnalyzer.SpotDetector(loadedimg_forspot, self.AnalysisGui, nuclei_image, 'Ch5')
                if displaygui.spotPreviewMethod.currentText() == "Circle":
                    ch5_spots_img = self.ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                else:
                    ch5_spots_img = self.ImageAnalyzer.SPOTS_TO_BOUNDARY(final_spots)
            
        return ch1_spots_img, ch2_spots_img, ch3_spots_img, ch4_spots_img, ch5_spots_img


    def ON_ADJUST_INTENSITY(self, input_img, min_range, max_range):
        eplsion = 0.005
        mid_img = np.zeros((input_img.shape),dtype='uint16')
        input_img[input_img < min_range] = 0
        mid_img = input_img.astype('uint16')
        
        scale_factor = 255/(max_range + eplsion)
        mid_img = np.round(mid_img * scale_factor).astype('uint16')
        
        mid_img[mid_img > 255] = 255
        #output_img = cv2.normalize(input_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        output_img = mid_img.astype('uint8')
        
        return output_img
    