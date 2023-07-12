from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
import numpy as np
import time
import pandas as pd
import matplotlib.image as mpimg
import cv2
from PIL import Image
from skimage import exposure
from PIL import Image, ImageQt
import qimage2ndarray
from Analysis import ImageAnalyzer
from AnalysisGUI import analyzer
from scipy.ndimage import label
from skimage.color import label2rgb

class imagedisplayer(analyzer,QWidget):
    METADATA_DATAFRAME = pd.DataFrame()
    imgchannels = pd.DataFrame()
    def __init__(self,analysisgui,centralwidget):
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
        #analysisgui = analyzer(centralwidget)
        self.AnalysisGui = analysisgui
    def display_initializer(self, out_df, displaygui, IO_GUI):
            
            
            self.METADATA_DATAFRAME = out_df
            
            displaybtn = IO_GUI.DisplayCheckBox
            if displaybtn.isChecked() == True:
            # Image scroller and spinbox initialization
                numoffiles = np.asarray(out_df['Column'], dtype=int).__len__()
            ###### COLUMN INITIALIZER
                displaygui.ColScroller.setMaximum(np.asarray(out_df['Column'], dtype=int).max())
                displaygui.ColScroller.setMinimum(np.asarray(out_df['Column'], dtype=int).min())
                displaygui.ColScroller.setValue(np.asarray(out_df['Column'], dtype=int).min())

                displaygui.ColSpinBox.setMaximum(np.asarray(out_df['Column'], dtype=int).max())
                displaygui.ColSpinBox.setMinimum(np.asarray(out_df['Column'], dtype=int).min())
                displaygui.ColSpinBox.setValue(np.asarray(out_df['Column'], dtype=int).min())

                ### ROW INITIALIZER

                displaygui.RowScroller.setMaximum(np.asarray(out_df['Row'], dtype=int).max())
                displaygui.RowScroller.setMinimum(np.asarray(out_df['Row'], dtype=int).min())
                displaygui.RowScroller.setValue(np.asarray(out_df['Row'], dtype=int).min())

                displaygui.RowSpinBox.setMaximum(np.asarray(out_df['Row'], dtype=int).max())
                displaygui.RowSpinBox.setMinimum(np.asarray(out_df['Row'], dtype=int).min())
                displaygui.RowSpinBox.setValue(np.asarray(out_df['Row'], dtype=int).min())

                ##### Z-STACK INITIALIZER

                displaygui.ZScroller.setMaximum(np.asarray(out_df['ZSlice'], dtype=int).max())
                displaygui.ZScroller.setMinimum(np.asarray(out_df['ZSlice'], dtype=int).min())
                displaygui.ZScroller.setValue(np.asarray(out_df['ZSlice'], dtype=int).min())

                displaygui.ZSpinBox.setMaximum(np.asarray(out_df['ZSlice'], dtype=int).max())
                displaygui.ZSpinBox.setMinimum(np.asarray(out_df['ZSlice'], dtype=int).min())
                displaygui.ZSpinBox.setValue(np.asarray(out_df['ZSlice'], dtype=int).min())

                #### FOV INITIALIZER

                displaygui.FOVScroller.setMaximum(np.asarray(out_df['FieldIndex'], dtype=int).max())
                displaygui.FOVScroller.setMinimum(np.asarray(out_df['FieldIndex'], dtype=int).min())
                displaygui.FOVScroller.setValue(np.asarray(out_df['FieldIndex'], dtype=int).min())

                displaygui.FOVSpinBox.setMaximum(np.asarray(out_df['FieldIndex'], dtype=int).max())
                displaygui.FOVSpinBox.setMinimum(np.asarray(out_df['FieldIndex'], dtype=int).min())
                displaygui.FOVSpinBox.setValue(np.asarray(out_df['FieldIndex'], dtype=int).min())

                ###### TIME POINT INITIALIZER

                displaygui.TScroller.setMaximum(np.asarray(out_df['TimePoint'], dtype=int).max())
                displaygui.TScroller.setMinimum(np.asarray(out_df['TimePoint'], dtype=int).min())
                displaygui.TScroller.setValue(np.asarray(out_df['TimePoint'], dtype=int).min())

                displaygui.TSpinBox.setMaximum(np.asarray(out_df['TimePoint'], dtype=int).max())
                displaygui.TSpinBox.setMinimum(np.asarray(out_df['TimePoint'], dtype=int).min())
                displaygui.TSpinBox.setValue(np.asarray(out_df['TimePoint'], dtype=int).min())

                # Histogram Max Min initialization for slider and spinbox

                displaygui.MaxHistSlider.setMaximum(255)
                displaygui.MaxHistSlider.setMinimum(0)
                displaygui.MaxHistSlider.setValue(255)

                displaygui.MaxHistSpinBox.setMaximum(255)
                displaygui.MaxHistSpinBox.setMinimum(0)
                displaygui.MaxHistSpinBox.setValue(255)

                displaygui.MinHistSlider.setMaximum(255)
                displaygui.MinHistSlider.setMinimum(0)
                displaygui.MinHistSlider.setValue(0)

                displaygui.MinHistSpinBox.setMaximum(255)
                displaygui.MinHistSpinBox.setMinimum(0)
                displaygui.MinHistSpinBox.setValue(0)
                
                
                
            else:
                pass

    
    def COL_SCROLLER_MOVE_UPDATE(self, displaygui):
            
            self.Col_Scroller_ind = displaygui.ColScroller.value()
            displaygui.ColSpinBox.setValue(self.Col_Scroller_ind)
            self.GET_IMAGE_NAME(displaygui)
            #self.READ_IMAGE(displaygui, fnames[self.Img_Scroller_ind-1])
            
    def COL_SPINBOX_UPDATE(self, displaygui):
            
            self.Col_Spinbox_ind = displaygui.ColSpinBox.value()
            displaygui.ColScroller.setValue(self.Col_Spinbox_ind)
            self.GET_IMAGE_NAME(displaygui)
            #self.READ_IMAGE( displaygui, fnames[self.Spinbox_ind-1])

    def ROW_SCROLLER_MOVE_UPDATE(self, displaygui):
            
            self.Row_Scroller_ind = displaygui.RowScroller.value()
            displaygui.RowSpinBox.setValue(self.Row_Scroller_ind)
            self.GET_IMAGE_NAME(displaygui)
            #self.READ_IMAGE(displaygui, fnames[self.Img_Scroller_ind-1])
            
    def ROW_SPINBOX_UPDATE(self, displaygui):
            
            self.Row_Spinbox_ind = displaygui.RowSpinBox.value()
            displaygui.RowScroller.setValue(self.Row_Spinbox_ind)
            self.GET_IMAGE_NAME(displaygui)
            #self.READ_IMAGE( displaygui, fnames[self.Spinbox_ind-1])
    def Z_SCROLLER_MOVE_UPDATE(self, displaygui):
            
            self.Z_Scroller_ind = displaygui.ZScroller.value()
            displaygui.ZSpinBox.setValue(self.Z_Scroller_ind)
            self.GET_IMAGE_NAME(displaygui)
            #self.READ_IMAGE(displaygui, fnames[self.Img_Scroller_ind-1])
            
    def Z_SPINBOX_UPDATE(self, displaygui):
            
            self.Z_Spinbox_ind = displaygui.ZSpinBox.value()
            displaygui.ZScroller.setValue(self.Z_Spinbox_ind)
            self.GET_IMAGE_NAME(displaygui)
            #self.READ_IMAGE( displaygui, fnames[self.Spinbox_ind-1])
    def FOV_SCROLLER_MOVE_UPDATE(self, displaygui):
            
            self.FOV_Scroller_ind = displaygui.FOVScroller.value()
            displaygui.FOVSpinBox.setValue(self.FOV_Scroller_ind)
            self.GET_IMAGE_NAME(displaygui)
            #self.READ_IMAGE(displaygui, fnames[self.Img_Scroller_ind-1])
            
    def FOV_SPINBOX_UPDATE(self, displaygui):
            
            self.FOV_Spinbox_ind = displaygui.FOVSpinBox.value()
            displaygui.FOVScroller.setValue(self.FOV_Spinbox_ind)
            self.GET_IMAGE_NAME(displaygui)
            #self.READ_IMAGE( displaygui, fnames[self.Spinbox_ind-1])
            
    def T_SCROLLER_MOVE_UPDATE(self, displaygui):
            
            self.T_Scroller_ind = displaygui.TScroller.value()
            displaygui.TSpinBox.setValue(self.T_Scroller_ind)
            self.GET_IMAGE_NAME(displaygui)
            #self.READ_IMAGE(displaygui, fnames[self.Img_Scroller_ind-1])
            
    def T_SPINBOX_UPDATE(self, displaygui):
            
            self.T_Spinbox_ind = displaygui.TSpinBox.value()
            displaygui.TScroller.setValue(self.T_Spinbox_ind)
            self.GET_IMAGE_NAME(displaygui)
            #self.READ_IMAGE( displaygui, fnames[self.Spinbox_ind-1])
            
    ##### Image histogram controller funcions            
    def MAX_HIST_SPINBOX_UPDATE(self, displaygui):
            
            self.MaxSpinbox_ind = displaygui.MaxHistSpinBox.value()
            self.MinSlider_ind = displaygui.MinHistSlider.value()
            self.MinSpinbox_ind = displaygui.MinHistSpinBox.value()
            displaygui.MaxHistSlider.setValue(self.MaxSpinbox_ind)
            
            if self.MaxSpinbox_ind <= self.MinSlider_ind:
                
                displaygui.MinHistSlider.setValue(self.MaxSpinbox_ind)
                displaygui.MinHistSpinBox.setValue(self.MaxSpinbox_ind)
                
                        
            self.READ_IMAGE(displaygui, self.imgchannels)
            
    def MAX_HIST_SLIDER_UPDATE(self, displaygui):
            
            self.MaxSlider_ind = displaygui.MaxHistSlider.value()
            self.MinSlider_ind = displaygui.MinHistSlider.value()
            self.MinSpinbox_ind = displaygui.MinHistSpinBox.value()
            displaygui.MaxHistSpinBox.setValue(self.MaxSlider_ind)
            
            if self.MaxSlider_ind <= self.MinSlider_ind:
                
                displaygui.MinHistSlider.setValue(self.MaxSlider_ind)
                displaygui.MinHistSpinBox.setValue(self.MaxSlider_ind)
            
            self.READ_IMAGE(displaygui, self.imgchannels)
            
    def MIN_HIST_SPINBOX_UPDATE(self, displaygui):
            
            self.MinSpinbox_ind = displaygui.MinHistSpinBox.value()
            self.MaxSlider_ind = displaygui.MaxHistSlider.value()
            self.MaxSpinbox_ind = displaygui.MaxHistSpinBox.value()
            displaygui.MinHistSlider.setValue(self.MinSpinbox_ind)
            
            if self.MinSpinbox_ind >= self.MaxSlider_ind:
                
                displaygui.MaxHistSlider.setValue(self.MinSpinbox_ind)
                displaygui.MaxHistSpinBox.setValue(self.MinSpinbox_ind)
            
            self.READ_IMAGE(displaygui, self.imgchannels)
            
    def MIN_HIST_SLIDER_UPDATE(self, displaygui):
            
            self.MinSlider_ind = displaygui.MinHistSlider.value()
            self.MaxSlider_ind = displaygui.MaxHistSlider.value()
            self.MaxSpinbox_ind = displaygui.MaxHistSpinBox.value()
            
            if self.MinSlider_ind >= self.MaxSlider_ind:
                
                displaygui.MaxHistSlider.setValue(self.MinSlider_ind)
                displaygui.MaxHistSpinBox.setValue(self.MinSlider_ind)
                
            displaygui.MinHistSpinBox.setValue(self.MinSlider_ind)
            
            self.READ_IMAGE(displaygui, self.imgchannels)
            
    def GET_IMAGE_NAME(self,displaygui):
            
            self.imgchannels = self.METADATA_DATAFRAME.loc[
                                    (self.METADATA_DATAFRAME['Column'] == str(displaygui.ColScroller.value())) & 
                                    (self.METADATA_DATAFRAME['Row'] == str(displaygui.RowScroller.value())) & 
                                    (self.METADATA_DATAFRAME['TimePoint'] == str(displaygui.TScroller.value())) & 
                                    (self.METADATA_DATAFRAME['FieldIndex'] == str(displaygui.FOVScroller.value())) & 
                                    (self.METADATA_DATAFRAME['ZSlice'] == str(displaygui.ZScroller.value()))
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
            if 'All_Channels' in locals():
                del All_Channels
            
                
            if displaygui.Ch1CheckBox.isChecked() == True:
                
                ch1_img = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']=='1']['ImageName'].iloc[0])
                ch1_img = (ch1_img/256).astype('uint8')
                self.CH1_img = ch1_img
                self.height, self.width = np.shape(ch1_img)
                
            if displaygui.Ch2CheckBox.isChecked() == True:

                ch2_img = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']=='2']['ImageName'].iloc[0])
                ch2_img = (ch2_img/256).astype('uint8')
                self.height, self.width = np.shape(ch2_img)

            if displaygui.Ch3CheckBox.isChecked() == True:

                ch3_img = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']=='3']['ImageName'].iloc[0])
                ch3_img = (ch3_img/256).astype('uint8')
                self.height, self.width = np.shape(ch3_img)
                
            if displaygui.Ch4CheckBox.isChecked() == True:

                ch4_img = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']=='4']['ImageName'].iloc[0])
                ch4_img = (ch4_img/256).astype('uint8')
                self.height, self.width = np.shape(ch4_img)
            
            if self.height or self.width:
                
                self.RGB_channels = np.zeros((self.height, self.width, 3))
                if 'ch2_img' in locals():
                    self.RGB_channels[:,:,0] = ch2_img
                if 'ch3_img' in locals():
                    self.RGB_channels[:,:,1] = ch3_img
                if 'ch4_img' in locals():
                    self.RGB_channels[:,:,2] = ch4_img
                
                self.ADJUST_IMAGE_CONTRAST(displaygui, self.CH1_img, self.RGB_channels )
               
            
    def ADJUST_IMAGE_CONTRAST(self, displaygui, CH1 , RGB_CHANNELS):
            
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
                    
                CH1 = exposure.rescale_intensity(CH1, out_range=(self.ch1_hist_min, self.ch1_hist_max))
                RGB_Channels[:,:,0] = exposure.rescale_intensity(RGB_Channels[:,:,0], out_range=(self.ch2_hist_min,
                                                                                                 self.ch2_hist_max))
                RGB_Channels[:,:,1] = exposure.rescale_intensity(RGB_Channels[:,:,1], out_range=(self.ch3_hist_min,
                                                                                                 self.ch3_hist_max))
                RGB_Channels[:,:,2] = exposure.rescale_intensity(RGB_Channels[:,:,2], out_range=(self.ch4_hist_min,
                                                                                                 self.ch4_hist_max))
            self.MERGEIAMGES(displaygui, CH1, RGB_Channels)
            
    def MERGEIAMGES(self,displaygui, CH1, RGB_Channels):
        
            if displaygui.Ch1CheckBox.isChecked() == True:
                ch1_rgb = np.stack((CH1,)*3, axis=-1)
            else:
                ch1_rgb = np.zeros(RGB_Channels.shape, dtype = np.uint8)
            All_Channels = cv2.addWeighted(ch1_rgb, 1, RGB_Channels, 1, 0)
            height, width, ch = np.shape(All_Channels)
            totalBytes = All_Channels.nbytes
            #print(self.AnalysisGui.NucleiChannel.currentIndex().dtype)
            if displaygui.NuclMaskCheckBox.isChecked() == True:
                
                self.input_image = self.IMAGE_TO_BE_MASKED(displaygui)
                bound, filled_res = ImageAnalyzer.neuceli_segmenter(self.input_image)
                #cv2.imwrite('mask_saved.jpg',bound)
                if displaygui.NucPreviewMethod.currentText() == "Boundary":
                    
                    All_Channels[bound != 0] = [255,0,0]
                    
                if displaygui.NucPreviewMethod.currentText() == "Area":
                
                    labeled_array, num_features = label(filled_res)
                    rgblabel = label2rgb(labeled_array,alpha=0.1, bg_label = 0)
                    rgblabel = cv2.normalize(rgblabel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    image_input_stack = np.stack((self.input_image,)*3, axis=-1)
                    All_Channels = cv2.addWeighted(rgblabel,0.2, ch1_rgb, 1, 0)
                    ##############
            if displaygui.SpotsCheckBox.isChecked() == True:
                
                ch1_spots_log, ch2_spots_log, ch3_spots_log, ch4_spots_log = self.IMAGE_FOR_SPOT_DETECTION(displaygui)

                if displaygui.SpotPreviewMethod.currentText() == "Dots":
                    
                    if ch1_spots_log!=[]:
                    
                        All_Channels[ch1_spots_log != 0] = [255,255,255]
                        
                    if ch2_spots_log!=[]:
                    
                        All_Channels[ch2_spots_log != 0] = [255,0,0]
                        
                    if ch3_spots_log!=[]:
                    
                        All_Channels[ch3_spots_log != 0] = [0,255,0]
                        
                    if ch4_spots_log!=[]:
                    
                        All_Channels[ch4_spots_log != 0] = [0,0,255]
                    
                    
                if displaygui.NucPreviewMethod.currentText() == "Cross":
                    
                    pass
                

            self.SHOWIMAGE(displaygui, All_Channels, width, height, totalBytes)
            
    def SHOWIMAGE(self, displaygui, img, width, height, totalBytes):
            
                        
            #displaygui.scene.addPixmap(QtGui.QPixmap.fromImage(qimage2ndarray.array2qimage(img)))
            displaygui.viewer.setPhoto(QtGui.QPixmap.fromImage(qimage2ndarray.array2qimage(img)))
#             displaygui.ImageView.setScene(displaygui.scene)
#             self.fitInView(displaygui.scene, displaygui)
            
    def IMAGE_TO_BE_MASKED(self, displaygui):
        
        if self.AnalysisGui.NucMaxZprojectCheckBox.isChecked() == True:
            maskchannel = str(self.AnalysisGui.NucleiChannel.currentIndex()+1)
            self.imgformask = self.METADATA_DATAFRAME.loc[
                                    (self.METADATA_DATAFRAME['Column'] == str(displaygui.ColScroller.value())) & 
                                    (self.METADATA_DATAFRAME['Row'] == str(displaygui.RowScroller.value())) & 
                                    (self.METADATA_DATAFRAME['TimePoint'] == str(displaygui.TScroller.value())) & 
                                    (self.METADATA_DATAFRAME['FieldIndex'] == str(displaygui.FOVScroller.value())) & 
                                    (self.METADATA_DATAFRAME['Channel'] == maskchannel)
                                    ]
            loadedimg_formask = ImageAnalyzer.max_z_project(self.imgformask)
            ImageForNucMask = cv2.normalize(loadedimg_formask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            
            maskchannel = str(self.AnalysisGui.NucleiChannel.currentIndex()+1)
            
            loadedimg_formask = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']== maskchannel]['ImageName'].iloc[0])
            
            ImageForNucMask = cv2.normalize(loadedimg_formask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return ImageForNucMask
       
    
    def IMAGE_FOR_SPOT_DETECTION(self, displaygui):
        
        ch1_spots_log, ch2_spots_log, ch3_spots_log, ch4_spots_log = [],[],[],[]
        
        if self.AnalysisGui.SpotCh1CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['Column'] == str(displaygui.ColScroller.value())) & 
                                        (self.METADATA_DATAFRAME['Row'] == str(displaygui.RowScroller.value())) & 
                                        (self.METADATA_DATAFRAME['TimePoint'] == str(displaygui.TScroller.value())) & 
                                        (self.METADATA_DATAFRAME['FieldIndex'] == str(displaygui.FOVScroller.value())) & 
                                        (self.METADATA_DATAFRAME['Channel'] == '1')
                                        ]
                loadedimg_forspot = ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ch1_spots_log = ImageAnalyzer.LOG_spotDetector(ImageForSpots)
                
            else:

                loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']== '1']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ch1_spots_log = ImageAnalyzer.LOG_spotDetector(ImageForSpots)
        

        if self.AnalysisGui.SpotCh2CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['Column'] == str(displaygui.ColScroller.value())) & 
                                        (self.METADATA_DATAFRAME['Row'] == str(displaygui.RowScroller.value())) & 
                                        (self.METADATA_DATAFRAME['TimePoint'] == str(displaygui.TScroller.value())) & 
                                        (self.METADATA_DATAFRAME['FieldIndex'] == str(displaygui.FOVScroller.value())) & 
                                        (self.METADATA_DATAFRAME['Channel'] == '2')
                                        ]
                loadedimg_forspot = ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ch2_spots_log = ImageAnalyzer.LOG_spotDetector(ImageForSpots)
                
            else:
                
                loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']== '2']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ch2_spots_log = ImageAnalyzer.LOG_spotDetector(ImageForSpots)
       
                
        if self.AnalysisGui.SpotCh3CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['Column'] == str(displaygui.ColScroller.value())) & 
                                        (self.METADATA_DATAFRAME['Row'] == str(displaygui.RowScroller.value())) & 
                                        (self.METADATA_DATAFRAME['TimePoint'] == str(displaygui.TScroller.value())) & 
                                        (self.METADATA_DATAFRAME['FieldIndex'] == str(displaygui.FOVScroller.value())) & 
                                        (self.METADATA_DATAFRAME['Channel'] == '3')
                                        ]
                loadedimg_forspot = ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ch3_spots_log = ImageAnalyzer.LOG_spotDetector(ImageForSpots)
            else:

                loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']== '3']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ch3_spots_log = ImageAnalyzer.LOG_spotDetector(ImageForSpots)
        
                
        if self.AnalysisGui.SpotCh4CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['Column'] == str(displaygui.ColScroller.value())) & 
                                        (self.METADATA_DATAFRAME['Row'] == str(displaygui.RowScroller.value())) & 
                                        (self.METADATA_DATAFRAME['TimePoint'] == str(displaygui.TScroller.value())) & 
                                        (self.METADATA_DATAFRAME['FieldIndex'] == str(displaygui.FOVScroller.value())) & 
                                        (self.METADATA_DATAFRAME['Channel'] == '4')
                                        ]
                loadedimg_forspot = ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ch4_spots_log = ImageAnalyzer.LOG_spotDetector(ImageForSpots)
            else:

                loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']== '4']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ch4_spots_log = ImageAnalyzer.LOG_spotDetector(ImageForSpots)
            
                
        return ch1_spots_log, ch2_spots_log, ch3_spots_log, ch4_spots_log
    
    def fitInView(self, scene, displaygui, scale=True):
        rect = QtCore.QRectF(displaygui.scene.sceneRect())
        
        displaygui.scene.setSceneRect(rect)
            
        unity = displaygui.ImageView.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
        displaygui.ImageView.scale(1 / unity.width(), 1 / unity.height())
        viewrect = displaygui.ImageView.viewport().rect()
        scenerect = displaygui.ImageView.transform().mapRect(rect)
        factor = min(viewrect.width() / scenerect.width(),
                     viewrect.height() / scenerect.height())
        displaygui.ImageView.scale(factor, factor)
        self._zoom = 0
    
    def wheelEvent(self, event: QtGui.QWheelEvent):
        print('ttt')
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:                
            factor = 0.8
            self._zoom -= 1
        if self._zoom > 0:
            displaygui.ImageView.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0

    