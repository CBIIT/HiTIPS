from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QWidget, QMessageBox
from xml.dom import minidom
import numpy as np
import pandas as pd
import glob
import json
from . import AnalysisGUI, IO_ResourceGUI, GridLayout, DisplayGUI_Copy1, BatchAnalyzer, Analysis
import os
import sys
from aicsimageio import AICSImage
import cv2
import pathlib
import numpy as np
import dask.array as da
import nd2reader
from imaris_ims_file_reader.ims import ims

class ImageReader(object):
    
    def __init__(self, ControlPanel, inout_resource_gui, displaygui, analysisgui):
        
        self.ControlPanel =ControlPanel
        self.analysisgui = analysisgui
        self.inout_resource_gui = inout_resource_gui
        self.displaygui = displaygui
        self.inout_resource_gui.LoadMetadataButton.clicked.connect(lambda: self.ON_CLICK_LOADBUTTON())
        self.inout_resource_gui.LoadImageButton.clicked.connect(lambda: self.ON_CLICK_LOADIMGBUTTON())
        
    def ON_CLICK_LOADBUTTON(self):
        if self.inout_resource_gui.DeviceType.currentText()=='CellVoyager':
            options = QtWidgets.QFileDialog.Options()
            self.fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select Metadata Files...',
                                                                    '', "MLF files (*.mlf)"
                                                                    , options=options)
            if self.fnames:
                filename, file_extension = os.path.splitext(self.fnames[0])
                self.ControlPanel.Meta_Data_df = self.READ_FROM_METADATA(self.fnames[0])
                self.inout_resource_gui.DisplayCheckBox.setEnabled(True)
                self.analysisgui.setEnabled(True)

                PixPerMic_Text= 'Pixel Size = '"{:.2f}".format(float(self.Meta_Data_df["PixPerMic"].iloc[0])) + '\u03BC'+'m'
                self.inout_resource_gui.NumFilesLoadedLbl.setText(QtCore.QCoreApplication.translate("MainWindow", PixPerMic_Text))
                self.inout_resource_gui.NumFilesLoadedLbl.setStyleSheet("color: blue")

        if self.inout_resource_gui.DeviceType.currentText()=='MicroManager':
            options = QtWidgets.QFileDialog.Options()
            self.Data_dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption= "Select Data Directory", options=options)

            if self.Data_dir:

                self.ControlPanel.Meta_Data_df = self.MICROMANAGER_READER(self.Data_dir)
                self.inout_resource_gui.DisplayCheckBox.setEnabled(True)
                self.analysisgui.setEnabled(True)
                PixPerMic_Text= 'Pixel Size = '"{:.2f}".format(float(self.Meta_Data_df["PixPerMic"].iloc[0])) + '\u03BC'+'m'
                self.inout_resource_gui.NumFilesLoadedLbl.setText(QtCore.QCoreApplication.translate("MainWindow", PixPerMic_Text))
                self.inout_resource_gui.NumFilesLoadedLbl.setStyleSheet("color: blue")

            
    def ON_CLICK_LOADIMGBUTTON(self):
        
        options = QtWidgets.QFileDialog.Options()
        self.fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select Image Files...',
                                                                '', "Images (*.czi *.tiff *.tif *.nd2 *.ims)"
                                                                , options=options)
        if self.fnames:
            self.ControlPanel.Meta_Data_df = self.LOAD_BIOFORMAT_DATA(self.fnames)
            self.inout_resource_gui.DisplayCheckBox.setEnabled(True)
            self.analysisgui.setEnabled(True)
    
            PixPerMic_Text= 'Pixel Size = '"{:.2f}".format(self.Meta_Data_df['PixPerMic'].iloc[0]) + '\u03BC'+'m'
            self.inout_resource_gui.NumFilesLoadedLbl.setText(QtCore.QCoreApplication.translate("MainWindow", PixPerMic_Text))
            self.inout_resource_gui.NumFilesLoadedLbl.setStyleSheet("color: blue")
    
        
        
    def READ_FROM_METADATA(self, metadatafilename):
    
        PATH_TO_FILES = os.path.split(metadatafilename)[0]
        self.mydoc = minidom.parse(metadatafilename)
        self.items = self.mydoc.getElementsByTagName('bts:MeasurementRecord')
        print(PATH_TO_FILES)
        metadatafilename_mrf = os.path.join(PATH_TO_FILES,'MeasurementDetail.mrf')
        mydoc_mrf = minidom.parse(metadatafilename_mrf)
        PATH_TO_FILES = os.path.split(metadatafilename_mrf)[0]
        items_mrf = mydoc_mrf.getElementsByTagName('bts:MeasurementChannel')
        
        df_cols = ["ImageName", "column", "row", "time_point", "field_index", "z_slice", "channel", 
                   "x_coordinates", "y_coordinates","z_coordinate", "action_index", "action", "Type", "Time", "PixPerMic"]
        rows = []
        
        for i in range(self.items.length):
            
            fullstring = self.items[i].firstChild.data
            substring = "Error"

            
            if fullstring.find(substring) == -1:
                if self.items[i].attributes['bts:Type'].value=='IMG':
                    rows.append({

                         "ImageName": os.path.join(PATH_TO_FILES, self.items[i].firstChild.data), 
                         "column": self.items[i].attributes['bts:Column'].value, 
                         "row": self.items[i].attributes['bts:Row'].value, 
                         "time_point": self.items[i].attributes['bts:TimePoint'].value, 
                         "field_index": self.items[i].attributes['bts:FieldIndex'].value, 
                         "z_slice": self.items[i].attributes['bts:ZIndex'].value, 
                         "channel": self.items[i].attributes['bts:Ch'].value,
                         "x_coordinates": self.items[i].attributes['bts:X'].value,
                         "y_coordinates": self.items[i].attributes['bts:Y'].value,
                         "z_coordinate": self.items[i].attributes['bts:Z'].value,
                         "action_index": self.items[i].attributes['bts:ActionIndex'].value,
                         "action": self.items[i].attributes['bts:Action'].value, 
                         "Type": self.items[i].attributes['bts:Type'].value, 
                         "Time": self.items[i].attributes['bts:Time'].value,
                         "PixPerMic": items_mrf[0].attributes['bts:HorizontalPixelDimension'].value
                    })
        
        self.Meta_Data_df = pd.DataFrame(rows, columns = df_cols)
        
        
        ##### mrf data
        
        df_cols = ["Source","channel"]
        rows = []
        
        for i in range(items_mrf.length):
            
            rows.append({

                 "Source": items_mrf[i].attributes['bts:ShadingCorrectionSource'].value, 
                 "channel": items_mrf[i].attributes['bts:Ch'].value,
            })
            _translate = QtCore.QCoreApplication.translate
            if str(items_mrf[i].attributes['bts:Ch'].value)=='1':
                text1 = "Ch1: " + items_mrf[i].attributes['bts:ShadingCorrectionSource'].value[5:11]
                self.displaygui.Ch1CheckBox.setText(_translate("MainWindow", text1))
            if str(items_mrf[i].attributes['bts:Ch'].value)=='2':
                text2 = "Ch2: " + items_mrf[i].attributes['bts:ShadingCorrectionSource'].value[5:11]
                self.displaygui.Ch2CheckBox.setText(_translate("MainWindow", text2))
            if str(items_mrf[i].attributes['bts:Ch'].value)=='3':
                text3 = "Ch3: " + items_mrf[i].attributes['bts:ShadingCorrectionSource'].value[5:11]
                self.displaygui.Ch3CheckBox.setText(_translate("MainWindow", text3))
            if str(items_mrf[i].attributes['bts:Ch'].value)=='4':
                text4 = "Ch4: " + items_mrf[i].attributes['bts:ShadingCorrectionSource'].value[5:11]
                self.displaygui.Ch4CheckBox.setText(_translate("MainWindow", text4))
            if str(items_mrf[i].attributes['bts:Ch'].value)=='5':
                text5 = "Ch5: " + items_mrf[i].attributes['bts:ShadingCorrectionSource'].value[5:11]
                self.displaygui.Ch5CheckBox.setText(_translate("MainWindow", text5))


        self.Mrf_Data_df = pd.DataFrame(rows, columns = df_cols)
        
        return self.Meta_Data_df

    
    def MICROMANAGER_READER(self, metadatadir):

        metadata_mm=pd.DataFrame()

        col_list = ['Time', 'Width', 'Height', 'PixelSize_um', 'Channel', 'FrameIndex', 'SlicePosition', 'Slice', 'PositionIndex', 
                    'PixelSizeUm', 'YPositionUm', 'XPositionUm', 'FileName', 'SliceIndex', 'ZPositionUm']

        text_files = glob.glob(metadatadir + "/**/metadata.txt", recursive = True)
        f_ind=1
        for text_file in text_files:
            with open(text_file) as f:
                string = f.read()
                jsonData = json.loads(string)
            submetadata_mm = pd.DataFrame(jsonData).T
            submetadata_mm = submetadata_mm[col_list]
            submetadata_mm['FileName'] = os.path.dirname(text_file) + "/" + submetadata_mm['FileName'].astype(str)
            submetadata_mm['field_ind'] = f_ind*np.ones(len(submetadata_mm), dtype=int)
            metadata_mm = pd.concat([metadata_mm, submetadata_mm])
            f_ind=f_ind+1
    
        
        df_cols = ["ImageName", "column", "row", "time_point", "field_index", "z_slice", "channel", 
           "x_coordinates", "y_coordinates","z_coordinate", "action_index", "action", "Type", "Time", "PixPerMic"]

        ch_names = ["", "DAPI", "GFP", "Cy5", "CFP", "Cy3"]

        rows = []

        for i in range(len(metadata_mm)):

            if metadata_mm.index[i] != 'Summary':
                rows.append({

                     "ImageName": metadata_mm["FileName"].iloc[i], 
                     "column": str(1), 
                     "row": str(1), 
                     "time_point": str(int(os.path.basename(metadata_mm["FileName"].iloc[i]).split("_")[1])+1), 
                     "field_index": str(metadata_mm["field_ind"].iloc[i]), 
                     "z_slice": str(metadata_mm["SliceIndex"].iloc[i]+1), 
                     "channel": str(ch_names.index(metadata_mm["Channel"].iloc[i].replace("Filter_",""))),
                     "x_coordinates": str(metadata_mm["XPositionUm"].iloc[i]),
                     "y_coordinates": str(metadata_mm["YPositionUm"].iloc[i]),
                     "z_coordinate": str(metadata_mm["ZPositionUm"].iloc[i]),
                     "action_index": str(1),
                     "action": str(1), 
                     "Type": metadata_mm["Channel"].iloc[i].replace("Filter_",""), 
                     "Time": str(metadata_mm["Time"].iloc[i]),
                     "PixPerMic": str(metadata_mm["PixelSizeUm"].iloc[i])
                })

        self.Meta_Data_df = pd.DataFrame(rows, columns = df_cols)
        unique_channels = self.Meta_Data_df["channel"].unique() 

        _translate = QtCore.QCoreApplication.translate
        if "1" in unique_channels:
            text1 = "Ch1: " + ch_names[1]
            self.displaygui.Ch1CheckBox.setText(_translate("MainWindow", text1))
        if "2" in unique_channels:
            text2 = "Ch2: " + ch_names[2]
            self.displaygui.Ch2CheckBox.setText(_translate("MainWindow", text2))
        if "3" in unique_channels:
            text3 = "Ch3: " + ch_names[3]
            self.displaygui.Ch3CheckBox.setText(_translate("MainWindow", text3))
        if "4" in unique_channels:
            text4 = "Ch4: " + ch_names[4]
            self.displaygui.Ch4CheckBox.setText(_translate("MainWindow", text4))
        if "5" in unique_channels:
            text5 = "Ch5: " + ch_names[5]
            self.displaygui.Ch5CheckBox.setText(_translate("MainWindow", text5))
            
                
        return self.Meta_Data_df
    

    def LOAD_BIOFORMAT_DATA(self, image_full_path):
        
        self.fnames = image_full_path
        
        if self.fnames:
            
            df_cols = ["ImageName", "column", "row", "time_point", "field_index", "z_slice", "channel", 
                   "x_coordinates", "y_coordinates","z_coordinate", "action_index", "action", "Type", "Time", "PixPerMic"]
            rows = []
            field_ind=0
            col_value=1
            row_value=1
            for fname in self.fnames:
                field_ind+=1

                if pathlib.Path(fname).suffix == '.nd2':
                    with nd2reader.ND2Reader(fname) as images:
                    # Check which dimensions are present
                        dimensions = set(images.sizes.keys())

                        # Define defaults for missing dimensions
                        defaults = { 'x': 0, 'y': 0, 'z': 1, 't': 1, 'v': 0, 'c': 1   }

                        # Get the sizes for each dimension, using defaults for missing dimensions
                        sizes = {k: images.sizes.get(k, defaults.get(k, 0)) for k in dimensions}

                        # Iterate through each image
                        for fov_idx in range(sizes.get('v', 1)):
                            for t_idx in range(sizes.get('t', 1)):
                                for c_idx in range(sizes.get('c', 1)):
                                    for z_idx in range(sizes.get('z',1)):

                                        rows.append({

                                            "ImageName": "dask_array", 
                                             "column":str(col_value), 
                                             "row": str(row_value), 
                                             "time_point": str(t_idx+1), 
                                             "field_index": str(fov_idx+1), 
                                             "z_slice": str(z_idx+1), 
                                             "channel": str(c_idx+1),
                                             "x_coordinates": str(0),
                                             "y_coordinates": str(0),
                                             "z_coordinate": str(0),
                                             "action_index": str(1),
                                             "action": str(1), 
                                             "Type": images.get_frame_2D(t=t_idx, v=fov_idx, z=z_idx, c=c_idx), 
                                             "Time": str(0),
                                             "PixPerMic": images.metadata['pixel_microns']
                                                    })
                
                elif pathlib.Path(fname).suffix == '.ims':
                    images = ims(fname)
                    # Check which dimensions are present

                    # Iterate through each image
                    for fov_idx in range(1):
                        for t_idx in range(images.TimePoints):
                            for c_idx in range(images.Channels):
                                for z_idx in range(images.shape[2]):

                                    rows.append({

                                        "ImageName": "dask_array", 
                                         "column":str(col_value), 
                                         "row": str(row_value), 
                                         "time_point": str(t_idx+1), 
                                         "field_index": str(fov_idx+1), 
                                         "z_slice": str(z_idx+1), 
                                         "channel": str(c_idx+1),
                                         "x_coordinates": str(0),
                                         "y_coordinates": str(0),
                                         "z_coordinate": str(0),
                                         "action_index": str(1),
                                         "action": str(1), 
                                         "Type": images[t_idx, c_idx, z_idx, :, :], 
                                         "Time": str(0),
                                         "PixPerMic": images.resolution[-1]
                                                })
                else:
        
        
                    img = AICSImage(fname)
                    img.dims  # returns string "STCZYX"
                    img.shape # returns tuple of dimension sizes in STCZYX order
                    if 'S' in dir(img.dims):
                        s_arr = range(img.dims.S)
                    else:
                        s_arr= range(1)
                    for s in s_arr:
                        for t in range(img.dims.T):
                            for c in range(img.dims.C):
                                for z in range(img.dims.Z):

                                    rows.append({

                                        "ImageName": "dask_array", 
                                         "column":str(col_value), 
                                         "row": str(row_value), 
                                         "time_point": str(t+1), 
                                         "field_index": str(field_ind), 
                                         "z_slice": str(z+1), 
                                         "channel": str(c+1),
                                         "x_coordinates": str(0),
                                         "y_coordinates": str(0),
                                         "z_coordinate": str(0),
                                         "action_index": str(s+1),
                                         "action": str(1), 
                                         "Type": img.get_image_data("YX", S=s, C=c, T=t, Z=z), 
                                         "Time": str(0),
                                         "PixPerMic": img.physical_pixel_sizes.Y
                                                })


            self.Meta_Data_df = pd.DataFrame(rows, columns = df_cols)
            
            return self.Meta_Data_df
 