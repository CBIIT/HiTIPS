from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QWidget, QMessageBox
from xml.dom import minidom
import numpy as np
import pandas as pd
import glob
import json
from . import AnalysisGUI, IO_ResourceGUI, GridLayout, DisplayGUI, BatchAnalyzer, Analysis
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
    """
    This class provides functionality for reading and loading various types of biological image data.

    Attributes:
    - ControlPanel: A reference to the ControlPanel object.
    - analysisgui: A reference to the AnalysisGUI object.
    - inout_resource_gui: A reference to the IO_ResourceGUI object.
    - displaygui: A reference to the DisplayGUI object.

    Methods:
    - __init__: Initializes the ImageReader with references to GUI components.
    - ON_CLICK_LOADBUTTON: Handles the action to load metadata from selected files.
    - ON_CLICK_LOADIMGBUTTON: Handles the action to load images from selected files.
    - READ_FROM_METADATA: Reads metadata from a given file and extracts relevant information.
    - MICROMANAGER_READER: Reads and processes metadata from MicroManager software.
    - LOAD_BIOFORMAT_DATA: Loads image data from various bioformats supported files.
    """
    def __init__(self, ControlPanel, inout_resource_gui, displaygui, analysisgui):
        
        """
        Initializes the ImageReader object with references to GUI components.

        Parameters:
        - ControlPanel: Reference to the ControlPanel object.
        - inout_resource_gui: Reference to the IO_ResourceGUI object.
        - displaygui: Reference to the DisplayGUI object.
        - analysisgui: Reference to the AnalysisGUI object.
        """
        self.ControlPanel =ControlPanel
        self.analysisgui = analysisgui
        self.inout_resource_gui = inout_resource_gui
        self.displaygui = displaygui
        self.inout_resource_gui.LoadMetadataButton.clicked.connect(lambda: self.ON_CLICK_LOADBUTTON())
        self.inout_resource_gui.LoadImageButton.clicked.connect(lambda: self.ON_CLICK_LOADIMGBUTTON())
        
    def ON_CLICK_LOADBUTTON(self):
        """
        Handles the action to load metadata from selected files based on the selected device type in the GUI.

        This method allows users to select and load metadata files, processes these files depending on the device type selected in the GUI, 
        and updates the GUI components with the loaded metadata information.
        """
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
        """
        Handles the action to load image files from selected paths.

        This method allows users to select and load image files in various formats. It updates the GUI components with information 
        about the loaded images and prepares the data for further analysis.
        """
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
        
        """
        Reads metadata from a specified file and extracts relevant image information.

        Parameters:
        - metadatafilename (str): Path to the metadata file.

        Returns:
        - DataFrame: A pandas DataFrame containing the extracted metadata information.

        This method reads and processes metadata from specified CellVoyager files, extracts relevant information, 
        and returns it in a structured DataFrame format.
        """
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
        
        """
        Reads and processes metadata from the MicroManager software.

        Parameters:
        - metadatadir (str): Directory containing the metadata files.

        Returns:
        - DataFrame: A pandas DataFrame containing metadata information processed from MicroManager files.

        This method processes metadata from a specified directory containing MicroManager files, organizes the metadata into a structured format, 
        and returns it as a pandas DataFrame.
        """
        
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
        """
        Loads image data from various bioformats supported files.

        Parameters:
        - image_full_path (list of str): Paths to the image files to be loaded.

        Returns:
        - DataFrame: A pandas DataFrame containing information extracted from the image files.

        This method supports loading and processing image data from various file formats (like .czi, .tiff, .tif, .nd2, .ims), 
        organizing this information into a structured DataFrame format.
        """
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
 