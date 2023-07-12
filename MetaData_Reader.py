from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
from xml.dom import minidom
import numpy as np
import pandas as pd
import HiTIPS

class CellVoyager(object):
        
    def READ_FROM_METADATA(self, metadatafilename):
    
        PATH_TO_FILES = os.path.split(metadatafilename)[0]
        self.mydoc = minidom.parse(metadatafilename)
        self.items = self.mydoc.getElementsByTagName('bts:MeasurementRecord')
        
        df_cols = ["ImageName", "Column", "Row", "TimePoint", "FieldIndex", "ZSlice", "Channel", 
                   "X_coordinates", "Y_coordinates","Z_coordinate", "ActionIndex", "Action", "Type", "Time"]
        rows = []
        
        for i in range(self.items.length):
            
            fullstring = self.items[i].firstChild.data
            substring = "Error"

            if fullstring.find(substring) == -1:
                rows.append({

                     "ImageName": os.path.join(PATH_TO_FILES, self.items[i].firstChild.data), 
                     "Column": self.items[i].attributes['bts:Column'].value, 
                     "Row": self.items[i].attributes['bts:Row'].value, 
                     "TimePoint": self.items[i].attributes['bts:TimePoint'].value, 
                     "FieldIndex": self.items[i].attributes['bts:FieldIndex'].value, 
                     "ZSlice": self.items[i].attributes['bts:ZIndex'].value, 
                     "Channel": self.items[i].attributes['bts:Ch'].value,
                     "X_coordinates": self.items[i].attributes['bts:X'].value,
                     "Y_coordinates": self.items[i].attributes['bts:Y'].value,
                     "Z_coordinate": self.items[i].attributes['bts:Z'].value,
                     "ActionIndex": self.items[i].attributes['bts:ActionIndex'].value,
                     "Action": self.items[i].attributes['bts:Action'].value, 
                     "Type": self.items[i].attributes['bts:Type'].value, 
                     "Time": self.items[i].attributes['bts:Time'].value
                })
            
        
        self.Meta_Data_df = pd.DataFrame(rows, columns = df_cols)
        
        metadatafilename = os.path.join(PATH_TO_FILES,'MeasurementDetail.mrf')
        mydoc = minidom.parse(metadatafilename)
        PATH_TO_FILES = os.path.split(metadatafilename)[0]
        items = mydoc.getElementsByTagName('bts:MeasurementChannel')
        
        
        PixPerMic_Text= 'Pixel Size = '"{:.2f}".format(float(items[0].attributes['bts:HorizontalPixelDimension'].value)) + '\u03BC'+'m'
        self.inout_resource_gui.NumFilesLoadedLbl.setText(QtCore.QCoreApplication.translate("MainWindow", PixPerMic_Text))
        self.inout_resource_gui.NumFilesLoadedLbl.setStyleSheet("color: blue")