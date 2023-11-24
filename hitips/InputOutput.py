from PyQt5 import QtCore, QtGui, QtWidgets
from .HiTIPS import ControlPanel
from PyQt5.QtWidgets import QWidget
import os
from . import MetaData_Reader

class inputoutput_control(QWidget):
        
    
    def __init__(self):
        super().__init__()   
        
    def ON_CLICK_LOADBUTTON(self):
        
        options = QtWidgets.QFileDialog.Options()
        self.fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select Image Files...',
                                                                '', "Image files (*.tiff *tif  *.jpg); XML files (*.mlf *.xml)"
                                                                , options=options)
        
        filename, file_extension = os.path.splitext(self.fnames[0])
        #print(file_extension)
        #if file_extension=='.mlf':
        cellvoyager_reader = MetaData_Reader.CellVoyager()
        cellvoyager_reader.READ_FROM_METADATA(self.fnames[0])
        
    def READ_FROM_METADATA(self, metadatafilename):
    
        self.mydoc = minidom.parse(metadatafilename)
        self.items = self.mydoc.getElementsByTagName('bts:MeasurementRecord')
        
        df_cols = ["ImageName", "Column", "Row", "TimePoint", "FieldIndex", "ZSlice", "Channel"]
        rows = []
        
        for i in range(self.items.length):
    
            rows.append({
                
                 "ImageName": self.items[i].firstChild.data, 
                 "Column": self.items[i].attributes['bts:Column'].value, 
                 "Row": self.items[i].attributes['bts:Row'].value, 
                 "TimePoint": self.items[i].attributes['bts:TimePoint'].value, 
                 "FieldIndex": self.items[i].attributes['bts:FieldIndex'].value, 
                 "ZSlice": self.items[i].attributes['bts:ZIndex'].value, 
                 "Channel": self.items[i].attributes['bts:Ch'].value
                })
        
        self.metadata_df = pd.DataFrame(rows, columns = df_cols)
        cp = HiTIPS.ControlPanel()
        cp.RETURN_METADATA_DATAFRAME (self.metadata_df)
        