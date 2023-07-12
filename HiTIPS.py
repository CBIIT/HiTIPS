from PyQt5 import QtCore, QtGui, QtWidgets
import DisplayGUI, AnalysisGUI, IO_ResourceGUI, GridLayout, DisplayGUI_Copy1, BatchAnalyzer, Analysis
from PyQt5.QtWidgets import QWidget, QMessageBox
import Display, InputOutput, MetaData_Reader, Display_Copy1
import pandas as pd
from xml.dom import minidom
import os
import sys
from aicsimageio import AICSImage
import cv2
import pathlib
from aicsimageio.readers.nd2_reader import ND2Reader
import numpy as np
import czifile
import dask.array as da
import json
import nd2reader

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

class ControlPanel(QWidget):
    
    EXIT_CODE_REBOOT = -1234567890
       
    #self.displaygui = QtWidgets.QGroupBox()    
    Meta_Data_df = pd.DataFrame()
    
    def controlUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
#         MainWindow.resize(1000, 550)
        font = QtGui.QFont()
        font.setItalic(True)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
     #         layout = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)

        self.gridLayout_centralwidget = QtWidgets.QGridLayout(self.centralwidget)
             
######  Instantiating GUI classes

        self.inout_resource_gui = IO_ResourceGUI.InOut_resource(self.centralwidget,self.gridLayout_centralwidget)
        self.analysisgui = AnalysisGUI.analyzer(self.centralwidget, self.gridLayout_centralwidget)
#         self.gridLayout_centralwidget.addWidget(self.analysisgui, 1, 11, 10, 12)
        self.analysisgui.setEnabled(False)
        self.displaygui = DisplayGUI_Copy1.display()
        self.displaygui.show()
        #self.displaygui = DisplayGUI.display(self.centralwidget)
        self.displaygui.setEnabled(False)
          
        self.inputoutputcontrol = InputOutput.inputoutput_control()
        
#         self.gridLayout_centralwidget.addWidget(self.inout_resource_gui, 1, 1, 4, 10)
        
        self.image_analyzer = Analysis.ImageAnalyzer(self.analysisgui, self.inout_resource_gui)
        #self.ImDisplay = Display.imagedisplayer(self.analysisgui,self.centralwidget)
        self.ImDisplay = Display_Copy1.imagedisplayer(self.analysisgui,self.centralwidget, self.analysisgui)
        self.PlateGrid = GridLayout.gridgenerator(self.centralwidget, self.gridLayout_centralwidget)
        self.PlateGrid.setEnabled(False)
        
        self.CV_Reader = MetaData_Reader.CellVoyager()        
    
#         self.setLayout(self.gridLayout_centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)
        
######  Input Output loader controllers

        self.inout_resource_gui.LoadMetadataButton.clicked.connect(lambda: self.ON_CLICK_LOADBUTTON(self.inout_resource_gui))
        self.inout_resource_gui.LoadImageButton.clicked.connect(lambda: self.ON_CLICK_LOADIMGBUTTON(self.inout_resource_gui))
        self.inout_resource_gui.DisplayCheckBox.stateChanged.connect(lambda:
                                                                     self.ImDisplay.display_initializer(self.Meta_Data_df,
                                                                     self.displaygui, self.inout_resource_gui))
        
        self.inout_resource_gui.DisplayCheckBox.stateChanged.connect(lambda: 
                                                                     self.PlateGrid.GRID_INITIALIZER(self.Meta_Data_df,
                                                                                                     self.displaygui,
                                                                                            self.inout_resource_gui,
                                                                                                    self.ImDisplay))
        self.PlateGrid.tableWidget.itemClicked.connect(lambda: self.PlateGrid.on_click_table(self.Meta_Data_df,
                                                                                                     self.displaygui,
                                                                                            self.inout_resource_gui,
                                                                                                    self.ImDisplay))
        self.PlateGrid.FOVlist.itemClicked.connect(lambda: self.PlateGrid.on_click_list(self.Meta_Data_df,self.ImDisplay, self.displaygui))
        self.PlateGrid.Zlist.itemClicked.connect(lambda: self.PlateGrid.on_click_list(self.Meta_Data_df,self.ImDisplay, self.displaygui))
        self.PlateGrid.Timelist.itemClicked.connect(lambda: self.PlateGrid.on_click_list(self.Meta_Data_df,self.ImDisplay, self.displaygui))
      
        #self.inout_resource_gui.DisplayCheckBox.stateChanged.connect(lambda: INSTANTIATE_DISPLAY())
                                                                     
####### Display GUI controlers
        
#         self.displaygui.ColScroller.sliderMoved.connect(lambda:
#                                                         self.ImDisplay.COL_SCROLLER_MOVE_UPDATE(self.displaygui))
#         self.displaygui.ColSpinBox.valueChanged.connect(lambda: 
#                                                         self.ImDisplay.COL_SPINBOX_UPDATE(self.displaygui))
        
#         self.displaygui.RowScroller.sliderMoved.connect(lambda:
#                                                         self.ImDisplay.ROW_SCROLLER_MOVE_UPDATE(self.displaygui))
        
#         self.displaygui.RowSpinBox.valueChanged.connect(lambda:
#                                                         self.ImDisplay.ROW_SPINBOX_UPDATE(self.displaygui))
        
#         self.displaygui.ZScroller.sliderMoved.connect(lambda: 
#                                                       self.ImDisplay.Z_SCROLLER_MOVE_UPDATE(self.displaygui))
        
#         self.displaygui.ZSpinBox.valueChanged.connect(lambda: 
#                                                       self.ImDisplay.Z_SPINBOX_UPDATE(self.displaygui))
        
#         self.displaygui.FOVScroller.sliderMoved.connect(lambda:
#                                                         self.ImDisplay.FOV_SCROLLER_MOVE_UPDATE(self.displaygui))
#         self.displaygui.FOVSpinBox.valueChanged.connect(lambda: self.ImDisplay.FOV_SPINBOX_UPDATE(self.displaygui))
        
#         self.displaygui.TScroller.sliderMoved.connect(lambda: self.ImDisplay.T_SCROLLER_MOVE_UPDATE(self.displaygui))
#         self.displaygui.TSpinBox.valueChanged.connect(lambda: self.ImDisplay.T_SPINBOX_UPDATE(self.displaygui))
        ###### CHANNELS CHECKBOXES
        self.displaygui.Ch1CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.Ch2CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.Ch3CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.Ch4CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.Ch5CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.displaygui.Ch1maxproject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.Ch2maxproject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.Ch3maxproject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.Ch4maxproject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.Ch5maxproject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
         ###### histogram controllers
        self.displaygui.MaxHistSlider.sliderReleased.connect(lambda:
                                                           self.ImDisplay.MAX_HIST_SLIDER_UPDATE(self.displaygui))
        
        self.displaygui.MinHistSlider.sliderReleased.connect(lambda:
                                                           self.ImDisplay.MIN_HIST_SLIDER_UPDATE(self.displaygui))
        
#         self.displaygui.MinHistSpinBox.valueChanged.connect(lambda:
#                                                             self.ImDisplay.MIN_HIST_SPINBOX_UPDATE(self.displaygui))
        
#         self.displaygui.MaxHistSpinBox.valueChanged.connect(lambda:
#                                                             self.ImDisplay.MAX_HIST_SPINBOX_UPDATE(self.displaygui))
        
        ####### Nuclei and spot visualization controllers
        
        self.displaygui.NuclMaskCheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.NucDetectionSlider.sliderReleased.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.NucSeparationSlider.sliderReleased.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.NucleiAreaSlider.sliderReleased.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.analysisgui.NucDetectMethod.currentIndexChanged.connect(lambda: 
                                                                     self.analysisgui.INITIALIZE_SEGMENTATION_PARAMETERS())
        self.analysisgui.NucDetectMethod.currentIndexChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        # self.analysisgui.NucleiChannel.currentIndexChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.NucleiChannel.activated.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.displaygui.NucPreviewMethod.currentIndexChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        
        self.displaygui.SpotsCheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.analysisgui.SpotCh1CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.SpotCh2CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.SpotCh3CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.SpotCh4CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.SpotCh5CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        
        ####### Analysis Gui Controllers
        self.batchanalysis = BatchAnalyzer.BatchAnalysis(self.analysisgui, self.image_analyzer, self.inout_resource_gui, self.displaygui, self.ImDisplay)
        #self.analysisgui.NucMaxZprojectCheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.NucRemoveBoundaryCheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.analysisgui.SpotMaxZProject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.RunAnalysis.clicked.connect(lambda: self.batchanalysis.ON_APPLYBUTTON(self.Meta_Data_df))
        
        self.analysisgui.ResetButton.clicked.connect(lambda: self.ON_RESET_BUTTON())
        self.analysisgui.ThresholdSlider.sliderReleased.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.SensitivitySpinBox.valueChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.spotanalysismethod.currentIndexChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.analysisgui.ThresholdSlider.sliderReleased.connect(lambda: self.image_analyzer.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.analysisgui.SensitivitySpinBox.valueChanged.connect(lambda: self.image_analyzer.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.analysisgui.SpotPerChSpinBox.valueChanged.connect(lambda: self.image_analyzer.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.analysisgui.SpotPerChSpinBox.valueChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        
        self.analysisgui.SpotareaminSpinBox.valueChanged.connect(lambda: self.image_analyzer.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.analysisgui.SpotareaminSpinBox.valueChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.analysisgui.SpotareamaxSpinBox.valueChanged.connect(lambda: self.image_analyzer.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.analysisgui.SpotareamaxSpinBox.valueChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.analysisgui.SpotIntegratedIntensitySpinBox.valueChanged.connect(lambda: self.image_analyzer.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.analysisgui.SpotIntegratedIntensitySpinBox.valueChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))

        self.analysisgui.spotanalysismethod.currentIndexChanged.connect(lambda: self.image_analyzer.UPDATE_SPOT_ANALYSIS_PARAMS())
        
        self.analysisgui.spotchannelselect.currentIndexChanged.connect(lambda: self.image_analyzer.UPDATE_SPOT_ANALYSIS_GUI_PARAMS())
      #  self.analysisgui.CloseButton.clicked.connect(self.closeEvent)
        ##################
        
        ####### Menu Bar 
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 0))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuTool = QtWidgets.QMenu(self.menubar)
        self.menuTool.setObjectName("menuTool")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad = QtWidgets.QMenu(self.menuFile)
        self.actionLoad.setObjectName("actionLoad")
        self.actionLoad_image = QtWidgets.QAction(self.actionLoad)
        self.actionLoad_image.setObjectName("actionLoad_image")
        self.LoadConfig = QtWidgets.QAction(MainWindow)
        self.LoadConfig.setObjectName("LoadConfig")
        self.saveConfig = QtWidgets.QAction(MainWindow)
        self.saveConfig.setObjectName("saveConfig")
        self.actionexit = QtWidgets.QAction(MainWindow)
        self.actionexit.setObjectName("actionexit")
        self.menuFile.addMenu(self.actionLoad)
        self.actionLoad.addAction(self.actionLoad_image)
        self.menuFile.addAction(self.actionexit)
        self.menuTool.addAction(self.LoadConfig)
        self.menuTool.addAction(self.saveConfig)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTool.menuAction())
        
        self.saveConfig.triggered.connect(lambda:self.analysisgui.file_save(self.image_analyzer))
        self.LoadConfig.triggered.connect(lambda:self.analysisgui.LOAD_CONFIGURATION(self.image_analyzer))
        
        
        
        
        
        self.retranslateUi(MainWindow)
        self.analysisgui.AnalysisMode.setCurrentIndex(self.analysisgui.AnalysisMode.indexOf(self.analysisgui.Results))
        self.inout_resource_gui.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
            

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HiTIPS"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuTool.setTitle(_translate("MainWindow", "Tool"))
        self.actionLoad.setTitle(_translate("MainWindow", "Load"))
        self.actionLoad_image.setText(_translate("MainWindow", "Image"))
        self.LoadConfig.setText(_translate("MainWindow", "Load Configuration"))
        self.saveConfig.setText(_translate("MainWindow", "Save Configuration"))
        self.actionexit.setText(_translate("MainWindow", "exit"))
        
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
                
        PixPerMic_Text= 'Pixel Size = '"{:.2f}".format(float(items_mrf[0].attributes['bts:HorizontalPixelDimension'].value)) + '\u03BC'+'m'
        self.inout_resource_gui.NumFilesLoadedLbl.setText(QtCore.QCoreApplication.translate("MainWindow", PixPerMic_Text))
        self.inout_resource_gui.NumFilesLoadedLbl.setStyleSheet("color: blue")
        

    def ON_CLICK_LOADBUTTON(self, inout_resource_gui):
        if self.inout_resource_gui.DeviceType.currentText()=='CellVoyager':
            options = QtWidgets.QFileDialog.Options()
            self.fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select Metadata Files...',
                                                                    '', "MLF files (*.mlf)"
                                                                    , options=options)
            if self.fnames:
                filename, file_extension = os.path.splitext(self.fnames[0])
                self.READ_FROM_METADATA(self.fnames[0])
                self.inout_resource_gui.DisplayCheckBox.setEnabled(True)
                self.analysisgui.setEnabled(True)
                
        if self.inout_resource_gui.DeviceType.currentText()=='MicroManager':
            options = QtWidgets.QFileDialog.Options()
            self.Data_dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption= "Select Data Directory", options=options)
            
            if self.Data_dir:
                
                self.READ_UMANAGER_METADATA(self.Data_dir)
                self.inout_resource_gui.DisplayCheckBox.setEnabled(True)
                self.analysisgui.setEnabled(True)
    
    def READ_UMANAGER_METADATA(self, metadatadir):
        
        import glob
        import pandas as pd
        import json

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
            
        PixPerMic_Text= 'Pixel Size = '"{:.2f}".format(float(self.Meta_Data_df["PixPerMic"].iloc[0])) + '\u03BC'+'m'
        self.inout_resource_gui.NumFilesLoadedLbl.setText(QtCore.QCoreApplication.translate("MainWindow", PixPerMic_Text))
        self.inout_resource_gui.NumFilesLoadedLbl.setStyleSheet("color: blue")
            
    def ON_CLICK_LOADIMGBUTTON(self, inout_resource_gui):
        
        options = QtWidgets.QFileDialog.Options()
        self.fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select Image Files...',
                                                                '', "Images (*.czi *.tiff *.tif *.nd2)"
                                                                , options=options)
        filenames, file_extensions= [],[]
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
                        for fov_idx in range(sizes['v']):
                            for t_idx in range(sizes['t']):
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
                                
            self.inout_resource_gui.DisplayCheckBox.setEnabled(True)
            self.analysisgui.setEnabled(True)
    
            PixPerMic_Text= 'Pixel Size = '"{:.2f}".format(self.Meta_Data_df['PixPerMic'].iloc[0]) + '\u03BC'+'m'
            self.inout_resource_gui.NumFilesLoadedLbl.setText(QtCore.QCoreApplication.translate("MainWindow", PixPerMic_Text))
            self.inout_resource_gui.NumFilesLoadedLbl.setStyleSheet("color: blue")
            
            
        

#     def ON_CLICK_LOADIMGBUTTON(self, inout_resource_gui):
        
#         options = QtWidgets.QFileDialog.Options()
#         self.fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select Image Files...',
#                                                                 '', "Images (*.nd2)"
#                                                                 , options=options)
#         filenames, file_extensions= [],[]
#         if self.fnames:
            
#             df_cols = ["ImageName", "column", "row", "time_point", "field_index", "z_slice", "channel", 
#                    "x_coordinates", "y_coordinates","z_coordinate", "action_index", "action", "Type", "Time", "PixPerMic"]
#             rows = []
#             field_ind=0
#             col_value=1
#             row_value=1
#             for fname in self.fnames:
#                 field_ind+=1

#                 if pathlib.Path(fname).suffix == '.nd2':
#                     img = ND2Reader(fname)
#                 # else:
        
#                 img = AICSImage(fname)
# #                 img.data  # returns 6D STCZYX numpy array
#                 img.dims  # returns string "STCZYX"
#                 img.shape # returns tuple of dimension sizes in STCZYX order
#                 if 'S' in dir(img.dims):
#                     s_arr = range(img.dims.S)
#                 else:
#                     s_arr= range(1)
#                 for s in s_arr:
#                     for t in range(img.dims.T):
#                         for c in range(img.dims.C):
#                             for z in range(img.dims.Z):
                                
#                                 rows.append({
                                    
#                                     "ImageName": "dask_array", 
#                                      "column":str(col_value), 
#                                      "row": str(row_value), 
#                                      "time_point": str(t+1), 
#                                      "field_index": str(field_ind), 
#                                      "z_slice": str(z+1), 
#                                      "channel": str(c+1),
#                                      "x_coordinates": str(0),
#                                      "y_coordinates": str(0),
#                                      "z_coordinate": str(0),
#                                      "action_index": str(s+1),
#                                      "action": str(1), 
#                                      "Type": img.get_image_dask_data("YX", S=s, C=c, T=t, Z=z), 
#                                      "Time": str(0),
#                                      "PixPerMic": img.physical_pixel_sizes.Y
#                                             })
                                    

#             self.Meta_Data_df = pd.DataFrame(rows, columns = df_cols)
                                
#             self.inout_resource_gui.DisplayCheckBox.setEnabled(True)
#             self.analysisgui.setEnabled(True)
    
#             PixPerMic_Text= 'Pixel Size = '"{:.2f}".format(self.Meta_Data_df['PixPerMic'].iloc[0]) + '\u03BC'+'m'
#             self.inout_resource_gui.NumFilesLoadedLbl.setText(QtCore.QCoreApplication.translate("MainWindow", PixPerMic_Text))
#             self.inout_resource_gui.NumFilesLoadedLbl.setStyleSheet("color: blue")
            
    def ON_RESET_BUTTON(self):
        del self.batchanalysis
        self.batchanalysis = BatchAnalyzer.BatchAnalysis(self.analysisgui, self.image_analyzer, self.inout_resource_gui, self.displaygui, self.ImDisplay)
        QtWidgets.qApp.exit( ControlPanel.EXIT_CODE_REBOOT )
        
           
if __name__ == "__main__":
    
    currentExitCode = ControlPanel.EXIT_CODE_REBOOT
    while currentExitCode == ControlPanel.EXIT_CODE_REBOOT:
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        cp = ControlPanel()
        cp.controlUi(MainWindow)
        MainWindow.show()
        currentExitCode = app.exec_()
        app = None
       # sys.exit(app.exec_())

