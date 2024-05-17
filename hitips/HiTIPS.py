from PyQt5 import QtCore, QtGui, QtWidgets
from . import AnalysisGUI, IO_ResourceGUI, GridLayout, DisplayGUI, Analysis, MetaData_Reader, Display, BatchAnalyzer
from .GUI_parameters import Gui_Params
from PyQt5.QtWidgets import QWidget, QMessageBox
import pandas as pd
from xml.dom import minidom
import os
import sys

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

class ControlPanel(QWidget):
    
    EXIT_CODE_REBOOT = -1234567890
    Meta_Data_df = pd.DataFrame()
    
    def controlUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        font = QtGui.QFont()
        font.setItalic(True)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_centralwidget = QtWidgets.QGridLayout(self.centralwidget)
             
######  Instantiating GUI classes

        self.inout_resource_gui = IO_ResourceGUI.InOut_resource(self.centralwidget,self.gridLayout_centralwidget)
        self.displaygui = DisplayGUI.display()
        self.analysisgui = AnalysisGUI.analyzer(self.centralwidget, self.gridLayout_centralwidget, self.displaygui)
        
        self.gui_params = Gui_Params(self.analysisgui,self.inout_resource_gui, self.displaygui)
        self.analysisgui.set_gui_params(self.gui_params)
        self.ImDisplay = Display.imagedisplayer(self.centralwidget, self.gui_params, self.analysisgui)

        self.gui_params.set_ImDisplay(self.ImDisplay)
        
        self.displaygui.set_analysisgui(self.analysisgui)
        self.displaygui.set_imdisplay(self.ImDisplay)
        self.analysisgui.set_imdisplay(self.ImDisplay)
        self.inout_resource_gui.set_analysisgui(self.analysisgui)
        self.analysisgui.setEnabled(False)
        self.displaygui.show()
        self.displaygui.setEnabled(False)
        self.image_analyzer = Analysis.ImageAnalyzer(self.gui_params.params_dict)
        self.analysisgui.set_image_analyzer(self.image_analyzer)
        self.PlateGrid = GridLayout.gridgenerator(self, self.centralwidget, self.gridLayout_centralwidget, self.displaygui, self.inout_resource_gui, self.ImDisplay)
        self.PlateGrid.setEnabled(False)
        self.ImageReader = MetaData_Reader.ImageReader(self, self.inout_resource_gui, self.displaygui, self.analysisgui)
        self.inout_resource_gui.set_MetaData_Reader(self.ImageReader)
        
        self.batchanalysis = BatchAnalyzer.BatchAnalysis(self.gui_params.params_dict)

        
        MainWindow.setCentralWidget(self.centralwidget)
        
######  Input Output loader controllers
        
        self.inout_resource_gui.DisplayCheckBox.stateChanged.connect(lambda: self.ImDisplay.display_initializer(self.Meta_Data_df,  self.displaygui, self.inout_resource_gui))
        self.inout_resource_gui.DisplayCheckBox.stateChanged.connect(lambda: self.PlateGrid.GRID_INITIALIZER(self.Meta_Data_df, self.displaygui, self.inout_resource_gui, self.ImDisplay))

        self.analysisgui.RunAnalysis.clicked.connect(self.on_run_analysis)
        # self.analysisgui.RunAnalysis.clicked.connect(lambda: self.gui_params.update_values())
        # self.analysisgui.RunAnalysis.clicked.connect(lambda: self.batchanalysis.ON_APPLYBUTTON(self.Meta_Data_df))
        self.analysisgui.ResetButton.clicked.connect(lambda: self.ON_RESET_BUTTON())
      
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
        self.saveConfig.triggered.connect(lambda:self.gui_params.file_save())
        self.LoadConfig.triggered.connect(lambda:self.gui_params.LOAD_CONFIGURATION())       
        self.retranslateUi(MainWindow)
        self.analysisgui.AnalysisMode.setCurrentIndex(self.analysisgui.AnalysisMode.indexOf(self.analysisgui.Results))
        self.inout_resource_gui.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def on_run_analysis(self):
        self.gui_params.update_values()
        self.batchanalysis.update_params_dict(self.gui_params.params_dict)
        self.batchanalysis.ON_APPLYBUTTON(self.select_data_subset())
        
#     def select_data_subset(self):
                
#         sub_metadata = pd.DataFrame(columns = self.Meta_Data_df.columns)
        
#         if list(self.PlateGrid.checked_wells):
            
        
            
#         for well in list(self.PlateGrid.checked_wells):
#             for fov in list(self.PlateGrid.checked_fovs):
#                 for zslice in list(self.PlateGrid.checked_zs):
#                     for timepoint in list(self.PlateGrid.checked_times):
                        
#                         temp_metadata = self.Meta_Data_df.loc[(self.Meta_Data_df["row"] == str(int(well[0]) + 1)) & (self.Meta_Data_df["column"] == str(int(well[1]) + 1))&
#                                                              (self.Meta_Data_df["field_index"] == str(int(fov) + 1)) & (self.Meta_Data_df["z_slice"] == str(int(zslice) + 1)) &
#                                                              (self.Meta_Data_df["time_point"] == str(int(timepoint) + 1))]
                        
#                         sub_metadata = pd.concat([sub_metadata, temp_metadata])

#         return sub_metadata
    
    def select_data_subset(self):
        
        sub_metadata = pd.DataFrame(columns=self.Meta_Data_df.columns)

        if self.PlateGrid.checked_wells:
            well_list = list(self.PlateGrid.checked_wells)
        else:
            well_list = pd.Series([ (str(int(row) - 1), str(int(col) - 1)) for row, col in zip(self.Meta_Data_df['row'], self.Meta_Data_df['column'])]).unique()
        print("well_list:   ", well_list)
        if self.PlateGrid.checked_fovs:
            fov_list = list(self.PlateGrid.checked_fovs)
        else:
            fov_list = self.Meta_Data_df['field_index'].unique()
            fov_list = [str(int(fov) - 1) for fov in fov_list]
        print("fov_list:   ", fov_list)
        if self.PlateGrid.checked_zs:
            zslice_list = list(self.PlateGrid.checked_zs)
        else:
            zslice_list = self.Meta_Data_df['z_slice'].unique()
            zslice_list = [str(int(z) - 1) for z in zslice_list]
        print("zslice_list:   ", zslice_list)
        
        if self.PlateGrid.checked_times:
            timepoint_list = list(self.PlateGrid.checked_times)
        else:
            timepoint_list = self.Meta_Data_df['time_point'].unique()
            timepoint_list = [str(int(t) - 1) for t in timepoint_list]
        print("timepoint_list:   ", timepoint_list)
        
        for well in well_list:
            for fov in fov_list:
                for zslice in zslice_list:
                    for timepoint in timepoint_list:

                        temp_metadata = self.Meta_Data_df.loc[
                                (self.Meta_Data_df["row"] == str(int(well[0]) + 1)) &
                                (self.Meta_Data_df["column"] == str(int(well[1]) + 1)) &
                                (self.Meta_Data_df["field_index"] == str(int(fov) + 1)) &
                                (self.Meta_Data_df["z_slice"] == str(int(zslice) + 1)) &
                                (self.Meta_Data_df["time_point"] == str(int(timepoint) + 1))]
                        print("temp_metadata:    ", temp_metadata)  
                        sub_metadata = pd.concat([sub_metadata, temp_metadata]).reset_index(drop=True)
        print("sub_metadata:  ", sub_metadata)
        return sub_metadata
    
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
        
    def ON_RESET_BUTTON(self):
        del self.batchanalysis
        self.batchanalysis = BatchAnalyzer.BatchAnalysis(self.gui_params.params_dict)
        QtWidgets.qApp.exit( ControlPanel.EXIT_CODE_REBOOT )
        
def main():
    currentExitCode = ControlPanel.EXIT_CODE_REBOOT
    while currentExitCode == ControlPanel.EXIT_CODE_REBOOT:
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        cp = ControlPanel()
        cp.controlUi(MainWindow)
        MainWindow.show()
        currentExitCode = app.exec_()
        app = None
    
if __name__ == "__main__":

    main()