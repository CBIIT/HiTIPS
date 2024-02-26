from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
import pandas as pd
import numpy as np
from distutils import util
import sys
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

class analyzer(QWidget):

    def __init__(self, centralwidget, gridLayout_centralwidget, displaygui=None, ImDisplay=None, image_analyzer=None, gui_params=None):
        super().__init__(centralwidget)
        self.gridLayout_centralwidget = gridLayout_centralwidget
        self.displaygui = displaygui
        self.AnalysisLbl = QtWidgets.QLabel(centralwidget)
        self.ImDisplay = ImDisplay  
        self.image_analyzer = image_analyzer
        self.gui_params = gui_params
        
        self.gridLayout_centralwidget.addWidget(self.AnalysisLbl, 1, 17, 1, 3)
        font = QtGui.QFont()
        font.setFamily(".Farah PUA")
        font.setPointSize(20)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        font.setKerning(False)
        self.AnalysisLbl.setFont(font)
        self.AnalysisLbl.setObjectName("AnalysisLbl")
        
        
        self.RunAnalysis = QtWidgets.QPushButton(centralwidget)
        self.gridLayout_centralwidget.addWidget(self.RunAnalysis, 16, 18, 1, 2)
        self.RunAnalysis.setObjectName("RunAnalysis")
        
        self.ResetButton = QtWidgets.QPushButton(centralwidget)
        self.gridLayout_centralwidget.addWidget(self.ResetButton, 16, 21, 1, 2)
        self.ResetButton.setObjectName("ResetButton")
        

        
        self.AnalysisMode = QtWidgets.QToolBox(centralwidget)
        self.gridLayout_centralwidget.addWidget(self.AnalysisMode, 2, 11, 10, 15)
        self.gridLayout_AnalysisMode = QtWidgets.QGridLayout(self.AnalysisMode)
        self.gridLayout_AnalysisMode.setObjectName("gridLayout_AnalysisMode")
        
        font = QtGui.QFont()
        font.setPointSize(14)
        self.AnalysisMode.setFont(font)
        self.AnalysisMode.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.AnalysisMode.setFrameShadow(QtWidgets.QFrame.Plain)
        self.AnalysisMode.setObjectName("AnalysisMode")
        self.NucleiDetection = QtWidgets.QWidget()
        self.gridLayout_NucleiDetection = QtWidgets.QGridLayout(self.NucleiDetection)
        self.gridLayout_NucleiDetection.setObjectName("gridLayout_NucleiDetection")
        self.gridLayout_AnalysisMode.addWidget(self.NucleiDetection, 0, 0, 1, 1)
        ####################################
        ####### Nuclei Detection
        self.NucleiDetection.setObjectName("NucleiDetection")
        self.NucleiChLbl = QtWidgets.QLabel(self.NucleiDetection)
        self.gridLayout_NucleiDetection.addWidget(self.NucleiChLbl, 0, 0)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NucleiChLbl.setFont(font)
        self.NucleiChLbl.setObjectName("NucleiChLbl")
        self.NucDetectMethodLbl = QtWidgets.QLabel(self.NucleiDetection)

        self.gridLayout_NucleiDetection.addWidget(self.NucDetectMethodLbl, 1, 0)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NucDetectMethodLbl.setFont(font)
        self.NucDetectMethodLbl.setObjectName("NucDetectMethodLbl")
        self.NucleiChannel = QtWidgets.QComboBox(self.NucleiDetection)
        self.gridLayout_NucleiDetection.addWidget(self.NucleiChannel, 0, 1)
        self.NucleiChannel.setObjectName("NucleiChannel")
        self.NucleiChannel.addItem("Channel 1")
        self.NucleiChannel.addItem("Channel 2")
        self.NucleiChannel.addItem("Channel 3")
        self.NucleiChannel.addItem("Channel 4")
        self.NucleiChannel.addItem("Channel 5")
        self.NucleiChannel.setFont(font)
        self.NucleiChannel.activated.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.NucDetectMethod = QtWidgets.QComboBox(self.NucleiDetection)
        self.gridLayout_NucleiDetection.addWidget(self.NucDetectMethod, 1, 1)
        self.NucDetectMethod.setObjectName("NucDetectMethod")
        self.NucDetectMethod.addItem("ImageProc")
        self.NucDetectMethod.addItem("Cascade_watershed")
        self.NucDetectMethod.addItem("CellPose-CPU")
        self.NucDetectMethod.addItem("CellPose-GPU")
        self.NucDetectMethod.addItem("CellPose-Cyto")
        self.NucDetectMethod.addItem("DeepCell")
        self.NucDetectMethod.setFont(font)
        self.NucDetectMethod.currentIndexChanged.connect(lambda: self.INITIALIZE_SEGMENTATION_PARAMETERS())
        self.NucDetectMethod.currentIndexChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
#         self.verticalSpacer = QtWidgets.QSpacerItem(50, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
#         self.gridLayout_NucleiDetection.addItem(self.verticalSpacer, 1, 3, 1, 1)
        
        
        self.NucFirstThreshLbl = QtWidgets.QLabel(self.NucleiDetection)
      #  self.NucFirstThreshLbl.setGeometry(QtCore.QRect(10, 60, 61, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucFirstThreshLbl, 2, 0)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NucFirstThreshLbl.setFont(font)
        self.NucFirstThreshLbl.setObjectName("NucFirstThreshLbl")
        
        self.NucDetectionSlider = QtWidgets.QSlider(self.NucleiDetection)
        self.gridLayout_NucleiDetection.addWidget(self.NucDetectionSlider, 2, 1)
        self.NucDetectionSlider.setOrientation(QtCore.Qt.Horizontal)
        self.NucDetectionSlider.setObjectName("NucDetectionSlider")
        self.NucDetectionSlider.setMaximum(100)
        self.NucDetectionSlider.setMinimum(0)
        self.NucDetectionSlider.setValue(42)
        self.NucDetectionSlider.valueChanged.connect(lambda: self.SECOND_THRESH_LABEL_UPDATE())
        self.NucDetectionSlider.sliderReleased.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.NucSecondThreshSliderValue = QtWidgets.QLabel(self.NucleiDetection)
#         self.NucFirstThreshSliderValue.setGeometry(QtCore.QRect(10, 60, 61, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucSecondThreshSliderValue, 2, 2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NucSecondThreshSliderValue.setFont(font)
        self.NucSecondThreshSliderValue.setObjectName("NucSecondThreshSliderValue")
        self.NucSecondThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucDetectionSlider.value()))) 

        
        self.NucSeparationSlider = QtWidgets.QSlider(self.NucleiDetection)
        self.gridLayout_NucleiDetection.addWidget(self.NucSeparationSlider, 3, 1)
        self.NucSeparationSlider.setOrientation(QtCore.Qt.Horizontal)
        self.NucSeparationSlider.setObjectName("NucSeparationSlider")
        self.NucSeparationSlider.setMaximum(100)
        self.NucSeparationSlider.setMinimum(0)
        self.NucSeparationSlider.setValue(39)
        self.NucSeparationSlider.valueChanged.connect(lambda: self.FIRST_THRESH_LABEL_UPDATE())
        self.NucSeparationSlider.sliderReleased.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.NucFirstThreshSliderValue = QtWidgets.QLabel(self.NucleiDetection)
        self.gridLayout_NucleiDetection.addWidget(self.NucFirstThreshSliderValue, 3, 2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NucFirstThreshSliderValue.setFont(font)
        self.NucFirstThreshSliderValue.setObjectName("NucFirstThreshSliderValue")
        self.NucFirstThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucSeparationSlider.value()))) 
        
        self.NucSecondThreshLbl = QtWidgets.QLabel(self.NucleiDetection)
        self.gridLayout_NucleiDetection.addWidget(self.NucSecondThreshLbl, 3, 0)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NucSecondThreshLbl.setFont(font)
        self.NucSecondThreshLbl.setObjectName("NucSecondThreshLbl")
        
                
        
        self.NucleiareaLbl = QtWidgets.QLabel(self.NucleiDetection)
        self.gridLayout_NucleiDetection.addWidget(self.NucleiareaLbl, 4, 0)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NucleiareaLbl.setFont(font)
        self.NucleiareaLbl.setObjectName("NucleiareaLbl")
        
        self.NucleiAreaSlider = QtWidgets.QSlider(self.NucleiDetection)
        self.gridLayout_NucleiDetection.addWidget(self.NucleiAreaSlider, 4, 1)
        self.NucleiAreaSlider.setOrientation(QtCore.Qt.Horizontal)
        self.NucleiAreaSlider.setObjectName("NucleiAreaSlider")
        self.NucleiAreaSlider.setMaximum(1000)
        self.NucleiAreaSlider.setMinimum(0)
        self.NucleiAreaSlider.setValue(30)
        self.NucleiAreaSlider.valueChanged.connect(lambda: self.NUCLEI_AREA_LABEL_UPDATE())
        self.NucleiAreaSlider.sliderReleased.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.NucleiAreaSliderValue = QtWidgets.QLabel(self.NucleiDetection)
        self.gridLayout_NucleiDetection.addWidget(self.NucleiAreaSliderValue, 4, 2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NucleiAreaSliderValue.setFont(font)
        self.NucleiAreaSliderValue.setObjectName("NucleiAreaSliderValue")
        self.NucleiAreaSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucleiAreaSlider.value()))) 
        
        self.NucRemoveBoundaryCheckBox = QtWidgets.QCheckBox(self.NucleiDetection)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.gridLayout_NucleiDetection.addWidget(self.NucRemoveBoundaryCheckBox, 5, 0,1, 3)
        self.NucRemoveBoundaryCheckBox.setObjectName("NucRemoveBoundaryCheckBox")
        self.NucRemoveBoundaryCheckBox.setChecked(False)
        self.NucRemoveBoundaryCheckBox.setFont(font)
        self.NucRemoveBoundaryCheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.NucMaxZprojectCheckBox = QtWidgets.QCheckBox(self.NucleiDetection)
        self.gridLayout_NucleiDetection.addWidget(self.NucMaxZprojectCheckBox, 6, 0,1, 2)
        self.NucMaxZprojectCheckBox.setObjectName("NucMaxZprojectCheckBox")
        self.NucMaxZprojectCheckBox.setChecked(True)
        self.NucMaxZprojectCheckBox.setFont(font)
        
        self.AnalysisMode.addItem(self.NucleiDetection, "")
        
        #### Cell Boundary GUI
        self.CellBoundary = QtWidgets.QWidget()
#         self.CellBoundary.setGeometry(QtCore.QRect(0, 0, 381, 171))
        self.gridLayout_AnalysisMode.addWidget(self.CellBoundary, 1, 0, 1, 1)
        self.gridLayout_CellBoundary = QtWidgets.QGridLayout(self.CellBoundary)
        self.gridLayout_CellBoundary.setObjectName("gridLayout_CellBoundary")
        
        self.CellBoundary.setObjectName("CellBoundary")
        self.CytoChLbl = QtWidgets.QLabel(self.CellBoundary)
        self.gridLayout_CellBoundary.addWidget(self.CytoChLbl, 0, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.CytoChLbl.setFont(font)
        self.CytoChLbl.setObjectName("CytoChLbl")
        self.CytoCellTypeLbl = QtWidgets.QLabel(self.CellBoundary)
        self.gridLayout_CellBoundary.addWidget(self.CytoCellTypeLbl, 1, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.CytoCellTypeLbl.setFont(font)
        self.CytoCellTypeLbl.setObjectName("CytoCellTypeLbl")
        self.CytoDetectMethodLbl = QtWidgets.QLabel(self.CellBoundary)
        self.gridLayout_CellBoundary.addWidget(self.CytoDetectMethodLbl, 2, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.CytoDetectMethodLbl.setFont(font)
        self.CytoDetectMethodLbl.setObjectName("CytoDetectMethodLbl")
        self.CytoChannel = QtWidgets.QComboBox(self.CellBoundary)
        self.gridLayout_CellBoundary.addWidget(self.CytoChannel, 0, 1, 1, 2)
        
        self.CytoChannel.setObjectName("CytoChannel")
        self.CytoChannel.addItem("")
        self.CytoChannel.addItem("")
        self.CytoChannel.addItem("")
        self.CytoChannel.addItem("")
        self.CytoChannel.setFont(font)
        
        self.CytoDetectMethod = QtWidgets.QComboBox(self.CellBoundary)
#         self.CytoDetectMethod.setGeometry(QtCore.QRect(90, 60, 211, 31))
        self.gridLayout_CellBoundary.addWidget(self.CytoDetectMethod, 2, 1, 1, 2)
        self.CytoDetectMethod.setObjectName("CytoDetectMethod")
        self.CytoDetectMethod.addItem("")
        self.CytoDetectMethod.setFont(font)
        
        self.CytoCellType = QtWidgets.QComboBox(self.CellBoundary)
#         self.CytoCellType.setGeometry(QtCore.QRect(90, 30, 211, 31))
        self.gridLayout_CellBoundary.addWidget(self.CytoCellType, 1, 1, 1, 2)
        self.CytoCellType.setObjectName("CytoCellType")
        self.CytoCellType.setFont(font)
        
        self.CytoCellType.addItem("")
        
        
        self.AnalysisMode.addItem(self.CellBoundary, "")
        #############################################################################
        #### Secondary measurement
        self.secondarymeasurement = QtWidgets.QWidget()
#         self.secondarymeasurement.setGeometry(QtCore.QRect(0, 0, 381, 171))
        self.gridLayout_AnalysisMode.addWidget(self.secondarymeasurement, 2, 0, 1, 1)
        self.gridLayout_secondarymeasurement = QtWidgets.QGridLayout(self.secondarymeasurement)
        self.gridLayout_secondarymeasurement.setObjectName("gridLayout_secondarymeasurement")
        
        self.secondarymeasurement.setObjectName("secondarymeasurement")
        self.SecChLbl = QtWidgets.QLabel(self.secondarymeasurement)
#         self.CytoChLbl.setGeometry(QtCore.QRect(20, 0, 61, 31))
        self.gridLayout_secondarymeasurement.addWidget(self.SecChLbl, 0, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SecChLbl.setFont(font)
        self.SecChLbl.setObjectName("SecChLbl")
        self.SecAreaLbl = QtWidgets.QLabel(self.secondarymeasurement)
        self.gridLayout_secondarymeasurement.addWidget(self.SecAreaLbl, 0, 2, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SecAreaLbl.setFont(font)
        self.SecAreaLbl.setObjectName("SecAreaLbl")
        
        self.SecChannel = QtWidgets.QComboBox(self.secondarymeasurement)
        self.gridLayout_secondarymeasurement.addWidget(self.SecChannel, 1, 0, 1, 2)
        
        self.SecChannel.setObjectName("SecChannel")
        self.SecChannel.addItem("")
        self.SecChannel.addItem("ch1")
        self.SecChannel.addItem("ch2")
        self.SecChannel.addItem("ch3")
        self.SecChannel.addItem("ch4")
        self.SecChannel.addItem("ch5")
        self.SecChannel.setFont(font)
        
        self.SecArea = QtWidgets.QComboBox(self.secondarymeasurement)
        self.gridLayout_secondarymeasurement.addWidget(self.SecArea, 1, 2, 1, 2)
        self.SecArea.setObjectName("SecArea")
        self.SecArea.addItem("")
        self.SecArea.addItem("nuclei")
        self.SecArea.setFont(font)
        
        self.addsecmeasurement = QtWidgets.QPushButton(self.secondarymeasurement)
        self.gridLayout_secondarymeasurement.addWidget(self.addsecmeasurement, 2, 1, 1, 2)
        self.addsecmeasurement.setObjectName("addsecmeasurement")

        self.AnalysisMode.addItem(self.secondarymeasurement, "")
        #############################################################################
        
        
        
        #### Spot Detection 
        self.SpotDetection = QtWidgets.QWidget()
        self.gridLayout_AnalysisMode.addWidget(self.SpotDetection, 2, 0, 1, 1)
        self.gridLayout_SpotDetection = QtWidgets.QGridLayout(self.SpotDetection)
        self.gridLayout_SpotDetection.setObjectName("gridLayout_SpotDetection")
        self.SpotDetection.setObjectName("SpotDetection")
#         shift = 70

        self.SpotCh1CheckBox = QtWidgets.QCheckBox(self.SpotDetection)
        self.gridLayout_SpotDetection.addWidget(self.SpotCh1CheckBox, 0, 0, 1, 1)
        self.SpotCh1CheckBox.setObjectName("Ch1CheckBox")
        self.SpotCh1CheckBox.setStyleSheet("color: gray")
        self.SpotCh1CheckBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.SpotCh1CheckBox.customContextMenuRequested.connect(lambda: self.displaygui.createContextMenu("Ch1_spot").exec_(QtGui.QCursor.pos()))
        self.SpotCh1CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
       
        self.SpotCh2CheckBox = QtWidgets.QCheckBox(self.SpotDetection)
        self.gridLayout_SpotDetection.addWidget(self.SpotCh2CheckBox, 0, 1, 1, 1)
        self.SpotCh2CheckBox.setObjectName("Ch2CheckBox")
        self.SpotCh2CheckBox.setStyleSheet("color: green")
        self.SpotCh2CheckBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.SpotCh2CheckBox.customContextMenuRequested.connect(lambda: self.displaygui.createContextMenu("Ch2_spot").exec_(QtGui.QCursor.pos()))
        self.SpotCh2CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.SpotCh3CheckBox = QtWidgets.QCheckBox(self.SpotDetection)
        self.gridLayout_SpotDetection.addWidget(self.SpotCh3CheckBox, 0, 2, 1, 1)
        self.SpotCh3CheckBox.setObjectName("Ch3CheckBox")
        self.SpotCh3CheckBox.setStyleSheet("color: red")
        self.SpotCh3CheckBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.SpotCh3CheckBox.customContextMenuRequested.connect(lambda: self.displaygui.createContextMenu("Ch3_spot").exec_(QtGui.QCursor.pos()))
        self.SpotCh3CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.SpotCh4CheckBox = QtWidgets.QCheckBox(self.SpotDetection)
        self.gridLayout_SpotDetection.addWidget(self.SpotCh4CheckBox, 0, 3, 1, 1)
        self.SpotCh4CheckBox.setObjectName("Ch4CheckBox")
        self.SpotCh4CheckBox.setStyleSheet("color: blue")
        self.SpotCh4CheckBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.SpotCh4CheckBox.customContextMenuRequested.connect(lambda: self.displaygui.createContextMenu("Ch4_spot").exec_(QtGui.QCursor.pos()))
        self.SpotCh4CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.SpotCh5CheckBox = QtWidgets.QCheckBox(self.SpotDetection)
        self.gridLayout_SpotDetection.addWidget(self.SpotCh5CheckBox, 0, 4, 1, 1)
        self.SpotCh5CheckBox.setObjectName("Ch5CheckBox")
        self.SpotCh5CheckBox.setStyleSheet("color: orange")
        self.SpotCh5CheckBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.SpotCh5CheckBox.customContextMenuRequested.connect(lambda: self.displaygui.createContextMenu("Ch5_spot").exec_(QtGui.QCursor.pos()))
        self.SpotCh5CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.SpotLocationLbl = QtWidgets.QLabel(self.SpotDetection)
        self.gridLayout_SpotDetection.addWidget(self.SpotLocationLbl, 2, 0, 1, 2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SpotLocationLbl.setFont(font)
        
        self.SpotLocationCbox = QtWidgets.QComboBox(self.SpotDetection)
#         self.SpotLocationCbox.setGeometry(QtCore.QRect(90, 70, 211, 31))
        self.gridLayout_SpotDetection.addWidget(self.SpotLocationCbox, 2, 2, 1, 3)
        self.SpotLocationCbox.setObjectName("SpotLocationCbox")
        self.SpotLocationCbox.addItem("Center Of Mass")
        self.SpotLocationCbox.addItem("Max Intensity")
        self.SpotLocationCbox.addItem("Centroid")
        self.SpotLocationCbox.setFont(font)
        
        self.IntegratedIntensity = QtWidgets.QLabel(self.SpotDetection)
#         self.SpotLocationLbl.setGeometry(QtCore.QRect(3, 75, 80, 20))
        self.gridLayout_SpotDetection.addWidget(self.IntegratedIntensity, 3, 0, 1, 2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.IntegratedIntensity.setFont(font)
        
        self.IntegratedIntensityCbox = QtWidgets.QComboBox(self.SpotDetection)
#         self.SpotLocationCbox.setGeometry(QtCore.QRect(90, 70, 211, 31))
        self.gridLayout_SpotDetection.addWidget(self.IntegratedIntensityCbox, 3, 2, 1, 3)
        self.IntegratedIntensityCbox.setObjectName("IntegratedIntensityCbox")
        self.IntegratedIntensityCbox.addItem("Predefined Gaussian Mask")
        self.IntegratedIntensityCbox.addItem("Fit Gaussian Mask")
        self.IntegratedIntensityCbox.setFont(font)
        
        self.PSFsize = QtWidgets.QLabel(self.SpotDetection)
        self.gridLayout_SpotDetection.addWidget(self.PSFsize, 4, 0, 1, 2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.PSFsize.setFont(font)
        
        self.PSFsizeSpinBox = QtWidgets.QDoubleSpinBox(self.SpotDetection)
        self.gridLayout_SpotDetection.addWidget(self.PSFsizeSpinBox, 4, 2, 1, 3)
        self.PSFsizeSpinBox.setDecimals(2)
        self.PSFsizeSpinBox.setSingleStep(0.05)
        self.PSFsizeSpinBox.setValue(1.6)
        
        self.SpotMaxZProject = QtWidgets.QCheckBox(self.SpotDetection)
        self.SpotMaxZProject.setChecked(True)
        self.gridLayout_SpotDetection.addWidget(self.SpotMaxZProject, 5, 0, 1, 3)
        self.SpotMaxZProject.setObjectName("SpotMaxZProject")
        self.SpotMaxZProject.setFont(font)
        self.SpotMaxZProject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.RemoveBrightJunk = QtWidgets.QCheckBox(self.SpotDetection)
        self.RemoveBrightJunk.setChecked(False)
        self.gridLayout_SpotDetection.addWidget(self.RemoveBrightJunk, 6, 0, 1, 3)
        self.RemoveBrightJunk.setObjectName("RemoveBrightJunk")
        self.RemoveBrightJunk.setFont(font)
        self.AnalysisMode.addItem(self.SpotDetection, "")
        #######################################################
        #### Spot Analysis
        self.SpotAnalysis = QtWidgets.QWidget()
#         self.SpotAnalysis.setGeometry(QtCore.QRect(0, 0, 381, 171))
        self.gridLayout_AnalysisMode.addWidget(self.SpotAnalysis, 3, 0, 1, 1)
        self.gridLayout_SpotAnalysis = QtWidgets.QGridLayout(self.SpotAnalysis)
        self.SpotAnalysis.setObjectName("gridLayout_SpotAnalysis")

        self.SpotAnalysis.setObjectName("SpotAnalysis")
        
        self.spotchannelselectlbl = QtWidgets.QLabel(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.spotchannelselectlbl, 0, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.spotchannelselectlbl.setFont(font)
        self.spotchannelselectlbl.setObjectName("spotchannelselectlbl")
        
        self.spotchannelselect = QtWidgets.QComboBox(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.spotchannelselect, 0, 1, 1, 1)
        self.spotchannelselect.setObjectName("NucleiChannel")
        self.spotchannelselect.addItem("All")
        self.spotchannelselect.addItem("Ch1")
        self.spotchannelselect.addItem("Ch2")
        self.spotchannelselect.addItem("Ch3")
        self.spotchannelselect.addItem("Ch4")
        self.spotchannelselect.addItem("Ch5")
        self.spotchannelselect.setFont(font)
        
        
        self.spotanalysismethodLbl = QtWidgets.QLabel(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.spotanalysismethodLbl, 1, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.spotanalysismethodLbl.setFont(font)
        self.spotanalysismethodLbl.setObjectName("spotanalysismethodLbl")
        self.thresholdmethodLbl = QtWidgets.QLabel(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.thresholdmethodLbl, 2, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.thresholdmethodLbl.setFont(font)
        self.thresholdmethodLbl.setObjectName("thresholdmethodLbl")
        self.thresholdvalueLbl = QtWidgets.QLabel(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.thresholdvalueLbl, 3, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.thresholdvalueLbl.setFont(font)
        self.thresholdvalueLbl.setObjectName("thresholdvalueLbl")
        
        
        self.spotanalysismethod = QtWidgets.QComboBox(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.spotanalysismethod, 1, 1, 1, 1)
        self.spotanalysismethod.setObjectName("NucleiChannel")
        self.spotanalysismethod.addItem("LOG")
        self.spotanalysismethod.addItem("Gaussian")
        self.spotanalysismethod.addItem("IntensityThreshold")
        self.spotanalysismethod.addItem("EnhancedLOG")
        self.spotanalysismethod.setFont(font)
        self.spotanalysismethod.currentIndexChanged.connect(lambda: self.gui_params.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.spotanalysismethod.currentIndexChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        
        self.thresholdmethod = QtWidgets.QComboBox(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.thresholdmethod, 2, 1, 1, 1)
        self.thresholdmethod.setObjectName("thresholdmethod")
        self.thresholdmethod.addItem("Auto")
        self.thresholdmethod.addItem("Manual")
        self.thresholdmethod.setFont(font)
        
        self.ThresholdSlider = QtWidgets.QSlider(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.ThresholdSlider, 3, 1, 1, 1)
        self.ThresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.ThresholdSlider.setObjectName("ThresholdSlider")
        self.ThresholdSlider.valueChanged.connect(lambda: self.SPOT_THRESH_LABEL_UPDATE())
        self.ThresholdSlider.sliderReleased.connect(lambda: self.gui_params.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.ThresholdSlider.sliderReleased.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        

        self.SpotThreshSliderValue = QtWidgets.QLabel(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.SpotThreshSliderValue, 3, 2, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SpotThreshSliderValue.setFont(font)
        self.SpotThreshSliderValue.setObjectName("SpotThreshSliderValue")
        self.SpotThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.ThresholdSlider.value()))) 
        self.ThresholdSlider.setMaximum(100)
        self.ThresholdSlider.setMinimum(0)
        self.ThresholdSlider.setValue(100)

        self.sensitivityLbl = QtWidgets.QLabel(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.sensitivityLbl, 4, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.sensitivityLbl.setFont(font)
        self.sensitivityLbl.setObjectName("sensitivityLbl")
        
        self.SensitivitySpinBox = QtWidgets.QSpinBox(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.SensitivitySpinBox, 4, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SensitivitySpinBox.setFont(font)
        self.SensitivitySpinBox.setObjectName("SensitivitySpinBox")
        self.SensitivitySpinBox.setStyleSheet("color: red")
        self.SensitivitySpinBox.setMaximum(9)
        self.SensitivitySpinBox.setMinimum(1)
        self.SensitivitySpinBox.setValue(3)
        self.SensitivitySpinBox.valueChanged.connect(lambda: self.gui_params.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.SensitivitySpinBox.valueChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        
        
        
        self.ResizeFactor = QtWidgets.QSpinBox(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.ResizeFactor, 5, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.ResizeFactor.setFont(font)
        self.ResizeFactor.setObjectName("ResizeFactor")
        self.ResizeFactor.setValue(1)
        self.ResizeFactor.valueChanged.connect(lambda: self.gui_params.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.ResizeFactor.valueChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        
        self.ResizeFactorLabel = QtWidgets.QLabel(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.ResizeFactorLabel, 5, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.ResizeFactorLabel.setFont(font)
        
        
        self.SpotareaminLbl = QtWidgets.QLabel(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.SpotareaminLbl, 6, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SpotareaminLbl.setFont(font)    
        
        self.SpotareaminSpinBox = QtWidgets.QSpinBox(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.SpotareaminSpinBox, 6, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SpotareaminSpinBox.setFont(font)
        self.SpotareaminSpinBox.setObjectName("SpotareaminSpinBox")
        self.SpotareaminSpinBox.setValue(0)
        self.SpotareaminSpinBox.valueChanged.connect(lambda: self.gui_params.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.SpotareaminSpinBox.valueChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
          
        self.SpotareamaxLbl = QtWidgets.QLabel(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.SpotareamaxLbl, 6, 2, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SpotareamaxLbl.setFont(font)    
        
        self.SpotareamaxSpinBox = QtWidgets.QSpinBox(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.SpotareamaxSpinBox, 6, 3, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SpotareamaxSpinBox.setFont(font)
        self.SpotareamaxSpinBox.setObjectName("SpotareamaxSpinBox")
        self.SpotareamaxSpinBox.setValue(99)
        self.SpotareamaxSpinBox.valueChanged.connect(lambda: self.gui_params.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.SpotareamaxSpinBox.valueChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        
        self.SpotIntegratedIntensityLbl = QtWidgets.QLabel(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.SpotIntegratedIntensityLbl, 7, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SpotIntegratedIntensityLbl.setFont(font)    
        
        self.SpotIntegratedIntensitySpinBox = QtWidgets.QSpinBox(self.SpotAnalysis)
        self.gridLayout_SpotAnalysis.addWidget(self.SpotIntegratedIntensitySpinBox, 7, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SpotIntegratedIntensitySpinBox.setFont(font)
        self.SpotIntegratedIntensitySpinBox.setObjectName("SpotIntegratedIntensitySpinBox")
        self.SpotIntegratedIntensitySpinBox.setMinimum(0)
        self.SpotIntegratedIntensitySpinBox.setMaximum(200000)
        self.SpotIntegratedIntensitySpinBox.setValue(100) 
        self.SpotIntegratedIntensitySpinBox.valueChanged.connect(lambda: self.gui_params.UPDATE_SPOT_ANALYSIS_PARAMS())
        self.SpotIntegratedIntensitySpinBox.valueChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        
        self.AnalysisMode.addItem(self.SpotAnalysis, "")
        #############################################
        #### Nuclei and spot tracking
        self.nuc_spot_track = QtWidgets.QWidget()
#         self.CellBoundary.setGeometry(QtCore.QRect(0, 0, 381, 171))
        self.gridLayout_AnalysisMode.addWidget(self.nuc_spot_track, 4, 0, 1, 1)
        self.gridLayout_nuc_spot_track = QtWidgets.QGridLayout(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.setObjectName("gridLayout_nuc_spot_track")
        
        self.nuc_spot_track.setObjectName("nuc_spot_track")
        self.NucTrackMethodlbl = QtWidgets.QLabel(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.NucTrackMethodlbl, 0, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NucTrackMethodlbl.setFont(font)
        self.NucTrackMethodlbl.setObjectName("NucTrackMethodlbl")
        
        self.NucSearchRadiuslbl = QtWidgets.QLabel(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.NucSearchRadiuslbl, 1, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NucSearchRadiuslbl.setFont(font)
        self.NucSearchRadiuslbl.setObjectName("NucSearchRadiuslbl")
        
        
        self.NucTrackMethod = QtWidgets.QComboBox(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.NucTrackMethod, 0, 1, 1, 2)
        
        self.NucTrackMethod.setObjectName("NucTrackMethod")
        self.NucTrackMethod.addItem("Bayesian")
        self.NucTrackMethod.addItem("DeepCell")
        self.NucTrackMethod.setFont(font)
        
        self.NucSearchRadiusSpinbox = QtWidgets.QSpinBox(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.NucSearchRadiusSpinbox, 1, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NucSearchRadiusSpinbox.setFont(font)
        self.NucSearchRadiusSpinbox.setObjectName("NucSearchRadiusSpinbox")
        self.NucSearchRadiusSpinbox.setMaximum(500)
        self.NucSearchRadiusSpinbox.setMinimum(1)
        self.NucSearchRadiusSpinbox.setValue(100)
        
        self.SpotSearchRadiuslbl = QtWidgets.QLabel(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.SpotSearchRadiuslbl, 2, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SpotSearchRadiuslbl.setFont(font)
        self.SpotSearchRadiuslbl.setObjectName("SpotSearchRadiuslbl")

        
        self.SpotSearchRadiusSpinbox = QtWidgets.QSpinBox(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.SpotSearchRadiusSpinbox, 2, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SpotSearchRadiusSpinbox.setFont(font)
        self.SpotSearchRadiusSpinbox.setObjectName("SpotSearchRadiusSpinbox")
        self.SpotSearchRadiusSpinbox.setMaximum(500)
        self.SpotSearchRadiusSpinbox.setMinimum(1)
        self.SpotSearchRadiusSpinbox.setValue(4)
        
        self.Sec_SpotSearchRadiuslbl = QtWidgets.QLabel(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.Sec_SpotSearchRadiuslbl, 3, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Sec_SpotSearchRadiuslbl.setFont(font)
        self.Sec_SpotSearchRadiuslbl.setObjectName("Sec_SpotSearchRadiuslbl")
        
        self.Sec_SpotSearchRadiusSpinbox = QtWidgets.QSpinBox(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.Sec_SpotSearchRadiusSpinbox, 3, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Sec_SpotSearchRadiusSpinbox.setFont(font)
        self.Sec_SpotSearchRadiusSpinbox.setObjectName("Sec_SpotSearchRadiusSpinbox")
        self.Sec_SpotSearchRadiusSpinbox.setMaximum(500)
        self.Sec_SpotSearchRadiusSpinbox.setMinimum(1)
        self.Sec_SpotSearchRadiusSpinbox.setValue(4)
        
        self.MintrackLengthLbl = QtWidgets.QLabel(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.MintrackLengthLbl, 4, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.MintrackLengthLbl.setFont(font)
        self.MintrackLengthLbl.setObjectName("MintrackLengthLbl")
        
        self.MintrackLengthSpinbox = QtWidgets.QSpinBox(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.MintrackLengthSpinbox, 4, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.MintrackLengthSpinbox.setFont(font)
        self.MintrackLengthSpinbox.setObjectName("MintrackLengthSpinbox")
        self.MintrackLengthSpinbox.setMaximum(50000)
        self.MintrackLengthSpinbox.setMinimum(1)
        self.MintrackLengthSpinbox.setValue(100)
        
        
        self.RegistrationmethodLbl = QtWidgets.QLabel(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.RegistrationmethodLbl, 5, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.RegistrationmethodLbl.setFont(font)
        self.RegistrationmethodLbl.setObjectName("RegistrationmethodLbl")
        
        self.Registrationmethod = QtWidgets.QComboBox(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.Registrationmethod, 5, 1, 1, 2)
        
        self.Registrationmethod.setObjectName("Registrationmethod")
        self.Registrationmethod.addItem("PhaseCorrelation")
        self.Registrationmethod.addItem("IntensityBased")
        self.Registrationmethod.setFont(font)
        
        
        self.FittingnmethodLbl = QtWidgets.QLabel(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.FittingnmethodLbl, 6, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.FittingnmethodLbl.setFont(font)
        self.FittingnmethodLbl.setObjectName("FittingnmethodLbl")
        
        self.Fittingnmethod = QtWidgets.QComboBox(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.Fittingnmethod, 6, 1, 1, 2)
        
        self.Fittingnmethod.setObjectName("Fittingnmethod")
        self.Fittingnmethod.addItem("TwoStateHMM")
        self.Fittingnmethod.addItem("ThreeStateHMM")
        self.Fittingnmethod.setFont(font)
        
        self.maxspotspercelllbl = QtWidgets.QLabel(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.maxspotspercelllbl, 7, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.maxspotspercelllbl.setFont(font)
        self.maxspotspercelllbl.setObjectName("maxspotspercelllbl")
        
        self.maxspotspercellSpinbox = QtWidgets.QSpinBox(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.maxspotspercellSpinbox, 7, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.maxspotspercellSpinbox.setFont(font)
        self.maxspotspercellSpinbox.setObjectName("maxspotspercellSpinbox")
        self.maxspotspercellSpinbox.setMaximum(500)
        self.maxspotspercellSpinbox.setMinimum(1)
        self.maxspotspercellSpinbox.setValue(2)
        
        self.minburstdurationlbl = QtWidgets.QLabel(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.minburstdurationlbl, 8, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.minburstdurationlbl.setFont(font)
        self.minburstdurationlbl.setObjectName("minburstdurationlbl")
        
        self.minburstdurationSpinbox = QtWidgets.QSpinBox(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.minburstdurationSpinbox, 8, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.minburstdurationSpinbox.setFont(font)
        self.minburstdurationSpinbox.setObjectName("minburstdurationSpinbox")
        self.minburstdurationSpinbox.setMaximum(500)
        self.minburstdurationSpinbox.setMinimum(1)
        self.minburstdurationSpinbox.setValue(1)
        
        self.patchsizelbl = QtWidgets.QLabel(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.patchsizelbl, 9, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.patchsizelbl.setFont(font)
        self.patchsizelbl.setObjectName("patchsizelbl")
        
        self.patchsize = QtWidgets.QComboBox(self.nuc_spot_track)
        self.gridLayout_nuc_spot_track.addWidget(self.patchsize, 9, 1, 1, 1)
        
        self.patchsize.setObjectName("patchsize")
        self.patchsize.addItem("128")
        self.patchsize.addItem("256")
        self.patchsize.addItem("512")
        self.patchsize.setFont(font)
        
        
        
        self.AnalysisMode.addItem(self.nuc_spot_track, "")
        
        
        #############################################################################
        
        
        #### Resutls
        self.Results = QtWidgets.QWidget()
        self.gridLayout_AnalysisMode.addWidget(self.Results, 5, 0, 1, 1)
        self.Results.setObjectName("Results")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.Results)
        self.gridLayout_3.setObjectName("gridLayout_3")
        
        self.NucMaskCheckBox = QtWidgets.QCheckBox(self.Results)
        self.NucMaskCheckBox.setObjectName("NucMaskCheckBox")
        self.gridLayout_3.addWidget(self.NucMaskCheckBox, 0, 0, 1, 1)
        self.NucMaskCheckBox.setFont(font)
        
        self.SpotsDistance = QtWidgets.QCheckBox(self.Results)
        self.SpotsDistance.setObjectName("SpotsDistance")
        self.gridLayout_3.addWidget(self.SpotsDistance, 0, 1, 1, 1)
        self.SpotsDistance.setEnabled(False)
        self.SpotsDistance.setFont(font)
        
        self.NucInfoChkBox = QtWidgets.QCheckBox(self.Results)
        self.NucInfoChkBox.setObjectName("NucInfoChkBox")
        self.gridLayout_3.addWidget(self.NucInfoChkBox, 1, 0, 1, 1)
        self.NucInfoChkBox.setFont(font)
        
        self.SpotsLocation = QtWidgets.QCheckBox(self.Results)
        self.SpotsLocation.setObjectName("SpotsLocation")
        self.gridLayout_3.addWidget(self.SpotsLocation, 1, 1, 1, 1)
        self.SpotsLocation.setFont(font)
        
        self.Cell_Tracking = QtWidgets.QCheckBox(self.Results)
        self.Cell_Tracking.setObjectName("Cell_Tracking")
        self.gridLayout_3.addWidget(self.Cell_Tracking, 2, 0, 1, 1)
        self.Cell_Tracking.setFont(font)
        
        self.Spot_Tracking = QtWidgets.QCheckBox(self.Results)
        self.Spot_Tracking.setObjectName("Spot_Tracking")
        self.gridLayout_3.addWidget(self.Spot_Tracking, 2, 1, 1, 1)
        self.Spot_Tracking.setFont(font)
        self.AnalysisMode.addItem(self.Results, "")
        
        _translate = QtCore.QCoreApplication.translate
        self.AnalysisLbl.setText(_translate("MainWindow", "Analysis"))
        self.RunAnalysis.setText(_translate("MainWindow", "Run Analysis"))
        self.ResetButton.setText(_translate("MainWindow", "Reset"))
        ### nuclei detection
        self.NucleiChLbl.setText(_translate("MainWindow", "Channel"))
        self.NucDetectMethodLbl.setText(_translate("MainWindow", "Method"))
        self.NucleiChannel.setItemText(0, _translate("MainWindow", "Channel 1"))
        self.NucleiChannel.setItemText(1, _translate("MainWindow", "Channel 2"))
        self.NucleiChannel.setItemText(2, _translate("MainWindow", "Channel 3"))
        self.NucleiChannel.setItemText(3, _translate("MainWindow", "Channel 4"))
        self.NucleiChannel.setItemText(4, _translate("MainWindow", "Channel 5"))
        self.NucDetectMethod.setItemText(0, _translate("MainWindow", "Int.-based"))
        self.NucDetectMethod.setItemText(1, _translate("MainWindow", "Marker Controlled"))
        self.NucDetectMethod.setItemText(2, _translate("MainWindow", "CellPose-CPU"))
        self.NucDetectMethod.setItemText(3, _translate("MainWindow", "CellPose-GPU"))
        self.NucDetectMethod.setItemText(4, _translate("MainWindow", "CellPose-Cyto"))
        self.NucDetectMethod.setItemText(5, _translate("MainWindow", "DeepCell"))
        
        self.NucMaxZprojectCheckBox.setText(_translate("MainWindow", "MaxZ Projection"))
        self.NucRemoveBoundaryCheckBox.setText(_translate("MainWindow", "Remove Boundary Nuclei"))
        self.NucFirstThreshLbl.setText(_translate("MainWindow", "Detection"))
        self.NucSecondThreshLbl.setText(_translate("MainWindow", "Separation"))
        self.NucleiareaLbl.setText(_translate("MainWindow", "Area (\u03BCm)\u00b2>"))
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.NucleiDetection), _translate("MainWindow", 
                                                                                                  "Nuclei Detection"))
        
        #### spot detection
        self.SpotCh1CheckBox.setText(_translate("MainWindow", "Ch1"))
        self.SpotCh2CheckBox.setText(_translate("MainWindow", "Ch2"))
        self.SpotCh3CheckBox.setText(_translate("MainWindow", "Ch3"))
        self.SpotCh4CheckBox.setText(_translate("MainWindow", "Ch4"))
        self.SpotCh5CheckBox.setText(_translate("MainWindow", "Ch5"))
        self.ResizeFactorLabel.setText(_translate("MainWindow", "Resize factor:"))
        self.SpotareaminLbl.setText(_translate("MainWindow", "Spot Area Min:"))
        self.SpotareamaxLbl.setText(_translate("MainWindow", "Max:"))
        self.SpotIntegratedIntensityLbl.setText(_translate("MainWindow", "Min Integrated Int.:"))
        self.SpotMaxZProject.setText(_translate("MainWindow", "Max Z-projection"))
        self.RemoveBrightJunk.setText(_translate("MainWindow", "Remove Bright Objects"))
        self.SpotLocationLbl.setText(_translate("MainWindow", "Coordinates:"))
        self.IntegratedIntensity.setText(_translate("MainWindow", "Integrated Int. Mask:"))
        self.PSFsize.setText(_translate("MainWindow", "PSF size(Pix):"))
        
        self.SpotLocationCbox.setItemText(0, _translate("MainWindow", "CenterOfMass"))
        self.SpotLocationCbox.setItemText(1, _translate("MainWindow", "MaxIntensity"))
        self.SpotLocationCbox.setItemText(2, _translate("MainWindow", "Centroid"))
        
        self.IntegratedIntensityCbox.setItemText(0, _translate("MainWindow", "Predefined Gaussian Mask"))
        self.IntegratedIntensityCbox.setItemText(1, _translate("MainWindow", "Fit Gaussian Mask"))
        
        
        ### spot analysis
        self.spotanalysismethodLbl.setText(_translate("MainWindow", "Detection Method"))
        self.thresholdmethodLbl.setText(_translate("MainWindow", "Threshold Method"))
        self.thresholdvalueLbl.setText(_translate("MainWindow", "Threshold Value"))
        self.spotchannelselectlbl.setText(_translate("MainWindow", "Channel"))
        self.sensitivityLbl.setText(_translate("MainWindow", "Kernel Size"))
        self.spotanalysismethod.setItemText(0, _translate("MainWindow", "Laplacian of Gaussian"))  
        self.spotanalysismethod.setItemText(1, _translate("MainWindow", "Gaussian"))
        self.spotanalysismethod.setItemText(2, _translate("MainWindow", "Intensity Threshold"))
        self.spotanalysismethod.setItemText(3, _translate("MainWindow", "Enhanced LOG"))
        self.thresholdmethod.setItemText(0, _translate("MainWindow", "Auto"))
        self.thresholdmethod.setItemText(1, _translate("MainWindow", "Manual"))
        
        self.spotchannelselect.setItemText(0, _translate("MainWindow", "All"))
        self.spotchannelselect.setItemText(1, _translate("MainWindow", "Ch1"))
        self.spotchannelselect.setItemText(2, _translate("MainWindow", "Ch2"))
        self.spotchannelselect.setItemText(3, _translate("MainWindow", "Ch3"))
        self.spotchannelselect.setItemText(4, _translate("MainWindow", "Ch4"))
        self.spotchannelselect.setItemText(5, _translate("MainWindow", "Ch5"))
        
        #### cytoplasm analysis
        self.CytoChLbl.setText(_translate("MainWindow", "Channel"))
        self.CytoCellTypeLbl.setText(_translate("MainWindow", "Cell Type"))
        
        self.CytoDetectMethodLbl.setText(_translate("MainWindow", "Method"))
        self.CytoChannel.setItemText(0, _translate("MainWindow", "Channel 1"))
        self.CytoChannel.setItemText(1, _translate("MainWindow", "Channel 2"))
        self.CytoChannel.setItemText(2, _translate("MainWindow", "Channel 3"))
        self.CytoChannel.setItemText(3, _translate("MainWindow", "Channel 4"))
        self.CytoDetectMethod.setItemText(0, _translate("MainWindow", "CellPose"))
        self.CytoCellType.setItemText(0, _translate("MainWindow", ""))
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.CellBoundary), _translate("MainWindow", "Cell Boundary"))
        
        #### Intensity measurement
        self.SecChLbl.setText(_translate("MainWindow", "Channel"))
        self.SecAreaLbl.setText(_translate("MainWindow", "Area"))
        self.SecChannel.setItemText(0, _translate("MainWindow", ""))
        self.SecChannel.setItemText(1, _translate("MainWindow", "Channel 1"))
        self.SecChannel.setItemText(2, _translate("MainWindow", "Channel 2"))
        self.SecChannel.setItemText(3, _translate("MainWindow", "Channel 3"))
        self.SecChannel.setItemText(4, _translate("MainWindow", "Channel 4"))
        self.SecChannel.setItemText(5, _translate("MainWindow", "Channel 5"))
        self.SecArea.setItemText(0, _translate("MainWindow", ""))
        self.SecArea.setItemText(1, _translate("MainWindow", "Nuclei"))
        self.addsecmeasurement.setText(_translate("MainWindow", "Add Measurement"))
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.secondarymeasurement), _translate("MainWindow", "Intensity Measurement"))
        
        #### Nuclei and Spot Tracking
        self.NucTrackMethodlbl.setText(_translate("MainWindow", "Nuclei Tracking"))
        self.NucSearchRadiuslbl.setText(_translate("MainWindow", "Nuc Search Radius (pix)"))
        self.SpotSearchRadiuslbl.setText(_translate("MainWindow", "Spot Search Radius (pix)"))
        self.Sec_SpotSearchRadiuslbl.setText(_translate("MainWindow", "Secondary Spot Search Radius"))
        self.maxspotspercelllbl.setText(_translate("MainWindow", "Max Spots Per Cell"))
        self.minburstdurationlbl.setText(_translate("MainWindow", "Min Burst Duration (frames)"))
        self.MintrackLengthLbl.setText(_translate("MainWindow", "Min Track Length (Frames)"))
        self.patchsizelbl.setText(_translate("MainWindow", "Patch Size (Pix)"))
        self.patchsize.setItemText(0, _translate("MainWindow", "128"))
        self.patchsize.setItemText(1, _translate("MainWindow", "256"))
        self.patchsize.setItemText(2, _translate("MainWindow", "512"))
        self.NucTrackMethod.setItemText(0, _translate("MainWindow", "Bayesian"))
        self.NucTrackMethod.setItemText(1, _translate("MainWindow", "DeepCell"))
        
        self.RegistrationmethodLbl.setText(_translate("MainWindow", "Track Registration Method"))
        self.Registrationmethod.setItemText(0, _translate("MainWindow", "Phase Correlation"))
        self.Registrationmethod.setItemText(1, _translate("MainWindow", "Intensity-Based"))
        self.FittingnmethodLbl.setText(_translate("MainWindow", "Fitting Method"))
        self.Fittingnmethod.setItemText(0, _translate("MainWindow", "Two State HMM"))
        self.Fittingnmethod.setItemText(1, _translate("MainWindow", "Three State HMM"))
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.nuc_spot_track), _translate("MainWindow", "Nuclei/Spot Tracking"))
        ### results 
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.SpotDetection), _translate("MainWindow", "Spot Channels"))
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.SpotAnalysis), _translate("MainWindow", "Spot Detection Method"))
        self.NucMaskCheckBox.setText(_translate("MainWindow", "Nuclei Mask"))
        self.SpotsDistance.setText(_translate("MainWindow", "Spots Distances"))
        self.NucInfoChkBox.setText(_translate("MainWindow", "Nuclei Info"))
        self.SpotsLocation.setText(_translate("MainWindow", "Spots Location"))
        self.Cell_Tracking.setText(_translate("MainWindow", "Cell Tracking"))
        self.Spot_Tracking.setText(_translate("MainWindow", "Spot Tracking"))
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.Results), _translate("MainWindow", "Results"))
        
    def set_image_analyzer(self, image_analyzer):
        self.image_analyzer = image_analyzer 
    def set_imdisplay(self, ImDisplay):
        self.ImDisplay = ImDisplay 
    
    def FIRST_THRESH_LABEL_UPDATE(self):
        
        self.NucFirstThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucSeparationSlider.value())))
        
    def SECOND_THRESH_LABEL_UPDATE(self):
        
        self.NucSecondThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucDetectionSlider.value())))
        
    def NUCLEI_AREA_LABEL_UPDATE(self):
        
        self.NucleiAreaSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucleiAreaSlider.value()))) 
    
    def SPOT_THRESH_LABEL_UPDATE(self):
        
        self.SpotThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.ThresholdSlider.value())))
    
    def set_gui_params(self, gui_params):
        self.gui_params = gui_params
        
           
    def INITIALIZE_SEGMENTATION_PARAMETERS(self):
        
        if self.NucDetectMethod.currentText() == "Int.-based Processing":
            
            self.NucDetectionSlider.setValue(42)
            self.NucSeparationSlider.setValue(39)
            self.NucleiAreaSlider.setValue(30)
            
            
        if self.NucDetectMethod.currentText() == "Marker Controlled":
            
          
            self.NucDetectionSlider.setValue(98)
            self.NucSeparationSlider.setValue(25)
            self.NucleiAreaSlider.setValue(30)
        
        
        
