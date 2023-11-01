#from controlpanel import Ui_MainWindow
import functools
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QGridLayout
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

IO_GUI_HEIGHT = 260
class display(QWidget):
    
    def __init__(self, ImDisplay=None, analysisgui=None):
        super().__init__()
        
        self.analysisgui = analysisgui
        self.ImDisplay = ImDisplay
        self.channel_colors = { "Ch1": [128, 128, 128], "Ch2": [0, 255, 0], "Ch3": [255, 0, 0], "Ch4": [0, 0, 255],  "Ch5": [255, 165, 0], "Nuclei": [255,0,0],
                                "Ch1_spot": [128, 128, 128], "Ch2_spot": [0, 255, 0], "Ch3_spot": [255, 0, 0], "Ch4_spot": [0, 0, 255],  "Ch5_spot": [255, 165, 0]}

        
        self.lookup_table_rgb = { "Fire": [255, 128, 0], "Gray": [128, 128, 128], "Ice": [128, 128, 255], "Red": [255, 0, 0], "Green": [0, 255, 0],
                      "Blue": [0, 0, 255], "Cyan": [0, 255, 255], "Magenta": [255, 0, 255], "Yellow": [255, 255, 0], "Royal": [65, 105, 225],  
                      "Orange": [255, 165, 0], "Spring": [0, 255, 127], "Violet": [238, 130, 238], "Pink": [255, 192, 203], "HotPink": [255, 105, 180],
                      "Goldenrod": [218, 165, 32], "Rainbow": [127, 127, 127], "Ocean": [0, 127, 255], "Terrain": [139, 69, 19], "Neon": [255, 0, 102]}
        
        self.inverse_lookup = {tuple(v): k for k, v in self.lookup_table_rgb.items()}
        
        self.gridLayout_display = QGridLayout()
        self.setLayout(self.gridLayout_display)
        self.viewer = PhotoViewer(self)
        self.gridLayout_display.addWidget(self.viewer, 1, 1, 15, 15)

        
        self.MaxHistSlider = QtWidgets.QSlider(self)
        self.gridLayout_display.addWidget(self.MaxHistSlider, 16, 9, 1, 5)
        self.MaxHistSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MaxHistSlider.setObjectName("MaxHistSlider")
        self.MaxHistSlider.sliderReleased.connect(lambda:self.ImDisplay.MAX_HIST_SLIDER_UPDATE(self))
        
        self.MinHistSlider = QtWidgets.QSlider(self)
        self.gridLayout_display.addWidget(self.MinHistSlider, 16, 4, 1, 5)
        self.MinHistSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MinHistSlider.setObjectName("MinHistSlider")
        self.MinHistSlider.sliderReleased.connect(lambda:self.ImDisplay.MIN_HIST_SLIDER_UPDATE(self))
        
        self.HistChLabel = QtWidgets.QLabel(self)
        self.gridLayout_display.addWidget(self.HistChLabel, 16, 1, 1, 2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.HistChLabel.setFont(font)
        self.HistChLabel.setObjectName("HistChLabel")
        self.HistChannel = QtWidgets.QComboBox(self)
        self.gridLayout_display.addWidget(self.HistChannel, 16, 3, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.HistChannel.setFont(font)
        self.HistChannel.setObjectName("HistChannel")
        self.HistChannel.addItem("Ch 1")
        self.HistChannel.addItem("Ch 2")
        self.HistChannel.addItem("Ch 3")
        self.HistChannel.addItem("Ch 4")
        self.HistChannel.addItem("Ch 5")
       
        self.Ch1CheckBox = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.Ch1CheckBox, 1, 16, 1, 1)
        self.Ch1CheckBox.setObjectName("Ch1CheckBox")
        self.Ch1CheckBox.setStyleSheet("color: gray")
        self.Ch1CheckBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.Ch1CheckBox.customContextMenuRequested.connect(lambda: self.createContextMenu("Ch1").exec_(QtGui.QCursor.pos()))
        self.Ch1CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.Ch1maxproject = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.Ch1maxproject, 1, 17, 1, 1)
        self.Ch1maxproject.setObjectName("Ch1maxproject")
        self.Ch1maxproject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.Ch2CheckBox = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.Ch2CheckBox, 2, 16, 1, 1)
        self.Ch2CheckBox.setObjectName("Ch2CheckBox")
        self.Ch2CheckBox.setStyleSheet("color: green")
        self.Ch2CheckBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.Ch2CheckBox.customContextMenuRequested.connect(lambda: self.createContextMenu("Ch2").exec_(QtGui.QCursor.pos()))
        self.Ch2CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.Ch2maxproject = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.Ch2maxproject, 2, 17, 1, 1)
        self.Ch2maxproject.setObjectName("Ch2maxproject")
        self.Ch2maxproject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.Ch3CheckBox = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.Ch3CheckBox, 3, 16, 1, 1)
        self.Ch3CheckBox.setObjectName("Ch3CheckBox")
        self.Ch3CheckBox.setStyleSheet("color: red")
        self.Ch3CheckBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.Ch3CheckBox.customContextMenuRequested.connect(lambda: self.createContextMenu("Ch3").exec_(QtGui.QCursor.pos()))
        self.Ch3CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.Ch3maxproject = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.Ch3maxproject, 3, 17, 1, 1)
        self.Ch3maxproject.setObjectName("Ch3maxproject")
        self.Ch3maxproject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.Ch4CheckBox = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.Ch4CheckBox, 4, 16, 1, 1)
        self.Ch4CheckBox.setObjectName("Ch4CheckBox")
        self.Ch4CheckBox.setStyleSheet("color: blue")
        self.Ch4CheckBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.Ch4CheckBox.customContextMenuRequested.connect(lambda: self.createContextMenu("Ch4").exec_(QtGui.QCursor.pos()))
        self.Ch4CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.Ch4maxproject = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.Ch4maxproject, 4, 17, 1, 1)
        self.Ch4maxproject.setObjectName("Ch4maxproject")
        self.Ch4maxproject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.Ch5CheckBox = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.Ch5CheckBox, 5, 16, 1, 1)
        self.Ch5CheckBox.setObjectName("Ch5CheckBox")
        self.Ch5CheckBox.setStyleSheet("color: orange")
        self.Ch5CheckBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.Ch5CheckBox.customContextMenuRequested.connect(lambda: self.createContextMenu("Ch5").exec_(QtGui.QCursor.pos()))
        self.Ch5CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.Ch5maxproject = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.Ch5maxproject, 5, 17, 1, 1)
        self.Ch5maxproject.setObjectName("Ch5maxproject")        
        self.Ch5maxproject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.NuclMaskCheckBox = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.NuclMaskCheckBox, 7, 16, 1, 2)
        self.NuclMaskCheckBox.setObjectName("NucMaskCheckBox")
        self.NuclMaskCheckBox.setStyleSheet("color: red")
        self.NuclMaskCheckBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.NuclMaskCheckBox.customContextMenuRequested.connect(lambda: self.createContextMenu("Nuclei").exec_(QtGui.QCursor.pos()))
        self.NuclMaskCheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.NucPreviewMethod = QtWidgets.QComboBox(self)
        self.gridLayout_display.addWidget(self.NucPreviewMethod, 8, 16, 1, 2)
        self.NucPreviewMethod.setObjectName("NucPreviewMethod")
        self.NucPreviewMethod.addItem("Boundary")
        self.NucPreviewMethod.addItem("Area")
        self.NucPreviewMethod.addItem("Nuc.Index")
        self.NucPreviewMethod.currentIndexChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))

        self.SpotsCheckBox = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.SpotsCheckBox, 10, 16, 1, 2)
        self.SpotsCheckBox.setObjectName("SpotDetection")
        self.SpotsCheckBox.setStyleSheet("color: green")
        self.SpotsCheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
        
        self.spotPreviewMethod = QtWidgets.QComboBox(self)
        self.gridLayout_display.addWidget(self.spotPreviewMethod, 11, 16, 1, 2)
        self.spotPreviewMethod.setObjectName("spotPreviewMethod")
        self.spotPreviewMethod.addItem("Circle")
        self.spotPreviewMethod.addItem("Boundary")
        
        
        self.CytoPreviewCheck = QtWidgets.QCheckBox(self)
        self.gridLayout_display.addWidget(self.CytoPreviewCheck, 12, 16, 1, 2)
        self.CytoPreviewCheck.setObjectName("CytoPreviewCheck")
        self.CytoPreviewCheck.setStyleSheet("color: blue")
        
        self.CytoDisplayMethod = QtWidgets.QComboBox(self)
        self.gridLayout_display.addWidget(self.CytoDisplayMethod, 13, 16, 1, 2)
        self.CytoDisplayMethod.setObjectName("CytoDisplayMethod")
        self.CytoDisplayMethod.addItem("Boundary")
        self.CytoDisplayMethod.addItem("Area")
        
        
        _translate = QtCore.QCoreApplication.translate
        self.Ch1CheckBox.setText(_translate("MainWindow", "Ch1"))
        self.Ch1maxproject.setText(_translate("MainWindow", "Max.Z"))
        self.Ch2CheckBox.setText(_translate("MainWindow", "Ch2"))
        self.Ch2maxproject.setText(_translate("MainWindow", "Max.Z"))
        self.Ch3CheckBox.setText(_translate("MainWindow", "Ch3"))
        self.Ch3maxproject.setText(_translate("MainWindow", "Max.Z"))
        self.Ch4CheckBox.setText(_translate("MainWindow", "Ch4"))
        self.Ch4maxproject.setText(_translate("MainWindow", "Max.Z"))
        self.Ch5CheckBox.setText(_translate("MainWindow", "Ch5"))
        self.Ch5maxproject.setText(_translate("MainWindow", "Max.Z"))
        self.HistChLabel.setText(_translate("MainWindow", "Adjust Intensity"))
        self.HistChannel.setItemText(0, _translate("MainWindow", "Ch 1"))
        self.HistChannel.setItemText(1, _translate("MainWindow", "Ch 2"))
        self.HistChannel.setItemText(2, _translate("MainWindow", "Ch 3"))
        self.HistChannel.setItemText(3, _translate("MainWindow", "Ch 4"))
        self.HistChannel.setItemText(4, _translate("MainWindow", "Ch 5"))
        self.NuclMaskCheckBox.setText(_translate("MainWindow", "Nuclei"))
        self.NucPreviewMethod.setItemText(0, _translate("MainWindow", "Boundary"))
        self.NucPreviewMethod.setItemText(1, _translate("MainWindow", "Area"))
        self.NucPreviewMethod.setItemText(2, _translate("MainWindow", "Nuc.Index"))
        self.spotPreviewMethod.setItemText(0, _translate("MainWindow", "Circle"))
        self.spotPreviewMethod.setItemText(1, _translate("MainWindow", "Boundary"))
        
        self.SpotsCheckBox.setText(_translate("MainWindow", "Spots"))
        
        self.CytoPreviewCheck.setText(_translate("MainWindow", "Cell"))
        self.CytoDisplayMethod.setItemText(0, _translate("MainWindow", "Boundary"))
        self.CytoDisplayMethod.setItemText(1, _translate("MainWindow", "Area"))
        
    def set_analysisgui(self, analysisgui):
        self.analysisgui = analysisgui
    def set_imdisplay(self, ImDisplay):
        self.ImDisplay = ImDisplay
    def createContextMenu(self, channel, Nuclei=False):
        contextMenu = QtWidgets.QMenu(self)

        # Select Color Menu
        selectColorMenu = QtWidgets.QMenu("Select Color", contextMenu)
        selectColorMenu.setTearOffEnabled(True)  # Enable checkmarks

        for color_name in self.lookup_table_rgb.keys():
            # colorAction = QtWidgets.QAction(color_name, self)
            self.colorAction = QtWidgets.QAction(color_name, self)
            self.colorAction.setCheckable(True)  # Make the action checkable
            
            if self.channel_colors[channel] == self.lookup_table_rgb[color_name]:
                self.colorAction.setChecked(True)

            # Use functools.partial to set the channel and color arguments
            self.colorAction.triggered.connect(functools.partial(self.setColor, channel, self.lookup_table_rgb[color_name]))
            self.colorAction.triggered.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self))
            selectColorMenu.addAction(self.colorAction)

        contextMenu.addMenu(selectColorMenu)

        return contextMenu
    
    def setColor(self, channel, color):
        # Get the representative color from the lookup_table_rgb dictionary
        # rgb_values = self.lookup_table_rgb[color]
        rgb_values = color
        rgb_str = f"rgb({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]})"

        if channel == "Ch1":
            self.Ch1CheckBox.setStyleSheet(f"color: {rgb_str}")
        elif channel == "Ch2":
            self.Ch2CheckBox.setStyleSheet(f"color: {rgb_str}")
        elif channel == "Ch3":
            self.Ch3CheckBox.setStyleSheet(f"color: {rgb_str}")
        elif channel == "Ch4":
            self.Ch4CheckBox.setStyleSheet(f"color: {rgb_str}")
        elif channel == "Ch5":
            self.Ch5CheckBox.setStyleSheet(f"color: {rgb_str}")
        elif channel == "Nuclei":
            self.NuclMaskCheckBox.setStyleSheet(f"color: {rgb_str}")
        elif channel == "Ch1_spot":
            self.analysisgui.SpotCh1CheckBox.setStyleSheet(f"color: {rgb_str}")
        elif channel == "Ch2_spot":
            self.analysisgui.SpotCh2CheckBox.setStyleSheet(f"color: {rgb_str}")
        elif channel == "Ch3_spot":
            self.analysisgui.SpotCh3CheckBox.setStyleSheet(f"color: {rgb_str}")
        elif channel == "Ch4_spot":
            self.analysisgui.SpotCh4CheckBox.setStyleSheet(f"color: {rgb_str}")
        elif channel == "Ch5_spot":
            self.analysisgui.SpotCh5CheckBox.setStyleSheet(f"color: {rgb_str}")
            
        self.channel_colors[channel] = color
        
    # Placeholder function for max projection
    def maxProjection(self):
        print("Max Projection selected")

    # Placeholder function for mean projection
    def meanProjection(self):
        print("Mean Projection selected")




class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)
    first_photo=True
    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
#         self.setGeometry(QtCore.QRect(0, 0, 440, 420))
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0
               

    def setPhoto(self, pixmap=None):
#         self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        if self.first_photo:
            self.fitInView()
            self.first_photo=False
    

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)

    def contextMenuEvent(self, event):
        # Check if the click was made on the photo
        if self._photo.isUnderMouse():
            contextMenu = QtWidgets.QMenu(self)
            
            # Create Save Image action
            saveAction = QtWidgets.QAction("Save Image", self)
            saveAction.triggered.connect(self.saveImage)
            contextMenu.addAction(saveAction)
            
            contextMenu.exec_(event.globalPos())

    def saveImage(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        filePath, fileFilter = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", "", "TIFF (*.tiff);;TIFF (*.tif);;JPEG (*.jpeg);;PNG (*.png)", options=options)

        # Exit the function if no path is selected
        if not filePath:
            return

        # Determine the format based on the filter or use a default if nothing matches
        if "TIFF (*.tiff)" in fileFilter:
            format = "TIFF"
            if not filePath.lower().endswith(".tiff"):
                filePath += ".tiff"
        elif "TIFF (*.tif)" in fileFilter:
            format = "TIFF"
            if not filePath.lower().endswith(".tif"):
                filePath += ".tif"
        elif "JPEG (*.jpeg)" in fileFilter:
            format = "JPEG"
            if not filePath.lower().endswith(".jpeg"):
                filePath += ".jpeg"
        elif "PNG (*.png)" in fileFilter:
            format = "PNG"
            if not filePath.lower().endswith(".png"):
                filePath += ".png"
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Unsupported file format!")
            return

        # Save the image
        success = self._photo.pixmap().save(filePath, format)

        if not success:
            QtWidgets.QMessageBox.warning(self, "Error", "Image couldn't be saved!")
