#from controlpanel import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
IO_GUI_HEIGHT = 180
class display(QWidget):
    
    def __init__(self, centralwidget):
        super().__init__(centralwidget)
        
        self.DisplayToolbox = QtWidgets.QLabel(centralwidget)
        self.DisplayToolbox.setGeometry(QtCore.QRect(10, IO_GUI_HEIGHT, 200, 30))
        font = QtGui.QFont()
        font.setFamily(".Farah PUA")
        font.setPointSize(24)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        font.setKerning(False)
        self.DisplayToolbox.setFont(font)
        self.DisplayToolbox.setObjectName("AnalysisLbl")

        self.ColSpinBox = QtWidgets.QSpinBox(self)
        self.ColSpinBox.setGeometry(QtCore.QRect(450, 530+ IO_GUI_HEIGHT, 48, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.ColSpinBox.setFont(font)
        self.ColSpinBox.setObjectName("ColSpinBox")
        
        self.viewer = PhotoViewer(self)
        
        self.ApplyMaxMin = QtWidgets.QPushButton(self)
        self.ApplyMaxMin.setGeometry(QtCore.QRect(440, 130 + IO_GUI_HEIGHT, 71, 32))
        self.ApplyMaxMin.setObjectName("ApplyMaxMin")
        self.HistogramsView = QtWidgets.QGraphicsView(self)
        self.HistogramsView.setGeometry(QtCore.QRect(10, 30 + IO_GUI_HEIGHT, 231, 101))
        self.HistogramsView.setObjectName("HistogramsView")
        self.ResetHistogram = QtWidgets.QPushButton(self)
        self.ResetHistogram.setGeometry(QtCore.QRect(370, 130 + IO_GUI_HEIGHT, 71, 32))
        self.ResetHistogram.setObjectName("ResetHistogram")
        self.MaxHistSpinBox = QtWidgets.QSpinBox(self)
        self.MaxHistSpinBox.setGeometry(QtCore.QRect(470, 100 + IO_GUI_HEIGHT, 48, 24))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.MaxHistSpinBox.setFont(font)
        self.MaxHistSpinBox.setObjectName("MaxHistSpinBox")
        self.MaxHistSlider = QtWidgets.QSlider(self)
        self.MaxHistSlider.setGeometry(QtCore.QRect(280, 100 + IO_GUI_HEIGHT, 191, 22))
        self.MaxHistSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MaxHistSlider.setObjectName("MaxHistSlider")
        self.MaxHistLbl = QtWidgets.QLabel(self)
        self.MaxHistLbl.setGeometry(QtCore.QRect(250, 100 + IO_GUI_HEIGHT, 31, 21))
        self.MaxHistLbl.setObjectName("MaxHistLbl")
        self.MinHistSpinBox = QtWidgets.QSpinBox(self)
        self.MinHistSpinBox.setGeometry(QtCore.QRect(470, 70 + IO_GUI_HEIGHT, 48, 24))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.MinHistSpinBox.setFont(font)
        self.MinHistSpinBox.setObjectName("MinHistSpinBox")
        self.MinHistLbl = QtWidgets.QLabel(self)
        self.MinHistLbl.setGeometry(QtCore.QRect(250, 70 + IO_GUI_HEIGHT, 31, 21))
        self.MinHistLbl.setObjectName("MinHistLbl")
        self.MinHistSlider = QtWidgets.QSlider(self)
        self.MinHistSlider.setGeometry(QtCore.QRect(280, 70 + IO_GUI_HEIGHT, 191, 22))
        self.MinHistSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MinHistSlider.setObjectName("MinHistSlider")
        self.ColScroller = QtWidgets.QScrollBar(self)
        self.ColScroller.setGeometry(QtCore.QRect(30, 530 + IO_GUI_HEIGHT, 411, 16))
        self.ColScroller.setOrientation(QtCore.Qt.Horizontal)
        self.ColScroller.setObjectName("ColScroller")
        self.ZScroller = QtWidgets.QScrollBar(self)
        self.ZScroller.setGeometry(QtCore.QRect(30, 560 + IO_GUI_HEIGHT, 411, 16))
        self.ZScroller.setOrientation(QtCore.Qt.Horizontal)
        self.ZScroller.setObjectName("ZScroller")
        self.ZSpinBox = QtWidgets.QSpinBox(self)
        self.ZSpinBox.setGeometry(QtCore.QRect(450, 560 + IO_GUI_HEIGHT, 48, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.ZSpinBox.setFont(font)
        self.ZSpinBox.setObjectName("ZSpinBox")
        self.TScroller = QtWidgets.QScrollBar(self)
        self.TScroller.setGeometry(QtCore.QRect(30, 590 + IO_GUI_HEIGHT, 411, 16))
        self.TScroller.setOrientation(QtCore.Qt.Horizontal)
        self.TScroller.setObjectName("TScroller")
        self.TSpinBox = QtWidgets.QSpinBox(self)
        self.TSpinBox.setGeometry(QtCore.QRect(450, 590 + IO_GUI_HEIGHT, 48, 20))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.TSpinBox.setFont(font)
        self.TSpinBox.setObjectName("TSpinBox")
        self.ColLabel = QtWidgets.QLabel(self)
        self.ColLabel.setGeometry(QtCore.QRect(5, 530 + IO_GUI_HEIGHT, 21, 20))
        self.ColLabel.setObjectName("ColLabel")
        self.ZLabel = QtWidgets.QLabel(self)
        self.ZLabel.setGeometry(QtCore.QRect(10, 560 + IO_GUI_HEIGHT, 16, 16))
        self.ZLabel.setObjectName("ZLabel")
        self.TLabel = QtWidgets.QLabel(self)
        self.TLabel.setGeometry(QtCore.QRect(10, 590 + IO_GUI_HEIGHT, 16, 16))
        self.TLabel.setObjectName("TLabel")
        self.RowScroller = QtWidgets.QScrollBar(self)
        self.RowScroller.setGeometry(QtCore.QRect(10, 170 + IO_GUI_HEIGHT, 16, 351))
        self.RowScroller.setOrientation(QtCore.Qt.Vertical)
        self.RowScroller.setObjectName("RowScroller")
        self.RowSpinBox = QtWidgets.QSpinBox(self)
        self.RowSpinBox.setGeometry(QtCore.QRect(0, 146 + IO_GUI_HEIGHT, 41, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.RowSpinBox.setFont(font)
        self.RowSpinBox.setObjectName("RowSpinBox")
        self.RowLabel = QtWidgets.QLabel(self)
        self.RowLabel.setGeometry(QtCore.QRect(0, 130 + IO_GUI_HEIGHT, 31, 21))
        self.RowLabel.setObjectName("RowLabel")
        self.HistAutoAdjust = QtWidgets.QPushButton(self)
        self.HistAutoAdjust.setGeometry(QtCore.QRect(300, 130 + IO_GUI_HEIGHT, 71, 32))
        self.HistAutoAdjust.setObjectName("HistAutoAdjust")
        self.Ch1CheckBox = QtWidgets.QCheckBox(self)
        self.Ch1CheckBox.setGeometry(QtCore.QRect(450, 170 + IO_GUI_HEIGHT, 51, 20))
        self.Ch1CheckBox.setObjectName("Ch1CheckBox")
        self.Ch1CheckBox.setStyleSheet("color: gray")
        self.Ch2CheckBox = QtWidgets.QCheckBox(self)
        self.Ch2CheckBox.setGeometry(QtCore.QRect(450, 190 + IO_GUI_HEIGHT, 51, 20))
        self.Ch2CheckBox.setObjectName("Ch2CheckBox")
        self.Ch2CheckBox.setStyleSheet("color: red")
        self.Ch3CheckBox = QtWidgets.QCheckBox(self)
        self.Ch3CheckBox.setGeometry(QtCore.QRect(450, 210 + IO_GUI_HEIGHT, 51, 20))
        self.Ch3CheckBox.setObjectName("Ch3CheckBox")
        self.Ch3CheckBox.setStyleSheet("color: green")
        self.Ch4CheckBox = QtWidgets.QCheckBox(self)
        self.Ch4CheckBox.setGeometry(QtCore.QRect(450, 230 + IO_GUI_HEIGHT, 51, 20))
        self.Ch4CheckBox.setObjectName("Ch4CheckBox")
        self.Ch4CheckBox.setStyleSheet("color: blue")
        self.Ch5CheckBox = QtWidgets.QCheckBox(self)
        self.Ch5CheckBox.setGeometry(QtCore.QRect(450, 250 + IO_GUI_HEIGHT, 51, 20))
        self.Ch5CheckBox.setObjectName("Ch5CheckBox")
        self.Ch5CheckBox.setStyleSheet("color: orange")
        
        self.HistChLabel = QtWidgets.QLabel(self)
        self.HistChLabel.setGeometry(QtCore.QRect(250, 30 + IO_GUI_HEIGHT, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.HistChLabel.setFont(font)
        self.HistChLabel.setObjectName("HistChLabel")
        self.HistChannel = QtWidgets.QComboBox(self)
        self.HistChannel.setGeometry(QtCore.QRect(310, 30 + IO_GUI_HEIGHT, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.HistChannel.setFont(font)
        self.HistChannel.setObjectName("HistChannel")
        self.HistChannel.addItem("Ch 1")
        self.HistChannel.addItem("Ch 2")
        self.HistChannel.addItem("Ch 3")
        self.HistChannel.addItem("Ch 4")
        self.FOVScroller = QtWidgets.QScrollBar(self)
        self.FOVScroller.setGeometry(QtCore.QRect(80, 140 + IO_GUI_HEIGHT, 141, 16))
        self.FOVScroller.setOrientation(QtCore.Qt.Horizontal)
        self.FOVScroller.setObjectName("FOVScroller")
        self.FOVSpinBox = QtWidgets.QSpinBox(self)
        self.FOVSpinBox.setGeometry(QtCore.QRect(230, 140 + IO_GUI_HEIGHT, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.FOVSpinBox.setFont(font)
        self.FOVSpinBox.setObjectName("FOVSpinBox")
        self.FOVLabel = QtWidgets.QLabel(self)
        self.FOVLabel.setGeometry(QtCore.QRect(50, 140 + IO_GUI_HEIGHT, 31, 20))
        self.FOVLabel.setObjectName("FOVLabel")
        
        self.NuclMaskCheckBox = QtWidgets.QCheckBox(self)
        self.NuclMaskCheckBox.setGeometry(QtCore.QRect(5, 615 + IO_GUI_HEIGHT, 70, 20))
        self.NuclMaskCheckBox.setObjectName("NucMaskCheckBox")
        self.NuclMaskCheckBox.setStyleSheet("color: red")
        self.NucPreviewMethod = QtWidgets.QComboBox(self)
        self.NucPreviewMethod.setGeometry(QtCore.QRect(75, 610 + IO_GUI_HEIGHT, 110, 31))
        self.NucPreviewMethod.setObjectName("NucPreviewMethod")
        self.NucPreviewMethod.addItem("Boundary")
        self.NucPreviewMethod.addItem("Area")
        
        self.SpotsCheckBox = QtWidgets.QCheckBox(self)
        self.SpotsCheckBox.setGeometry(QtCore.QRect(205, 615 + IO_GUI_HEIGHT, 60, 20))
        self.SpotsCheckBox.setObjectName("SpotDetection")
        self.SpotsCheckBox.setStyleSheet("color: green")
        
        self.SpotPreviewMethod = QtWidgets.QComboBox(self)
        self.SpotPreviewMethod.setGeometry(QtCore.QRect(265, 610 + IO_GUI_HEIGHT, 80, 31))
        self.SpotPreviewMethod.setObjectName("SpotPreviewMethod")
        self.SpotPreviewMethod.addItem("Dots")
        self.SpotPreviewMethod.addItem("Cross")
        
        self.CytoPreviewCheck = QtWidgets.QCheckBox(self)
        self.CytoPreviewCheck.setGeometry(QtCore.QRect(365, 610 + IO_GUI_HEIGHT, 45, 31))
        self.CytoPreviewCheck.setObjectName("CytoPreviewCheck")
        self.CytoPreviewCheck.setStyleSheet("color: blue")
        
        self.CytoDisplayMethod = QtWidgets.QComboBox(self)
        self.CytoDisplayMethod.setGeometry(QtCore.QRect(410, 610 + IO_GUI_HEIGHT, 110, 31))
        self.CytoDisplayMethod.setObjectName("CytoDisplayMethod")
        self.CytoDisplayMethod.addItem("Boundary")
        self.CytoDisplayMethod.addItem("Area")
        
        
        _translate = QtCore.QCoreApplication.translate
        self.DisplayToolbox.setText(_translate("MainWindow", "Display Toolbox"))
        self.ApplyMaxMin.setText(_translate("MainWindow", "Apply"))
        self.ResetHistogram.setText(_translate("MainWindow", "Reset"))
        self.MaxHistLbl.setText(_translate("MainWindow", "Max"))
        self.MinHistLbl.setText(_translate("MainWindow", "Min"))
        self.ColLabel.setText(_translate("MainWindow", "Col"))
        self.ZLabel.setText(_translate("MainWindow", "Z"))
        self.TLabel.setText(_translate("MainWindow", "t"))
        self.RowLabel.setText(_translate("MainWindow", "Row"))
        self.HistAutoAdjust.setText(_translate("MainWindow", "Auto"))
        self.Ch1CheckBox.setText(_translate("MainWindow", "Ch1"))
        self.Ch2CheckBox.setText(_translate("MainWindow", "Ch2"))
        self.Ch3CheckBox.setText(_translate("MainWindow", "Ch3"))
        self.Ch4CheckBox.setText(_translate("MainWindow", "Ch4"))
        self.Ch5CheckBox.setText(_translate("MainWindow", "Ch5"))
        self.HistChLabel.setText(_translate("MainWindow", "Channel"))
        self.HistChannel.setItemText(0, _translate("MainWindow", "Ch 1"))
        self.HistChannel.setItemText(1, _translate("MainWindow", "Ch 2"))
        self.HistChannel.setItemText(2, _translate("MainWindow", "Ch 3"))
        self.HistChannel.setItemText(3, _translate("MainWindow", "Ch 4"))
        self.FOVLabel.setText(_translate("MainWindow", "FOV"))
        self.NuclMaskCheckBox.setText(_translate("MainWindow", "Nuclei"))
        self.NucPreviewMethod.setItemText(0, _translate("MainWindow", "Boundary"))
        self.NucPreviewMethod.setItemText(1, _translate("MainWindow", "Area"))
        self.SpotsCheckBox.setText(_translate("MainWindow", "Spots"))
        self.SpotPreviewMethod.setItemText(0, _translate("MainWindow", "Dots"))
        self.SpotPreviewMethod.setItemText(1, _translate("MainWindow", "Cross"))
        self.CytoPreviewCheck.setText(_translate("MainWindow", "Cell"))
        self.CytoDisplayMethod.setItemText(0, _translate("MainWindow", "Boundary"))
        self.CytoDisplayMethod.setItemText(1, _translate("MainWindow", "Area"))

class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self.setGeometry(QtCore.QRect(30, 170+ IO_GUI_HEIGHT, 411, 350))
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
            
#     def fitInView(self, scale=True):
#         rect = QtCore.QRectF(self._photo.pixmap().rect())
        
#         self.setSceneRect(rect)
            
#         unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
#         self.scale(1 / unity.width(), 1 / unity.height())
#         viewrect = self.viewport().rect()
#         scenerect = self.transform().mapRect(rect)
#         factor = min(viewrect.width() / scenerect.width(),
#                      viewrect.height() / scenerect.height())
#         self.scale(factor, factor)
#         self._zoom = 0        
    

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()
    

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

