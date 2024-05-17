from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QGroupBox, QHeaderView, QCheckBox, QListWidget
import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import Qt

WELL_PLATE_LENGTH = 24
WELL_PLATE_WIDTH = 16
WELL_PLATE_ROWS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

class gridgenerator(QWidget):
    first_time=True
    def __init__(self, ControlPanel, centralwidget, gridLayout_centralwidget, displaygui, inout_resource_gui, ImDisplay):
        super().__init__(centralwidget)
        self.displaygui = displaygui
        self.inout_resource_gui = inout_resource_gui
        self.ImDisplay = ImDisplay
        self.ControlPanel = ControlPanel
        self.gridLayout_centralwidget = gridLayout_centralwidget
        self.checked_wells = set()
        self.checked_fovs = set()
        self.checked_zs = set()
        self.checked_times = set()
        # self.selected_metadata = pd.DataFrame()
        # GroupBox for the table
        self.groupBox = QGroupBox(centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.groupBox.setSizePolicy(sizePolicy)
        self.gridLayout_centralwidget.addWidget(self.groupBox, 5, 1, 6, 7)
        self.gridLayout_grdigroupbox = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_grdigroupbox.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_grdigroupbox.setSpacing(0)

        # TableWidget setup
        self.tableWidget = QtWidgets.QTableWidget(self.groupBox)
        self.gridLayout_grdigroupbox.addWidget(self.tableWidget, 0, 0)
        self.tableWidget.setBaseSize(QtCore.QSize(3, 3))
        font = QtGui.QFont()
        font.setPointSize(6)
        font.setBold(False)
        font.setWeight(10)
        self.tableWidget.setFont(font)
        self.tableWidget.setColumnCount(WELL_PLATE_LENGTH)
        self.tableWidget.setRowCount(WELL_PLATE_WIDTH)
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        for i in range(WELL_PLATE_WIDTH):
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setVerticalHeaderItem(i, item)

        for i in range(WELL_PLATE_LENGTH):
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setHorizontalHeaderItem(i, item)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tableWidget.updateGeometry()
        self.tableWidget.itemClicked.connect(lambda: self.on_click_table(self.ControlPanel.Meta_Data_df, self.displaygui, self.inout_resource_gui, self.ImDisplay))


        ### Add Checkbox for Well Selection
        self.well_checkbox = QCheckBox("Well Selection", self.groupBox)
        self.gridLayout_grdigroupbox.addWidget(self.well_checkbox, 1, 0)
        self.well_checkbox.stateChanged.connect(self.disable_well_signals)

        ### FOV list

        self.groupBox1 = QGroupBox(centralwidget)
        self.gridLayout_centralwidget.addWidget(self.groupBox1, 5, 8, 6, 3)
        self.gridLayout_grdigroupbox1 = QtWidgets.QGridLayout(self.groupBox1)

        self.FOVlist = QtWidgets.QListWidget(self.groupBox1)
        self.gridLayout_grdigroupbox1.addWidget(self.FOVlist, 0, 0)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.FOVlist.setFont(font)
        self.FOVlist.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.FOVlist.setBatchSize(80)
        self.FOVlist.itemClicked.connect(lambda: self.on_click_list(self.ControlPanel.Meta_Data_df, self.ImDisplay, self.displaygui))

        ### Add Checkbox for Field Selection
        self.field_checkbox = QCheckBox("Field Selection", self.groupBox1)
        self.gridLayout_grdigroupbox1.addWidget(self.field_checkbox, 1, 0)
        self.field_checkbox.stateChanged.connect(self.disable_field_signals)

        ### Z list 
        self.Zlist = QtWidgets.QListWidget(self.groupBox1)
        self.gridLayout_grdigroupbox1.addWidget(self.Zlist, 0, 1)
        self.Zlist.setFont(font)
        self.Zlist.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.Zlist.setBatchSize(80)
        self.Zlist.itemClicked.connect(lambda: self.on_click_list(self.ControlPanel.Meta_Data_df, self.ImDisplay, self.displaygui))

        ### Add Checkbox for Z Selection
        self.z_checkbox = QCheckBox("Z Selection", self.groupBox1)
        self.gridLayout_grdigroupbox1.addWidget(self.z_checkbox, 1, 1)
        self.z_checkbox.stateChanged.connect(self.disable_z_signals)

        ### Time list 
        self.Timelist = QtWidgets.QListWidget(self.groupBox1)
        self.gridLayout_grdigroupbox1.addWidget(self.Timelist, 0, 2)
        self.Timelist.setFont(font)
        self.Timelist.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.Timelist.setBatchSize(80)
        self.Timelist.itemClicked.connect(lambda: self.on_click_list(self.ControlPanel.Meta_Data_df, self.ImDisplay, self.displaygui))

        ### Add Checkbox for Time Selection
        self.time_checkbox = QCheckBox("Time Selection", self.groupBox1)
        self.gridLayout_grdigroupbox1.addWidget(self.time_checkbox, 1, 2)
        self.time_checkbox.stateChanged.connect(self.disable_time_signals)

        _translate = QtCore.QCoreApplication.translate
        for i in range(len(WELL_PLATE_ROWS)):
            item = self.tableWidget.verticalHeaderItem(i)
            item.setText(_translate("MainWindow", WELL_PLATE_ROWS[i]))

        for i in range(WELL_PLATE_LENGTH):
            item = self.tableWidget.horizontalHeaderItem(i)
            item.setText(_translate("MainWindow", str(i+1)))

    def disable_well_signals(self, state):
        if state == Qt.Checked:
            self.tableWidget.itemClicked.disconnect()
            self.tableWidget.itemClicked.connect(lambda: self.on_click_table_save_wells(self.ControlPanel.Meta_Data_df))
        else:
            self.tableWidget.itemClicked.disconnect()
            self.tableWidget.itemClicked.connect(lambda: self.on_click_table(self.ControlPanel.Meta_Data_df, self.displaygui, self.inout_resource_gui, self.ImDisplay))

    def disable_field_signals(self, state):
        if state == Qt.Checked:
            self.FOVlist.itemClicked.disconnect()
            self.FOVlist.itemClicked.connect(self.on_click_FOV_list)
        else:
            self.FOVlist.itemClicked.disconnect()
            self.FOVlist.itemClicked.connect(lambda: self.on_click_list(self.ControlPanel.Meta_Data_df, self.ImDisplay, self.displaygui))

    def disable_z_signals(self, state):
        if state == Qt.Checked:
            self.Zlist.itemClicked.disconnect()
            self.Zlist.itemClicked.connect(self.on_click_Z_list)
        else:
            self.Zlist.itemClicked.disconnect()
            self.Zlist.itemClicked.connect(lambda: self.on_click_list(self.ControlPanel.Meta_Data_df, self.ImDisplay, self.displaygui))

    def disable_time_signals(self, state):
        if state == Qt.Checked:
            self.Timelist.itemClicked.disconnect()
            self.Timelist.itemClicked.connect(self.on_click_Time_list)
        else:
            self.Timelist.itemClicked.disconnect()
            self.Timelist.itemClicked.connect(lambda: self.on_click_list(self.ControlPanel.Meta_Data_df, self.ImDisplay, self.displaygui))

    def setup_connections(self):
        # Assuming checkboxes are instantiated and accessible
        self.field_checkbox.stateChanged.connect(self.disable_field_signals)
        self.z_checkbox.stateChanged.connect(self.disable_z_signals)
        self.time_checkbox.stateChanged.connect(self.disable_time_signals)
        # Initially setting up the connections based on checkbox states
        self.disable_field_signals(self.field_checkbox.isChecked())
        self.disable_z_signals(self.z_checkbox.isChecked())
        self.disable_time_signals(self.time_checkbox.isChecked())
        
    def on_click_table_save_wells(self, out_df):
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            img_row = currentQTableWidgetItem.row()
            img_col = currentQTableWidgetItem.column()

            # Check if there's a corresponding record in the dataframe
            df_checker = out_df.loc[(out_df['column'] == str(img_col + 1)) & (out_df['row'] == str(img_row + 1))]

            if not df_checker.empty:
                # Retrieve current item
                item = self.tableWidget.item(img_row, img_col)
                if not item:  # If there's no item, create one
                    item = QtWidgets.QTableWidgetItem()
                    self.tableWidget.setItem(img_row, img_col, item)               
                
                # Define a larger font for the tick mark
                font = item.font()
                font.setPointSize(10)  # Set the size as needed
                font.setBold(True)
                item.setFont(font)
                
                # Check current text and toggle
                current_text = item.text()
                if current_text == "✓":
                    item.setText("")  # If tick mark is present, remove it
                    self.checked_wells.discard((img_row, img_col))  # Remove from the set
                else:
                    item.setText("✓")  # If no tick mark, add it
                    self.checked_wells.add((img_row, img_col))  # Add to the set

                # print(f"Checked wells updated: {self.checked_wells}")
                
    def on_click_FOV_list(self, item):
        idx = self.FOVlist.row(item)
        current_text = item.text().strip()
        if "✓" in current_text:
            new_text = current_text.replace(" ✓", "")  # Remove checkmark
            self.checked_fovs.discard(idx)
        else:
            new_text = current_text + " ✓"  # Add checkmark
            self.checked_fovs.add(idx)
        item.setText(new_text)
        # print(f"Checked FOVs updated: {self.checked_fovs}")

    def on_click_Z_list(self, item):
        idx = self.Zlist.row(item)
        current_text = item.text().strip()
        if "✓" in current_text:
            new_text = current_text.replace(" ✓", "")  # Remove checkmark
            self.checked_zs.discard(idx)
        else:
            new_text = current_text + " ✓"  # Add checkmark
            self.checked_zs.add(idx)
        item.setText(new_text)
        # print(f"Checked Zs updated: {self.checked_zs}")

    def on_click_Time_list(self, item):
        idx = self.Timelist.row(item)
        current_text = item.text().strip()
        if "✓" in current_text:
            new_text = current_text.replace(" ✓", "")  # Remove checkmark
            self.checked_times.discard(idx)
        else:
            new_text = current_text + " ✓"  # Add checkmark
            self.checked_times.add(idx)
        item.setText(new_text)
        # print(f"Checked Times updated: {self.checked_times}")
        
    def GRID_INITIALIZER(self, out_df, displaygui, inout_resource_gui, ImDisplay):
        
        ### initailize well plate grid
        cols_in_use = np.unique(np.asarray(out_df['column'], dtype=int))
        rows_in_use = np.unique(np.asarray(out_df['row'], dtype=int))
        k=0
        for c in cols_in_use:
            for r in rows_in_use:
                
                df_checker = out_df.loc[(out_df['column'] == str(c)) & (out_df['row'] == str(r))]
                
                if df_checker.empty == False:
                    self.tableWidget.setItem(r-1, c-1, QtWidgets.QTableWidgetItem())
                    self.tableWidget.item(r-1, c-1).setBackground(QtGui.QColor(10,200,10))
                    
                    if k==0:

                        current_row = r-1
                        current_col = c-1
                        k=k+1
        
        # Set the initial color of the first selected well based on checkbox state
        
            
        self.tableWidget.setCurrentCell(current_row, current_col)
        self.on_click_table( out_df, displaygui, inout_resource_gui, ImDisplay)
        
    @pyqtSlot()
    def on_click_table(self, out_df, displaygui, inout_resource_gui, ImDisplay):    
        
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            img_row = currentQTableWidgetItem.row() + 1
            img_col = currentQTableWidgetItem.column() + 1
       
        df_checker = out_df.loc[(out_df['column'] == str(img_col)) & (out_df['row'] == str(img_row))]
        self.Timelist.clear()
        self.FOVlist.clear()
        self.Zlist.clear()
        
        # if self.well_checkbox.isChecked():
        #     self.tableWidget.item(currentQTableWidgetItem.row(), currentQTableWidgetItem.column()).setBackground(QtGui.QColor(255, 255, 0))
        # else:
        #     self.tableWidget.item(currentQTableWidgetItem.row(), currentQTableWidgetItem.column()).setBackground(QtGui.QColor(0, 0, 255))
        #### initalizae FOV and Z list
        
        df_checker = out_df.loc[(out_df['column'] == str(img_col)) & (out_df['row'] == str(img_row))]
        z_values = np.unique(np.asarray(df_checker['z_slice'], dtype=int))
        for i in range(z_values.__len__()):
           
            item = QtWidgets.QListWidgetItem()
            self.Zlist.addItem(item)
        
        _translate = QtCore.QCoreApplication.translate
        __sortingEnabled = self.Zlist.isSortingEnabled()
        self.Zlist.setSortingEnabled(False)
        for i in range(z_values.__len__()):
            item = self.Zlist.item(i)
            item.setText(_translate("MainWindow", str(z_values[i])))
            
            if i==0:
                self.Zlist.setCurrentItem(item)
              
        self.Zlist.setSortingEnabled(__sortingEnabled)

        ### Initialize FOV List
        df_checker = out_df.loc[(out_df['column'] == str(img_col)) & (out_df['row'] == str(img_row))]
        fov_values = np.unique(np.asarray(df_checker['field_index'], dtype=int))
            
        for i in range(fov_values.__len__()):
            
            item = QtWidgets.QListWidgetItem()
            self.FOVlist.addItem(item)

        _translate = QtCore.QCoreApplication.translate
        __sortingEnabled = self.FOVlist.isSortingEnabled()
        self.FOVlist.setSortingEnabled(False)
        for i in range(fov_values.__len__()):
            item = self.FOVlist.item(i)
            item.setText(_translate("MainWindow", str(fov_values[i])))
            
            if i==0:
                self.FOVlist.setCurrentItem(item)
        
        self.FOVlist.setSortingEnabled(__sortingEnabled)
        ### Initialize Time List
        df_checker = out_df.loc[(out_df['column'] == str(img_col)) & (out_df['row'] == str(img_row))]
        time_values = np.unique(np.asarray(df_checker['time_point'], dtype=int))
            
        for i in range(time_values.__len__()):
            
            item = QtWidgets.QListWidgetItem()
            self.Timelist.addItem(item)

        _translate = QtCore.QCoreApplication.translate
        __sortingEnabled = self.Timelist.isSortingEnabled()
        self.Timelist.setSortingEnabled(False)
        for i in range(time_values.__len__()):
            item = self.Timelist.item(i)
            item.setText(_translate("MainWindow", str(time_values[i])))
            
            if i==0:
                self.Timelist.setCurrentItem(item)
        
        self.Timelist.setSortingEnabled(__sortingEnabled)
        
        
        self.on_click_list(df_checker, ImDisplay, displaygui)
        
    def on_click_list(self,df_checker, ImDisplay, displaygui):
        
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            img_row = currentQTableWidgetItem.row() + 1
            img_col = currentQTableWidgetItem.column() + 1
            
            
        ImDisplay.grid_data[0] = img_col
        ImDisplay.grid_data[1] = img_row
        ImDisplay.grid_data[2] = np.array(''.join(filter(str.isdigit, self.Timelist.currentItem().text()))).astype(int)
        ImDisplay.grid_data[3] = np.array(''.join(filter(str.isdigit, self.FOVlist.currentItem().text()))).astype(int)
        ImDisplay.grid_data[4] = np.array(''.join(filter(str.isdigit, self.Zlist.currentItem().text()))).astype(int)
        ImDisplay.GET_IMAGE_NAME(displaygui)
        
        if self.first_time==True:
            first_ch = df_checker['channel'].unique()[0]
            if first_ch=='1':
                displaygui.Ch1CheckBox.setChecked(True) 
            if first_ch=='2':
                displaygui.Ch2CheckBox.setChecked(True)
            if first_ch=='3':
                displaygui.Ch3CheckBox.setChecked(True)
            if first_ch=='4':
                displaygui.Ch4CheckBox.setChecked(True)
            if first_ch=='5':
                displaygui.Ch5CheckBox.setChecked(True)
            self.first_time=False
        
    def itemActivated_event(item):
        return item.text()    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

