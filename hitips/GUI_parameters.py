import numpy as np
from PyQt5.QtWidgets import QComboBox, QSlider, QSpinBox
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
from distutils import util

class Gui_Params(object):

    def __init__(self, analysisgui, inout_resource_gui, displaygui, ImDisplay=None):
        self.ImDisplay = ImDisplay
        self.displaygui = displaygui
        self.AnalysisGui = analysisgui
        self.inout_resource_gui = inout_resource_gui
        self.spot_params_dict = self.INITIALIZE_SPOT_ANALYSIS_PARAMS()
        self.AnalysisGui.spotchannelselect.currentIndexChanged.connect(lambda: self.UPDATE_SPOT_ANALYSIS_GUI_PARAMS())
        self.inout_resource_gui.OutFldrButton.clicked.connect(lambda: self.OUTPUT_FOLDER_LOADBTN())
        self.Output_dir = []
        self.update_values()
        self.params_dict = self.gather_analysis_params()

    @staticmethod
    def headless_config_loader(file_path):
        det_method = ["Laplacian of Gaussian", "Gaussian", "Intensity Threshold", "Enhanced LOG"]
        thresh_method = ["Auto", "Manual"]

        conf = pd.read_csv(file_path, index_col='Parameter')
        conf_dict = conf.to_dict()['Value']

        def get_bool(key):
            return bool(util.strtobool(conf_dict[key]))

        params_dict = {
            "nuclei_channel": conf_dict['nuclei_channel'],
            "nuclei_detection_method": conf_dict['nuclei_detection_method'],
            "nuclei_z_project": get_bool('nuclei_z_project'),
            "remove_boundary_nuclei": get_bool('remove_boundary_nuclei'),
            "nuclei_detection": int(float(conf_dict['nuclei_detection'])),
            "nuclei_separation": int(float(conf_dict['nuclei_separation'])),
            "nuclei_area": int(float(conf_dict['nuclei_area'])),
            "ch1_spot": get_bool('ch1_spot'),
            "ch2_spot": get_bool('ch2_spot'),
            "ch3_spot": get_bool('ch3_spot'),
            "ch4_spot": get_bool('ch4_spot'),
            "ch5_spot": get_bool('ch5_spot'),
            "spot_coordinates": conf_dict['spot_coordinates'],
            "spot_z_project": get_bool('spot_z_project'),
            "Nuclei_Info_CheckBox_status": get_bool('Nuclei_Info_CheckBox_status'),
            "Spots_Location_status": get_bool('Spots_Location_status'),
            "Spots_Tracking_status": get_bool('Spots_Tracking_status'),
            "Nuclei_MaskCheckBox_status": get_bool('Nuclei_MaskCheckBox_status'),
            "Nuclei_MaxZproject_CheckBox_status": get_bool('Nuclei_MaxZproject_CheckBox_status'),
            "RemoveBrightJunk_status_check": get_bool('RemoveBrightJunk_status_check'),
            "NumCPUsSpinBox_value": int(conf_dict['NumCPUsSpinBox_value']),
            "Cell_Tracking_check_status": get_bool('Cell_Tracking_check_status'),
            "NucTrackingMethod": conf_dict['NucTrackingMethod'],
            "NucSearchRadius": int(conf_dict['NucSearchRadius']),
            "SpotSearchRadius_value": int(conf_dict['SpotSearchRadius_value']),
            "Sec_SpotSearchRadius_value": int(conf_dict['Sec_SpotSearchRadius_value']),
            "Seceondary_Channel_index": int(conf_dict['Seceondary_Channel_index']),
            "Secondary_Area_index": int(conf_dict['Secondary_Area_index']),
            "MintrackLength_value": int(conf_dict['MintrackLength_value']),
            "maxspotspercell_value": int(conf_dict['maxspotspercell_value']),
            "minburstduration_value": int(conf_dict['minburstduration_value']),
            "FittingMethod_index": int(conf_dict['FittingMethod_index']),
            "patchsize": int(conf_dict['patchsize']),
            "IntegratedIntensity_fitStatus": int(conf_dict['IntegratedIntensity_fitStatus']),
            "Registrationmethod": conf_dict['Registrationmethod'],
            "IntegratedIntensity_Index": int(conf_dict['IntegratedIntensity_Index']),
            "PSFsize_value": float(conf_dict['PSFsize_value'])
        }

        spot_params_dict = {
            ch: np.array([
                det_method.index(conf_dict[f'{ch.lower()}_spot_detection_method']),
                thresh_method.index(conf_dict[f'{ch.lower()}_spot_threshold_method']),
                int(float(conf_dict[f'{ch.lower()}_spot_threshold_value'])),
                int(float(conf_dict[f'{ch.lower()}_kernel_size'])),
                int(float(conf_dict[f'{ch.lower()}_spots/ch'])),
                int(float(conf_dict[f'{ch.lower()}_spots_area_min'])),
                int(float(conf_dict[f'{ch.lower()}_spots_area_max'])),
                int(float(conf_dict[f'{ch.lower()}_spots_integrated_intensity']))
            ], dtype=int)
            for ch in ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']
        }

        params_dict["spot_params_dict"] = spot_params_dict
        return params_dict

    def gather_analysis_params(self):
        self.UPDATE_SPOT_ANALYSIS_PARAMS()
        params_dict = {
            "spot_params_dict": self.spot_params_dict,
            "NucInfoChkBox_check_status": self.AnalysisGui.NucInfoChkBox.isChecked(),
            "SpotsLocation_check_status": self.AnalysisGui.SpotsLocation.isChecked(),
            "Spot_Tracking_check_status": self.AnalysisGui.Spot_Tracking.isChecked(),
            "SpotLocationCbox_currentText": self.AnalysisGui.SpotLocationCbox.currentText(),
            "SpotsDistance_check_status": self.AnalysisGui.SpotsDistance.isChecked(),
            "NucMaskCheckBox_status_check": self.AnalysisGui.NucMaskCheckBox.isChecked(),
            "NucMaxZprojectCheckBox_status_check": self.AnalysisGui.NucMaxZprojectCheckBox.isChecked(),
            "SpotMaxZProject_status_check": self.AnalysisGui.SpotMaxZProject.isChecked(),
            "RemoveBrightJunk_status_check": self.AnalysisGui.RemoveBrightJunk.isChecked(),
            "SpotCh1CheckBox_status_check": self.AnalysisGui.SpotCh1CheckBox.isChecked(),
            "SpotCh2CheckBox_status_check": self.AnalysisGui.SpotCh2CheckBox.isChecked(),
            "SpotCh3CheckBox_status_check": self.AnalysisGui.SpotCh3CheckBox.isChecked(),
            "SpotCh4CheckBox_status_check": self.AnalysisGui.SpotCh4CheckBox.isChecked(),
            "SpotCh5CheckBox_status_check": self.AnalysisGui.SpotCh5CheckBox.isChecked(),
            "NucleiChannel": self.AnalysisGui.NucleiChannel.currentText(),
            "NumCPUsSpinBox_value": self.inout_resource_gui.NumCPUsSpinBox.value(),
            "Cell_Tracking_check_status": self.AnalysisGui.Cell_Tracking.isChecked(),
            "NucTrackingMethod_currentText": self.AnalysisGui.NucTrackMethod.currentText(),
            "NucSearchRadiusSpinbox_current_value": self.AnalysisGui.NucSearchRadiusSpinbox.value(),
            "SpotSearchRadiusSpinbox_current_value": self.AnalysisGui.SpotSearchRadiusSpinbox.value(),
            "Sec_SpotSearchRadiusSpinbox_current_value": self.AnalysisGui.Sec_SpotSearchRadiusSpinbox.value(),
            "spotchannelselect_currentText": self.AnalysisGui.spotchannelselect.currentText(),
            "SecChannel_current_index": self.AnalysisGui.SecChannel.currentIndex(),
            "SecArea_current_index": self.AnalysisGui.SecArea.currentIndex(),
            "MintrackLengthSpinbox_current_value": self.AnalysisGui.MintrackLengthSpinbox.value(),
            "maxspotspercellSpinbox_current_value": self.AnalysisGui.maxspotspercellSpinbox.value(),
            "minburstdurationSpinbox_current_value": self.AnalysisGui.minburstdurationSpinbox.value(),
            "FittingMethod_index": self.AnalysisGui.Fittingnmethod.currentIndex(),
            "patchsize_currentText": int(self.AnalysisGui.patchsize.currentText()),
            "IntegratedIntensity_fitStatus": self.AnalysisGui.IntegratedIntensityCbox.currentIndex(),
            "NucDetectMethod_currentText": self.AnalysisGui.NucDetectMethod.currentText(),
            "Registrationmethod_currentText": self.AnalysisGui.Registrationmethod.currentText(),
            "NucSeparationSlider_value": self.AnalysisGui.NucSeparationSlider.value(),
            "NucDetectionSlider_value": self.AnalysisGui.NucDetectionSlider.value(),
            "NucleiAreaSlider_value": self.AnalysisGui.NucleiAreaSlider.value(),
            "NucRemoveBoundaryCheckBox_isChecked": self.AnalysisGui.NucRemoveBoundaryCheckBox.isChecked(),
            "Resize_Factor": self.AnalysisGui.ResizeFactor.value(),
            "IntegratedIntensityCbox_currentIndex": self.AnalysisGui.IntegratedIntensityCbox.currentIndex(),
            "SpotareaminSpinBox_value": self.AnalysisGui.SpotareaminSpinBox.value(),
            "SpotareamaxSpinBox_value": self.AnalysisGui.SpotareamaxSpinBox.value(),
            "SpotIntegratedIntensitySpinBox_value": self.AnalysisGui.SpotIntegratedIntensitySpinBox.value(),
            "PSFsizeSpinBox_value": self.AnalysisGui.PSFsizeSpinBox.value(),
            "Output_dir": self.Output_dir,
            "Seceondary_Channel_index": self.AnalysisGui.SecChannel.currentIndex(),
            "Secondary_Area_index": self.AnalysisGui.SecArea.currentIndex(),
            "ch1_spot_detection_method": self.spot_params_dict['Ch1'][0],
            "ch1_spot_threshold_method": self.spot_params_dict['Ch1'][1],
            "ch1_spot_threshold_value": self.spot_params_dict['Ch1'][2],
            "ch1_kernel_size": self.spot_params_dict['Ch1'][3],
            "ch1_spots/ch": self.spot_params_dict['Ch1'][4],
            "ch1_spots_area_min": self.spot_params_dict['Ch1'][5],
            "ch1_spots_area_max": self.spot_params_dict['Ch1'][6],
            "ch1_spots_integrated_intensity": self.spot_params_dict['Ch1'][7]
        }
    
        return params_dict

    def update_values(self):
        self.NucInfoChkBox_check_status = self.AnalysisGui.NucInfoChkBox.isChecked()
        self.SpotsLocation_check_status = self.AnalysisGui.SpotsLocation.isChecked()
        self.Spot_Tracking_check_status = self.AnalysisGui.Spot_Tracking.isChecked()
        self.SpotLocationCbox_currentText = self.AnalysisGui.SpotLocationCbox.currentText()
        self.SpotsDistance_check_status = self.AnalysisGui.SpotsDistance.isChecked()
        self.NucMaskCheckBox_status_check = self.AnalysisGui.NucMaskCheckBox.isChecked()
        self.NucMaxZprojectCheckBox_status_check = self.AnalysisGui.NucMaxZprojectCheckBox.isChecked()
        self.SpotMaxZProject_status_check = self.AnalysisGui.SpotMaxZProject.isChecked()
        self.RemoveBrightJunk_status_check = self.AnalysisGui.RemoveBrightJunk.isChecked()
        self.SpotCh1CheckBox_status_check = self.AnalysisGui.SpotCh1CheckBox.isChecked()
        self.SpotCh2CheckBox_status_check = self.AnalysisGui.SpotCh2CheckBox.isChecked()
        self.SpotCh3CheckBox_status_check = self.AnalysisGui.SpotCh3CheckBox.isChecked()
        self.SpotCh4CheckBox_status_check = self.AnalysisGui.SpotCh4CheckBox.isChecked()
        self.SpotCh5CheckBox_status_check = self.AnalysisGui.SpotCh5CheckBox.isChecked()
        self.NucleiChannel_index = self.AnalysisGui.NucleiChannel.currentIndex()
        self.NumCPUsSpinBox_value = self.inout_resource_gui.NumCPUsSpinBox.value()
        self.Cell_Tracking_check_status = self.AnalysisGui.Cell_Tracking.isChecked()
        self.NucTrackingMethod_currentText = self.AnalysisGui.NucTrackMethod.currentText()
        self.NucSearchRadiusSpinbox_current_value = self.AnalysisGui.NucSearchRadiusSpinbox.value()
        self.SpotSearchRadiusSpinbox_current_value = self.AnalysisGui.SpotSearchRadiusSpinbox.value()
        self.Sec_SpotSearchRadiusSpinbox_current_value = self.AnalysisGui.Sec_SpotSearchRadiusSpinbox.value()
        self.SecChannel_current_index = self.AnalysisGui.SecChannel.currentIndex()
        self.SecArea_current_index = self.AnalysisGui.SecArea.currentIndex()
        self.MintrackLengthSpinbox_current_value = self.AnalysisGui.MintrackLengthSpinbox.value()
        self.maxspotspercellSpinbox_current_value = self.AnalysisGui.maxspotspercellSpinbox.value()
        self.minburstdurationSpinbox_current_value = self.AnalysisGui.minburstdurationSpinbox.value()
        self.FittingMethod_index = self.AnalysisGui.Fittingnmethod.currentIndex()
        self.patchsize_currentText = int(self.AnalysisGui.patchsize.currentText())
        self.IntegratedIntensity_fitStatus = self.AnalysisGui.IntegratedIntensityCbox.currentIndex()
        self.NucDetectMethod_currentText = self.AnalysisGui.NucDetectMethod.currentText()
        self.Registrationmethod_currentText = self.AnalysisGui.Registrationmethod.currentText()
        self.NucSeparationSlider_value = self.AnalysisGui.NucSeparationSlider.value()
        self.NucDetectionSlider_value = self.AnalysisGui.NucDetectionSlider.value()
        self.NucleiAreaSlider_value = self.AnalysisGui.NucleiAreaSlider.value()
        self.NucRemoveBoundaryCheckBox_isChecked = self.AnalysisGui.NucRemoveBoundaryCheckBox.isChecked()
        self.Resize_Factor = self.AnalysisGui.ResizeFactor.value()
        self.IntegratedIntensityCbox_currentIndex = self.AnalysisGui.IntegratedIntensityCbox.currentIndex()
        self.SpotareaminSpinBox_value = self.AnalysisGui.SpotareaminSpinBox.value()
        self.SpotareamaxSpinBox_value = self.AnalysisGui.SpotareamaxSpinBox.value()
        self.SpotIntegratedIntensitySpinBox_value = self.AnalysisGui.SpotIntegratedIntensitySpinBox.value()
        self.PSFsizeSpinBox_value = self.AnalysisGui.PSFsizeSpinBox.value()
        self.spotchannelselect_currentText = self.AnalysisGui.spotchannelselect.currentText()
        self.params_dict = self.gather_analysis_params()
    
    def set_ImDisplay(self, ImDisplay):
        self.ImDisplay = ImDisplay
    
    def INITIALIZE_SPOT_ANALYSIS_PARAMS(self):

        def get_channel_params():
            return np.array([
                self.AnalysisGui.spotanalysismethod.currentIndex(),
                self.AnalysisGui.thresholdmethod.currentIndex(),
                self.AnalysisGui.ThresholdSlider.value(), 
                self.AnalysisGui.SensitivitySpinBox.value(),
                self.AnalysisGui.ResizeFactor.value(),
                self.AnalysisGui.SpotareaminSpinBox.value(),
                self.AnalysisGui.SpotareamaxSpinBox.value(),
                self.AnalysisGui.SpotIntegratedIntensitySpinBox.value()
            ])

        self.spot_params_dict = {
            f"Ch{i}": get_channel_params() for i in range(1, 6)
        }

        return self.spot_params_dict
    
    def UPDATE_SPOT_ANALYSIS_PARAMS(self):

        def get_channel_params():
            return np.array([
                self.AnalysisGui.spotanalysismethod.currentIndex(),
                self.AnalysisGui.thresholdmethod.currentIndex(),
                self.AnalysisGui.ThresholdSlider.value(), 
                self.AnalysisGui.SensitivitySpinBox.value(),
                self.AnalysisGui.ResizeFactor.value(),
                self.AnalysisGui.SpotareaminSpinBox.value(),
                self.AnalysisGui.SpotareamaxSpinBox.value(),
                self.AnalysisGui.SpotIntegratedIntensitySpinBox.value()
            ])

        current_channel = self.AnalysisGui.spotchannelselect.currentText()

        if current_channel == 'All':
            self.spot_params_dict = {
                f"Ch{i}": get_channel_params() for i in range(1, 6)
            }
        elif current_channel in ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']:
            self.spot_params_dict[current_channel] = get_channel_params()
        
            
    def UPDATE_SPOT_ANALYSIS_GUI_PARAMS(self):

        current_channel = self.AnalysisGui.spotchannelselect.currentText()

        if current_channel in ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']:

            gui_elements = [
                self.AnalysisGui.spotanalysismethod,  
                self.AnalysisGui.thresholdmethod,     
                self.AnalysisGui.ThresholdSlider,     
                self.AnalysisGui.SensitivitySpinBox,  
                self.AnalysisGui.ResizeFactor,    
                self.AnalysisGui.SpotareaminSpinBox,  
                self.AnalysisGui.SpotareamaxSpinBox,  
                self.AnalysisGui.SpotIntegratedIntensitySpinBox 
            ]

            for i, gui_element in enumerate(gui_elements):
                value = np.array(self.spot_params_dict[current_channel][i]).astype(int)
    
                try:
                    if isinstance(gui_element, QComboBox):
                        gui_element.currentIndexChanged.disconnect()
                    elif isinstance(gui_element, (QSlider, QSpinBox)):
                        gui_element.valueChanged.disconnect()
                except TypeError:
                    pass
    
                if isinstance(gui_element, QComboBox):
                    gui_element.setCurrentIndex(value)
                    gui_element.currentIndexChanged.connect(lambda: self.UPDATE_SPOT_ANALYSIS_PARAMS()) 
                    gui_element.currentIndexChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
                elif isinstance(gui_element, (QSlider, QSpinBox)):
                    gui_element.setValue(value)
                    gui_element.valueChanged.connect(lambda: self.UPDATE_SPOT_ANALYSIS_PARAMS())  
                    gui_element.valueChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))

    def SAVE_CONFIGURATION(self, csv_filename):
        det_method = ["Laplacian of Gaussian", "Gaussian", "Intensity Threshold", "Enhanced LOG"]
        thresh_method = ["Auto", "Manual"]
    
        def get_spot_params(channel):
            params = self.spot_params_dict[channel]
            return [
                det_method[int(params[0])],
                thresh_method[int(params[1])],
                params[2],
                params[3],
                params[4],
                params[5],
                params[6],
                params[7]
            ]
    
        config_data = {
            "nuclei_channel": self.AnalysisGui.NucleiChannel.currentText(),
            "nuclei_detection_method": self.AnalysisGui.NucDetectMethod.currentText(),
            "nuclei_z_project": self.AnalysisGui.NucMaxZprojectCheckBox.isChecked(),
            "remove_boundary_nuclei": self.AnalysisGui.NucRemoveBoundaryCheckBox.isChecked(),
            "nuclei_detection": self.AnalysisGui.NucDetectionSlider.value(),
            "nuclei_separation": self.AnalysisGui.NucSeparationSlider.value(),
            "nuclei_area": self.AnalysisGui.NucleiAreaSlider.value(),
            "ch1_spot": self.AnalysisGui.SpotCh1CheckBox.isChecked(),
            "ch2_spot": self.AnalysisGui.SpotCh2CheckBox.isChecked(),
            "ch3_spot": self.AnalysisGui.SpotCh3CheckBox.isChecked(),
            "ch4_spot": self.AnalysisGui.SpotCh4CheckBox.isChecked(),
            "ch5_spot": self.AnalysisGui.SpotCh5CheckBox.isChecked(),
            "spot_coordinates": self.AnalysisGui.SpotLocationCbox.currentText(),
            "spot_z_project": self.AnalysisGui.SpotMaxZProject.isChecked(),
            "Nuclei_Info_CheckBox_status": self.AnalysisGui.NucInfoChkBox.isChecked(),
            "Spots_Location_status": self.AnalysisGui.SpotsLocation.isChecked(),
            "Spots_Tracking_status": self.AnalysisGui.Spot_Tracking.isChecked(),
            "Nuclei_MaskCheckBox_status": self.AnalysisGui.NucMaskCheckBox.isChecked(),
            "Nuclei_MaxZproject_CheckBox_status": self.AnalysisGui.NucMaxZprojectCheckBox.isChecked(),
            "RemoveBrightJunk_status_check": self.AnalysisGui.RemoveBrightJunk.isChecked(),
            "NumCPUsSpinBox_value": self.inout_resource_gui.NumCPUsSpinBox.value(),
            "Cell_Tracking_check_status": self.AnalysisGui.Cell_Tracking.isChecked(),
            "NucTrackingMethod": self.AnalysisGui.NucTrackMethod.currentText(),
            "NucSearchRadius": self.AnalysisGui.NucSearchRadiusSpinbox.value(),
            "SpotSearchRadius_value": self.AnalysisGui.SpotSearchRadiusSpinbox.value(),
            "Sec_SpotSearchRadius_value": self.AnalysisGui.Sec_SpotSearchRadiusSpinbox.value(),
            "Seceondary_Channel_index": self.AnalysisGui.SecChannel.currentIndex(),
            "Secondary_Area_index": self.AnalysisGui.SecArea.currentIndex(),
            "MintrackLength_value": self.AnalysisGui.MintrackLengthSpinbox.value(),
            "maxspotspercell_value": self.AnalysisGui.maxspotspercellSpinbox.value(),
            "minburstduration_value": self.AnalysisGui.minburstdurationSpinbox.value(),
            "FittingMethod_index": self.AnalysisGui.Fittingnmethod.currentIndex(),
            "patchsize": int(self.AnalysisGui.patchsize.currentText()),
            "IntegratedIntensity_fitStatus": self.AnalysisGui.IntegratedIntensityCbox.currentIndex(),
            "Registrationmethod": self.AnalysisGui.Registrationmethod.currentText(),
            "IntegratedIntensity_Index": self.AnalysisGui.IntegratedIntensityCbox.currentIndex(),
            "PSFsize_value": self.AnalysisGui.PSFsizeSpinBox.value(),
            "SecChannel_current_index": self.AnalysisGui.SecChannel.currentIndex(),
            "SecArea_current_index": self.AnalysisGui.SecArea.currentIndex(),
            "SpotsDistance_check_status": self.AnalysisGui.SpotsDistance.isChecked(),
            "SpotIntegratedIntensitySpinBox_value": self.AnalysisGui.SpotIntegratedIntensitySpinBox.value(),
            "Resize_Factor": self.AnalysisGui.ResizeFactor.value()
        }
    
        for ch in ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']:
            ch_params = get_spot_params(ch)
            config_data.update({
                f"{ch.lower()}_spot_detection_method": ch_params[0],
                f"{ch.lower()}_spot_threshold_method": ch_params[1],
                f"{ch.lower()}_spot_threshold_value": ch_params[2],
                f"{ch.lower()}_kernel_size": ch_params[3],
                f"{ch.lower()}_spots/ch": ch_params[4],
                f"{ch.lower()}_spots_area_min": ch_params[5],
                f"{ch.lower()}_spots_area_max": ch_params[6],
                f"{ch.lower()}_spots_integrated_intensity": ch_params[7]
            })
    
        config_df = pd.DataFrame(list(config_data.items()), columns=['Parameter', 'Value'])
        config_df.set_index('Parameter', inplace=True)
        config_df.to_csv(csv_filename, index=True)
        
    def file_save(self):
        self.fnames, _  = QtWidgets.QFileDialog.getSaveFileName(None, 'Save File')
        self.csv_filename = self.fnames + r'.csv'
        self.SAVE_CONFIGURATION(self.csv_filename)
        
    def LOAD_CONFIGURATION(self):
        det_method = ["Laplacian of Gaussian", "Gaussian", "Intensity Threshold", "Enhanced LOG"] 
        thresh_method = ["Auto", "Manual"]
        
        options = QtWidgets.QFileDialog.Options()
        self.fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select Configuration File...',
                                                                '', "Configuration files (*.csv)", options=options)
        if not self.fnames:
            return  # Exit if no file selected

        conf = pd.read_csv(self.fnames[0], index_col='Parameter')
        conf_dict = conf.to_dict()['Value']

        def get_bool(key):
            return bool(util.strtobool(conf_dict[key]))
        
        def get_param(key, default):
            return conf_dict.get(key, default)

        try:
            self.AnalysisGui.NucleiChannel.setCurrentText(conf_dict['nuclei_channel'])
            self.AnalysisGui.NucDetectMethod.setCurrentText(conf_dict['nuclei_detection_method'])
            self.AnalysisGui.NucMaxZprojectCheckBox.setChecked(get_bool('nuclei_z_project'))
            self.AnalysisGui.NucRemoveBoundaryCheckBox.setChecked(get_bool('remove_boundary_nuclei'))
            self.AnalysisGui.NucDetectionSlider.setValue(int(float(conf_dict['nuclei_detection'])))
            self.AnalysisGui.NucSeparationSlider.setValue(int(float(conf_dict['nuclei_separation'])))
            self.AnalysisGui.NucleiAreaSlider.setValue(int(float(conf_dict['nuclei_area'])))
            self.AnalysisGui.SpotCh1CheckBox.setChecked(get_bool('ch1_spot'))
            self.AnalysisGui.SpotCh2CheckBox.setChecked(get_bool('ch2_spot'))
            self.AnalysisGui.SpotCh3CheckBox.setChecked(get_bool('ch3_spot'))
            self.AnalysisGui.SpotCh4CheckBox.setChecked(get_bool('ch4_spot'))
            self.AnalysisGui.SpotCh5CheckBox.setChecked(get_bool('ch5_spot'))
            self.AnalysisGui.SpotLocationCbox.setCurrentText(conf_dict['spot_coordinates'])
            self.AnalysisGui.SpotMaxZProject.setChecked(get_bool('spot_z_project'))
            self.AnalysisGui.spotanalysismethod.setCurrentText(conf_dict['ch1_spot_detection_method'])
            self.AnalysisGui.thresholdmethod.setCurrentText(conf_dict['ch1_spot_threshold_method'])
            self.AnalysisGui.ThresholdSlider.setValue(int(float(conf_dict['ch1_spot_threshold_value'])))
            self.AnalysisGui.SensitivitySpinBox.setValue(int(float(conf_dict['ch1_kernel_size'])))
            self.AnalysisGui.ResizeFactor.setValue(int(float(conf_dict['ch1_spots/ch'])))
            self.AnalysisGui.SpotareaminSpinBox.setValue(int(float(conf_dict['ch1_spots_area_min'])))
            self.AnalysisGui.SpotareamaxSpinBox.setValue(int(float(conf_dict['ch1_spots_area_max'])))
            self.AnalysisGui.SpotIntegratedIntensitySpinBox.setValue(int(float(conf_dict['ch1_spots_integrated_intensity'])))
            self.AnalysisGui.NucInfoChkBox.setChecked(get_bool('Nuclei_Info_CheckBox_status'))
            self.AnalysisGui.SpotsLocation.setChecked(get_bool('Spots_Location_status'))
            self.AnalysisGui.Spot_Tracking.setChecked(get_bool('Spots_Tracking_status'))
            self.AnalysisGui.NucMaskCheckBox.setChecked(get_bool('Nuclei_MaskCheckBox_status'))
            self.AnalysisGui.NucMaxZprojectCheckBox.setChecked(get_bool('Nuclei_MaxZproject_CheckBox_status'))
            self.AnalysisGui.RemoveBrightJunk.setChecked(get_bool('RemoveBrightJunk_status_check'))
            self.inout_resource_gui.NumCPUsSpinBox.setValue(int(conf_dict['NumCPUsSpinBox_value']))
            self.AnalysisGui.Cell_Tracking.setChecked(get_bool('Cell_Tracking_check_status'))
            self.AnalysisGui.NucTrackMethod.setCurrentText(conf_dict['NucTrackingMethod'])
            self.AnalysisGui.NucSearchRadiusSpinbox.setValue(int(conf_dict['NucSearchRadius']))
            self.AnalysisGui.SpotSearchRadiusSpinbox.setValue(int(conf_dict['SpotSearchRadius_value']))
            self.AnalysisGui.Sec_SpotSearchRadiusSpinbox.setValue(int(conf_dict['Sec_SpotSearchRadius_value']))
            # Use consistent parameter naming
            self.AnalysisGui.SecChannel.setCurrentIndex(int(get_param('Seceondary_Channel_index', 0)))
            self.AnalysisGui.SecArea.setCurrentIndex(int(get_param('Secondary_Area_index', 0)))
            self.AnalysisGui.MintrackLengthSpinbox.setValue(int(conf_dict['MintrackLength_value']))
            self.AnalysisGui.maxspotspercellSpinbox.setValue(int(conf_dict['maxspotspercell_value']))
            self.AnalysisGui.minburstdurationSpinbox.setValue(int(conf_dict['minburstduration_value']))
            self.AnalysisGui.Fittingnmethod.setCurrentIndex(int(conf_dict['FittingMethod_index']))
            self.AnalysisGui.patchsize.setCurrentText(str(conf_dict['patchsize']))
            self.AnalysisGui.IntegratedIntensityCbox.setCurrentIndex(int(conf_dict['IntegratedIntensity_fitStatus']))
            self.AnalysisGui.Registrationmethod.setCurrentText(conf_dict['Registrationmethod'])
            self.AnalysisGui.IntegratedIntensityCbox.setCurrentIndex(int(conf_dict['IntegratedIntensity_Index']))
            self.AnalysisGui.PSFsizeSpinBox.setValue(float(conf_dict['PSFsize_value']))
            self.AnalysisGui.SpotIntegratedIntensitySpinBox.setValue(int(conf_dict['SpotIntegratedIntensitySpinBox_value']))
            self.AnalysisGui.ResizeFactor.setValue(int(conf_dict["Resize_Factor"]))

            # Load spot parameters for each channel
            for ch in ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']:
                ch_lower = ch.lower()
                try:
                    self.spot_params_dict[ch] = np.array([
                        det_method.index(conf_dict[f'{ch_lower}_spot_detection_method']),
                        thresh_method.index(conf_dict[f'{ch_lower}_spot_threshold_method']),
                        int(float(conf_dict[f'{ch_lower}_spot_threshold_value'])),
                        int(float(conf_dict[f'{ch_lower}_kernel_size'])),
                        int(float(conf_dict[f'{ch_lower}_spots/ch'])),
                        int(float(conf_dict[f'{ch_lower}_spots_area_min'])),
                        int(float(conf_dict[f'{ch_lower}_spots_area_max'])),
                        int(float(conf_dict[f'{ch_lower}_spots_integrated_intensity']))
                    ], dtype=int)
                except (KeyError, ValueError) as e:
                    print(f"Warning: Error loading parameters for {ch}: {e}")
                    continue

        except Exception as e:
            pass
            # print(f"Error loading configuration: {e}")
            # QtWidgets.QMessageBox.warning(None, "Error", f"Failed to load configuration file: {e}")

    def OUTPUT_FOLDER_LOADBTN(self):
        options = QtWidgets.QFileDialog.Options()
        self.Output_dir = QtWidgets.QFileDialog.getExistingDirectory(None, caption= "Select Output Directory", options=options)
        self.params_dict["Output_dir"] = self.Output_dir
