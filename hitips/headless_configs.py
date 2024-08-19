import numpy as np
import pandas as pd
from distutils import util

def headless_config_loader(file_path):
    det_method = ["Laplacian of Gaussian", "Gaussian", "Intensity Threshold", "Enhanced LOG"]
    thresh_method = ["Auto", "Manual"]

    conf = pd.read_csv(file_path)
    conf_dict = dict(zip(conf['Parameter'], conf['Value']))

    def get_bool(key):
        return bool(util.strtobool(conf_dict[key]))

    params_dict = {
        "NucleiChannel": conf_dict['nuclei_channel'],
        "NucDetectMethod_currentText": conf_dict['nuclei_detection_method'],
        "nuclei_z_project": get_bool('nuclei_z_project'),
        "NucRemoveBoundaryCheckBox_isChecked": get_bool('remove_boundary_nuclei'),
        "NucDetectionSlider_value": int(float(conf_dict['nuclei_detection'])),
        "NucSeparationSlider_value": int(float(conf_dict['nuclei_separation'])),
        "NucleiAreaSlider_value": int(float(conf_dict['nuclei_area'])),
        "SpotCh1CheckBox_status_check": get_bool('ch1_spot'),
        "SpotCh2CheckBox_status_check": get_bool('ch2_spot'),
        "SpotCh3CheckBox_status_check": get_bool('ch3_spot'),
        "SpotCh4CheckBox_status_check": get_bool('ch4_spot'),
        "SpotCh5CheckBox_status_check": get_bool('ch5_spot'),
        "spot_coordinates": conf_dict['spot_coordinates'],
        "SpotMaxZProject_status_check": get_bool('spot_z_project'),
        "SpotLocationCbox_currentText": conf_dict['spot_coordinates'],
        "NucInfoChkBox_check_status": get_bool('Nuclei_Info_CheckBox_status'),
        "SpotsLocation_check_status": get_bool('Spots_Location_status'),
        "Spot_Tracking_check_status": get_bool('Spots_Tracking_status'),
        "NucMaskCheckBox_status_check": get_bool('Nuclei_MaskCheckBox_status'),
        "NucMaxZprojectCheckBox_status_check": get_bool('Nuclei_MaxZproject_CheckBox_status'),
        "RemoveBrightJunk_status_check": get_bool('RemoveBrightJunk_status_check'),
        "NumCPUsSpinBox_value": int(conf_dict['NumCPUsSpinBox_value']),
        "Cell_Tracking_check_status": get_bool('Cell_Tracking_check_status'),
        "NucTrackingMethod_currentText": conf_dict['NucTrackingMethod'],
        "NucSearchRadiusSpinbox_current_value": int(conf_dict['NucSearchRadius']),
        "SpotSearchRadiusSpinbox_current_value": int(conf_dict['SpotSearchRadius_value']),
        "Sec_SpotSearchRadiusSpinbox_current_value": int(conf_dict['Sec_SpotSearchRadius_value']),
        "Seceondary_Channel_index": int(conf_dict['Seceondary_Channel_index']),
        "Secondary_Area_index": int(conf_dict['Secondary_Area_index']),
        "MintrackLengthSpinbox_current_value": int(conf_dict['MintrackLength_value']),
        "maxspotspercellSpinbox_current_value": int(conf_dict['maxspotspercell_value']),
        "minburstdurationSpinbox_current_value": int(conf_dict['minburstduration_value']),
        "FittingMethod_index": int(conf_dict['FittingMethod_index']),
        "patchsize_currentText": int(conf_dict['patchsize']),
        "IntegratedIntensity_fitStatus": int(conf_dict['IntegratedIntensity_fitStatus']),
        "Registrationmethod_currentText": conf_dict['Registrationmethod'],
        "IntegratedIntensityCbox_currentIndex": int(conf_dict['IntegratedIntensity_Index']),
        "PSFsizeSpinBox_value": float(conf_dict['PSFsize_value']),
        "SecChannel_current_index": int(conf_dict["SecChannel_current_index"]),
        "SecArea_current_index": int(conf_dict["SecArea_current_index"]),
        "SpotsDistance_check_status": get_bool('SpotsDistance_check_status'),
        "SpotIntegratedIntensitySpinBox_value": int(conf_dict["SpotIntegratedIntensitySpinBox_value"]),
        "Resize_Factor": int(conf_dict["Resize_Factor"])
        
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
