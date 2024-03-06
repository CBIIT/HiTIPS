Results
=======

Results Folder
--------------

The results folder is organized into several subfolders to store different types of output data from the HiTIPS analysis:

- `configuration_files`: Includes the 'analysis_configuration' file that were used or generated during the analysis, 'dependencies_versions' that includes all the packages and their versions used for the analysis, and 'erros_log' that logs the errors (if any) during the analysis.

- `nuclei_masks`: Holds the generated masks for the segmented cell nuclei, crucial for identifying and analyzing individual cells and nuclei tracking.

- `well_nuclei_results`: Results related to the nuclei that have been analyzed and saved separately for each well of the plate.

- `well_spots_locations`: Contains information about the locations of spots detected and saved separately for each well of the plate.

- `whole_plate_results`: Stores comprehensive results that pertain to the entire plate rather than individual wells or nuclei.

- `cell_tracking`: Contains files related to the tracking of nuclei and spots across different time points. This folder contais the followinng subfolders:

    - 'annotated_spot_image_patches': Contains time-stacks of the nuclei with detected spots. Spots are tracked over time and annotated by spot label.
    
    - 'single_track_images': This folder includes time-stacks of the each tracked nuclei before and after frame to frame registration.
    
    - 'single_track_tables': Contains information related to each tracked nucleous over time. Each file is for a separate nucleus.
    
    - 'spot_image_patches': Contains time-stacks of the nuclei with detected spots. Spots are tracked over time and but are not annotated.

    - 'spot_intensity_tables': This contais both csv and txt file for each spot that stores the location, intensity and HMM information related to the spots.
    
    -'whole_field_track_images': This folder contains tracked and annotated nuclei within each field over time.


Table of information for the segmented cell nuclei
--------------------------------------------------

The table below describes the individual columns in that contain all the extracted information from the cell nuceli using HiTIPS. 



.. table:: Column Descriptions for Nuclei_Information.csv

   +-------------------+--------------------------------------------------------+
   | Column Name       | Description                                            |
   +===================+========================================================+
   | Experiment        | Identifier for the experiment or study.                |
   +-------------------+--------------------------------------------------------+
   | column            | The column location of the cell nucleus.               |
   +-------------------+--------------------------------------------------------+
   | row               | The row location of the cell nucleus.                  |
   +-------------------+--------------------------------------------------------+
   | time_point        | The time point at which the data was recorded.         |
   +-------------------+--------------------------------------------------------+
   | field_index       | Index indicating the field or view within the well.    |
   +-------------------+--------------------------------------------------------+
   | z_slice           | The slice or depth at which the image is taken.        |
   +-------------------+--------------------------------------------------------+
   | action_index      | Index representing a specific action or event.         |
   +-------------------+--------------------------------------------------------+
   | cell_index        | Index for identifying individual cells.                |
   +-------------------+--------------------------------------------------------+
   | centroid-0        | The X-coordinate of the nucleus centroid.              |
   +-------------------+--------------------------------------------------------+
   | centroid-1        | The Y-coordinate of the nucleus centroid.              |
   +-------------------+--------------------------------------------------------+
   | orientation       | Orientation of the nucleus, in radians.                |
   +-------------------+--------------------------------------------------------+
   | major_axis_length | Length of the major axis of the nucleus.               |
   +-------------------+--------------------------------------------------------+
   | minor_axis_length | Length of the minor axis of the nucleus.               |
   +-------------------+--------------------------------------------------------+
   | area              | Area of the nucleus.                                   |
   +-------------------+--------------------------------------------------------+
   | max_intensity     | Maximum intensity value within the nucleus.            |
   +-------------------+--------------------------------------------------------+
   | min_intensity     | Minimum intensity value within the nucleus.            |
   +-------------------+--------------------------------------------------------+
   | mean_intensity    | Mean intensity value within the nucleus.               |
   +-------------------+--------------------------------------------------------+
   | perimeter         | Perimeter or boundary length of the nucleus.           |
   +-------------------+--------------------------------------------------------+
   | solidity          | Measure of how solid or filled the nucleus is.         |
   +-------------------+--------------------------------------------------------+
   | sum_intensity     | Sum of intensity values within the nucleus.            |
   +-------------------+--------------------------------------------------------+
   | efc_ratio         | Ratio of eccentricity of the nucleus.                  |
   +-------------------+--------------------------------------------------------+


Table of information for the detected spots
-------------------------------------------

The table below describes the individual columns that contain all the extracted information from the spots using HiTIPS. 


.. table:: Column Descriptions for Spots_Information.csv

   +-------------------------+--------------------------------------------------------+
   | Column Name             | Description                                            |
   +=========================+========================================================+
   | Experiment              | Identifier for the experiment or study.                |
   +-------------------------+--------------------------------------------------------+
   | column                  | The column location of the spot.                       |
   +-------------------------+--------------------------------------------------------+
   | row                     | The row location of the spot.                          |
   +-------------------------+--------------------------------------------------------+
   | time_point              | The time point at which the data was recorded.         |
   +-------------------------+--------------------------------------------------------+
   | field_index             | Index indicating the field or view in the study.       |
   +-------------------------+--------------------------------------------------------+
   | z_slice                 | The slice or depth at which the image is taken.        |
   +-------------------------+--------------------------------------------------------+
   | channel                 | The channel or color of the spot.                      |
   +-------------------------+--------------------------------------------------------+
   | action_index            | Index representing a specific action or event.         |
   +-------------------------+--------------------------------------------------------+
   | cell_index              | Index for identifying individual cells.                |
   +-------------------------+--------------------------------------------------------+
   | x_location              | The X-coordinate of the spot.                          |
   +-------------------------+--------------------------------------------------------+
   | y_location              | The Y-coordinate of the spot.                          |
   +-------------------------+--------------------------------------------------------+
   | z_location              | The Z-coordinate of the spot.                          |
   +-------------------------+--------------------------------------------------------+
   | radial_distance         | Radial distance of the spot.                           |
   +-------------------------+--------------------------------------------------------+
   | area                    | Area of the spot.                                      |
   +-------------------------+--------------------------------------------------------+
   | max_intensity           | Maximum intensity value within the spot.               |
   +-------------------------+--------------------------------------------------------+
   | min_intensity           | Minimum intensity value within the spot.               |
   +-------------------------+--------------------------------------------------------+
   | mean_intensity          | Mean intensity value within the spot.                  |
   +-------------------------+--------------------------------------------------------+
   | perimeter               | Perimeter or boundary length of the spot.              |
   +-------------------------+--------------------------------------------------------+
   | solidity                | Measure of how solid or filled the spot is.            |
   +-------------------------+--------------------------------------------------------+
   | bbox-0                  | Bounding box coordinate 0 of the spot.                 |
   +-------------------------+--------------------------------------------------------+
   | bbox-1                  | Bounding box coordinate 1 of the spot.                 |
   +-------------------------+--------------------------------------------------------+
   | bbox-2                  | Bounding box coordinate 2 of the spot.                 |
   +-------------------------+--------------------------------------------------------+
   | bbox-3                  | Bounding box coordinate 3 of the spot.                 |
   +-------------------------+--------------------------------------------------------+
   | center_of_mass_coords   | Coordinates of the spot's center of mass.              |
   +-------------------------+--------------------------------------------------------+
   | max_intensity_coords    | Coordinates of the maximum intensity within the spot.  |
   +-------------------------+--------------------------------------------------------+
   | integrated_intensity    | Integrated intensity value within the spot.            |
   +-------------------------+--------------------------------------------------------+
   | spot_area_mean          | Mean area of spots.                                    |
   +-------------------------+--------------------------------------------------------+
   | spot_area_std           | Standard deviation of spot areas.                      |
   +-------------------------+--------------------------------------------------------+
   | spot_area_median        | Median area of spots.                                  |
   +-------------------------+--------------------------------------------------------+
   | spot_max_to_min         | Maximum to minimum intensity ratio within the spot.    |
   +-------------------------+--------------------------------------------------------+
   | spot_to_area_mean       | Mean ratio of spot intensity to area.                  |
   +-------------------------+--------------------------------------------------------+
   | spot_max_to_area_mean   | Mean ratio of maximum intensity to area within spots.  |
   +-------------------------+--------------------------------------------------------+

Cell Tracking Tables
--------------------


.. table:: Column Descriptions for Nuclei_Tracking_Information.csv

    +----------------------------------------+----------------------------------------------------------------+
    | Column Name                            | Description                                                    |
    +========================================+================================================================+
    | ID                                     | Identifier for the nucleus.                                    |
    +----------------------------------------+----------------------------------------------------------------+
    | t                                      | Time point of nucleus tracking.                                |
    +----------------------------------------+----------------------------------------------------------------+
    | x                                      | X-coordinate of the nucleus.                                   |
    +----------------------------------------+----------------------------------------------------------------+
    | y                                      | Y-coordinate of the nucleus.                                   |
    +----------------------------------------+----------------------------------------------------------------+
    | z                                      | Z-coordinate of the nucleus.                                   |
    +----------------------------------------+----------------------------------------------------------------+
    | parent                                 | Identifier of the parent nucleus.                              |
    +----------------------------------------+----------------------------------------------------------------+
    | root                                   | Identifier of the root nucleus.                                |
    +----------------------------------------+----------------------------------------------------------------+
    | state                                  | State of the nucleus.                                          |
    +----------------------------------------+----------------------------------------------------------------+
    | generation                             | Generation level of the nucleus.                               |
    +----------------------------------------+----------------------------------------------------------------+
    | eccentricity                           | Eccentricity of the nucleus.                                   |
    +----------------------------------------+----------------------------------------------------------------+
    | bbox-0                                 | Bounding box beginning X coordinate of the nucleus.            |
    +----------------------------------------+----------------------------------------------------------------+
    | bbox-1                                 | Bounding box end X coordinate of the nucleus.                  |
    +----------------------------------------+----------------------------------------------------------------+
    | bbox-2                                 | Bounding box beginning Y coordinate of the nucleus.            |
    +----------------------------------------+----------------------------------------------------------------+
    | bbox-3                                 | Bounding box end Y coordinate of the nucleus.                  |
    +----------------------------------------+----------------------------------------------------------------+
    | solidity                               | Measure of how solid or filled the nucleus is.                 |
    +----------------------------------------+----------------------------------------------------------------+
    | orientation                            | Orientation of the nucleus.                                    |
    +----------------------------------------+----------------------------------------------------------------+
    | perimeter                              | Perimeter or boundary length of the nucleus.                   |
    +----------------------------------------+----------------------------------------------------------------+
    | area                                   | Area of the nucleus.                                           |
    +----------------------------------------+----------------------------------------------------------------+
    | major_axis_length                      | Length of the major axis of the nucleus.                       |
    +----------------------------------------+----------------------------------------------------------------+
    | chXXX_spots_number                     | Number of spots in channel XXX associated with the nucleus.    |
    +----------------------------------------+----------------------------------------------------------------+
    | chXXX_spots_locations                  | Locations of spots in channel XXX associated with the nucleus. |
    +----------------------------------------+----------------------------------------------------------------+
    | chXXX_patch_spots_locations            | Patched locations of spots in channel XXX.                     |
    +----------------------------------------+----------------------------------------------------------------+
    | chXXX_transformed_spots_locations      | Transformed locations of spots in channel XXX.                 |
    +----------------------------------------+----------------------------------------------------------------+
    | chXXX_integrated_intensity             | Integrated intensity in channel XXX.                           |
    +----------------------------------------+----------------------------------------------------------------+
    | chXXX_spot_no_YYY_locations            | Locations of spot no. YYY in channel XXX.                      |
    +----------------------------------------+----------------------------------------------------------------+
    | chXXX_spot_no_YYY_integrated_intensity | Integrated intensity of spot no. YYY in channel XXX.           |
    +----------------------------------------+----------------------------------------------------------------+
    | chXXX_spot_no_YYY_x                    | X-coordinate of spot no. YYY in channel XXX.                   |
    +----------------------------------------+----------------------------------------------------------------+
    | chXXX_spot_no_YYY_y                    | Y-coordinate of spot no. YYY in channel XXX.                   |
    +----------------------------------------+----------------------------------------------------------------+
    | column                                 | Well column location of the nucleus inside the plate.          |
    +----------------------------------------+----------------------------------------------------------------+
    | row                                    | Well row location of the nucleus inside the plate.             |
    +----------------------------------------+----------------------------------------------------------------+
    | field_index                            | Field or view in the well.                                     |
    +----------------------------------------+----------------------------------------------------------------+
    | channel                                | Channel or image the image containing the spot.                |
    +----------------------------------------+----------------------------------------------------------------+
    | chXXX_spot_no_YYY_HMM_state            | Hidden Markov Model (HMM) state of spot no. YYY in channel XXX.|
    +----------------------------------------+----------------------------------------------------------------+


Spot Tracking Tables
--------------------

.. table:: Column Descriptions for Spot_Intensity_Information.csv

    +----------------------------------------+---------------------------------------------------------------+
    | Column Name                            | Description                                                   |
    +========================================+===============================================================+
    | column                                 | Well column location of the spot inside the plate.            |
    +----------------------------------------+---------------------------------------------------------------+
    | row                                    | Well row location of the spot inside the plate.               |
    +----------------------------------------+---------------------------------------------------------------+
    | field_index                            | Field or view in the well.                                    |
    +----------------------------------------+---------------------------------------------------------------+
    | channel                                | Channel or image containing the spot.                         |
    +----------------------------------------+---------------------------------------------------------------+
    | t                                      | Time point of spot tracking.                                  |
    +----------------------------------------+---------------------------------------------------------------+
    | chXXX_spot_no_YYY_x                    | X-coordinate of spot no. YYY in channel XXX.                  |
    +----------------------------------------+---------------------------------------------------------------+
    | chXXX_spot_no_YYY_y                    | Y-coordinate of spot no. YYY in channel XXX.                  |
    +----------------------------------------+---------------------------------------------------------------+
    | chXXX_spot_no_YYY_integrated_intensity | Integrated intensity of spot no. YYY in channel XXX.          |
    +----------------------------------------+---------------------------------------------------------------+
