import os
import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import fcluster, linkage
from skimage.measure import label, regionprops_table
from skimage.transform import warp_polar, rotate, rescale
from skimage.filters import median
from skimage.morphology import disk
from skimage.util import img_as_float
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage.registration import phase_cross_correlation
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from btrack.constants import BayesianUpdates
import btrack  


class Tracking():
    
    """
    This class provides functionalities for tracking biological cells in image stacks using various tracking algorithms.

    Methods:
    - RUN_BTRACK: Performs Bayesian tracking on a stack of labeled images.
    - deepcell_tracking: Uses DeepCell for tracking cells in an image stack.
    - mutual_information: Calculates the mutual information between two images.
    - zero_pad_patch: Pads an image to a desired size with zeros.
    - registration_features: Extracts registration features between reference and registered images.
    - rotation_register_img_prep: Prepares images for rotational registration.
    - run_registration_rotation: Performs image registration considering rotation.
    - phase_correlation: Computes phase correlation between reference and registered images.
    - rotate_point: Rotates a point around a given origin by a specified angle.
    - get_spot_patch: Extracts small patches around detected spots in an image.
    - merge_small_clusters: Merges small clusters based on a distance threshold.
    - run_clustering: Performs clustering on points with options for handling outliers.
    """

    def RUN_BTRACK(label_stack, parms_dict):
        """
        Performs Bayesian tracking on a stack of labeled images.

        Parameters:
        - label_stack (numpy.ndarray): Stack of labeled images to be tracked.
        - parms_dict (object): Parameters from the GUI controlling tracking behavior.

        Returns:
        - tracks_pd (pandas.DataFrame): DataFrame containing the tracking information.

        This method initializes a Bayesian tracker, configures it using the provided parameters, and performs tracking on the input label stack.
        It returns a DataFrame containing the tracked objects and their properties.
        """
        obj_from_generator = btrack.utils.segmentation_to_objects(label_stack, properties = ('bbox','area', 'perimeter', 'major_axis_length','orientation',
                                                                                             'solidity','eccentricity' ))
        # initialise a tracker session using a context manager
        with btrack.BayesianTracker() as tracker:
            tracker = btrack.BayesianTracker()
                # configure the tracker using a config file
                
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # config_file = os.path.join(current_dir, 'BayesianTracker', 'models', 'cell_config.json')
            config_file = os.path.join(current_dir, 'cell_config.json')
            tracker.configure_from_file(config_file)
            if label_stack.shape[0] > 1250:
                tracker.update_method = BayesianUpdates.APPROXIMATE
            # tracker.configure_from_file('/data2/HiTIPS_hmm/HiTIPS/BayesianTracker/models/cell_config.json')
            tracker.max_search_radius = parms_dict["NucSearchRadiusSpinbox_current_value"]

            # append the objects to be tracked
            tracker.append(obj_from_generator)

            # set the volume
        #     tracker.volume=((0, 1200), (0, 1200), (-1e5, 64.))

            # track them (in interactive mode)
            tracker.track_interactive(step_size=100)

            # generate hypotheses and run the global optimizer
            if label_stack.shape[0] < 1250:
                tracker.optimize()

#                             tracker.export(os.path.join('/data2/cell_tracking/','tracking.h5'), obj_type='obj_type_1')

#                             # get the tracks in a format for napari visualization
#                             data, properties, graph = tracker.to_napari(ndim=2)

            tracks = tracker.tracks
        # Create a list to store each DataFrame
        data_frames = []
        for i in range(len(tracks)):
            data_frames.append(pd.DataFrame(tracks[i].to_dict()))
    
        # Concatenate all DataFrames in the list
        tracks_pd = pd.concat(data_frames, ignore_index=True)
        tracks_pd = tracks_pd[tracks_pd['dummy'] == False]

        return tracks_pd
    
    def deepcell_tracking(t_stack_nuc,masks_stack, params_dict):
        """
        Performs cell tracking using the DeepCell tracking algorithm.

        Parameters:
        - t_stack_nuc (numpy.ndarray): Time-lapse stack of nuclear images.
        - masks_stack (numpy.ndarray): Corresponding stack of masks for the nuclei.
        - parms_dict (object): GUI parameters for tracking.

        Returns:
        - tracks_pd (pandas.DataFrame): DataFrame containing tracked cell information over time.

        This method uses DeepCell for tracking cells in a time-lapse image stack. It returns a DataFrame with detailed tracking information.
        """
        tracker = CellTracking()
        tracked_data = tracker.track(np.copy(np.expand_dims(t_stack_nuc, axis=-1)), np.copy(np.expand_dims(masks_stack, axis=-1)))

        tracks_pd = pd.DataFrame()
        for i in range(len(tracked_data['y_tracked'])):
            labeled_nuc, number_nuc = label(np.squeeze(tracked_data['y_tracked'][i]))
            dist_props = regionprops_table(labeled_nuc, properties=('label','centroid', 'bbox','area', 'perimeter', 'major_axis_length','orientation', 'solidity','eccentricity'))
            single_image_df = pd.DataFrame(dist_props)
            single_image_df['t']=i*np.ones(len(single_image_df),dtype=int)
            single_image_df.rename(columns={ "label":"ID"}, inplace = True)
            single_image_df.rename(columns={ "centroid-0":"y"}, inplace = True)
            single_image_df.rename(columns={ "centroid-1":"x"}, inplace = True)
            single_image_df['z']=i*np.zeros(len(single_image_df),dtype=int)
            single_image_df['generation']=i*np.zeros(len(single_image_df),dtype=int)
            single_image_df['parent']=single_image_df['ID']
            single_image_df['root']=single_image_df['ID']
            new_columns=['ID','t','x','y','z','parent','root','generation', 'area', 'major_axis_length', 'solidity', 'bbox-1', 'bbox-2', 'bbox-3', 'eccentricity', 'bbox-0', 'orientation', 'perimeter']
            new_df = single_image_df[new_columns]
            tracks_pd = pd.concat([tracks_pd,new_df], axis=0)
        
        return tracks_pd
    
    def mutual_information( hgram):
        """
        Calculates the mutual information between two images based on their joint histogram.

        Parameters:
        - hgram (numpy.ndarray): Joint histogram of the two images.

        Returns:
        - mi (float): Mutual information value.

        This method computes the mutual information, a measure of the mutual dependence between two images.
        """
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1) # marginal for x over y
        py = np.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    def zero_pad_patch(input_image, desired_size=(128,128)):
        """
        Pads an image with zeros to a desired size.

        Parameters:
        - input_image (numpy.ndarray): The input image to be padded.
        - desired_size (tuple): The desired size of the output image.

        Returns:
        - padded_image (numpy.ndarray): The zero-padded image.

        This method pads an input image with zeros to reach the specified size, centering the original image in the padded area.
        """
        padded_image1 = np.zeros(desired_size, dtype=float)
        w,h= input_image.shape
        if w>desired_size[0]:
            input_image = input_image[:desired_size[0],:]
        if h>desired_size[1]:
            input_image = input_image[:,:desired_size[1]]
        shift_w = int(desired_size[0]/2)
        shift_h = int(desired_size[1]/2)
        padded_image1[:w, :h] = input_image
        padded_image = np.roll(np.roll(padded_image1, int(shift_w-w/2),axis=0), int(shift_h-h/2), axis=1)    

        return padded_image
    
    def registration_features(ref_img, ref_mask, reg_img, reg_mask):
        """
        Extracts various features to evaluate the quality of image registration.

        Parameters:
        - ref_img (numpy.ndarray): The reference image.
        - ref_mask (numpy.ndarray): Mask of the reference image.
        - reg_img (numpy.ndarray): The registered image to compare with the reference.
        - reg_mask (numpy.ndarray): Mask of the registered image.

        Returns:
        - features (numpy.ndarray): Array of registration quality features.

        This method computes features such as cosine similarity, mutual information, SSIM, and others to evaluate registration quality.
        """
        csi1 = np.dot(np.ravel(ref_img),np.ravel(reg_img))/(np.linalg.norm(np.ravel(ref_img))*np.linalg.norm(np.ravel(reg_img)))

        hist_2d_1, x_edges, y_edges = np.histogram2d(reg_img.ravel(), ref_img.ravel(), bins=20)
        mi1 = Tracking.mutual_information(hist_2d_1)

        ssim_ind1 = ssim(ref_img, reg_img, data_range=ref_img.max() - ref_img.min())

        mse = -mean_squared_error(ref_img, reg_img)

        # ct = contingency_table(ref_mask, reg_mask)
        # are = adapted_rand_error(image_true=ref_mask, image_test=reg_mask, table=ct)[0]
        # # vi1, vi2 = variation_of_information(image0=ref_mask, image1=reg_mask , table=ct)

        eps_are=0.000000001

        psnr = peak_signal_noise_ratio(ref_img.astype("uint8"),reg_img.astype("uint8"))

        return np.array([csi1, mi1, ssim_ind1, mse,  psnr])#,1/(are+eps_are), -vi1, -vi2])
    
    def rotation_register_img_prep( ref_img, reg_img, rotation_angle, median_disk_size=3):
        """
        Prepares images for rotational registration.

        Parameters:
        - ref_img (numpy.ndarray): The reference image.
        - reg_img (numpy.ndarray): The image to be registered.
        - rotation_angle (float): The angle for initial rotation.
        - median_disk_size (int): The size of the median filter disk.

        Returns:
        - median_ref_img, median_reg_img (numpy.ndarray): Preprocessed images ready for registration.

        This method prepares images for rotation-based registration by applying median filtering and adjusting for initial rotation.
        """
        median_ref_img = median(ref_img*255/(ref_img.max()+0.0001), disk(median_disk_size))

        im2_rot = rotate(reg_img, rotation_angle, center=None, preserve_range=True)
        img2_rot = median(im2_rot*255/(im2_rot.max()+0.0001) ,disk(median_disk_size))

        im2_rot1 = Tracking.zero_pad_patch(img2_rot, desired_size=patch_size)

        #### correct for translation
        rot2_translation = phase_cross_correlation(median_ref_img, im2_rot1, upsample_factor=1)[0]
        # rot2_translation = np.array([0,0])
        median_reg_img = ndimage.shift(im2_rot1, rot2_translation, order=0)

        return median_ref_img, median_reg_img

    def run_registration_rotation( angle):
        """
        Performs image registration considering rotation.

        Parameters:
        - angle (float): The rotation angle to be applied.

        Returns:
        - registration_features (numpy.ndarray): Features indicative of registration quality.

        This method performs image registration by considering a specific rotation angle and returns features to evaluate the registration.
        """
        img1, im1_rot1 = Tracking.rotation_register_img_prep(rotated_nuc[-1], img_patch1, angle, median_disk_size=3)

        return Tracking.registration_features(img1, im1_rot1)

    def phase_correlation(ref_img, reg_img, intial_rotation = 15, rescale_factor = 5, nuc_length = 50):
        """
        Computes phase correlation between two images to determine the rotational alignment.

        Parameters:
        - ref_img (numpy.ndarray): The reference image.
        - reg_img (numpy.ndarray): The image to be registered.
        - initial_rotation (float): Initial rotation angle applied to the registered image.
        - rescale_factor (int): Factor by which the images are rescaled.
        - nuc_length (int): Length of the nucleus for determining the rescaling radius.

        Returns:
        - final_angle (float): The calculated final rotation angle.
        - rotated_img (numpy.ndarray): The rotated image after alignment.

        This method computes the phase correlation to align two images rotationally and returns the final rotation angle and the aligned image.
        """
        if ref_img.max()>0:    
            
            radius = nuc_length*rescale_factor
            ref_image = img_as_float(rescale(median(ref_img,disk(3)),rescale_factor))
            rotated = rotate(rescale(median(reg_img,disk(3)),rescale_factor), intial_rotation, preserve_range=True)
            image_polar = warp_polar(ref_image, radius=radius)
            rotated_polar = warp_polar(rotated, radius=radius)
            shifts, error, phasediff = phase_cross_correlation(image_polar, rotated_polar)
            final_angle = -shifts[0]
            rotated_img = rotate(rotate(reg_img, intial_rotation, order=0, preserve_range=True), final_angle, order=0, preserve_range=True)

        elif ref_img.max()==0:
            
            final_angle = 0
            rotated_img = rotate(reg_img, 0, order=0, preserve_range=True)
        
        return final_angle, rotated_img
    
    
    
    def rotate_point(origin, point, angle):
        """
        Rotates a point counterclockwise by a given angle around a given origin.

        Parameters:
        - origin (tuple): The origin point for rotation.
        - point (tuple): The point to be rotated.
        - angle (float): The rotation angle in radians.

        Returns:
        - (qx, qy) (tuple): The coordinates of the rotated point.

        This utility method rotates a point around a given origin by the specified angle.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy
    
    def get_spot_patch(single_track_copy, chnl, lbl1, rot_spot_patches, spot_boundary = 4):
        """
        Extracts small patches around detected spots in an image.

        Parameters:
        - single_track_copy (pandas.DataFrame): DataFrame containing spot tracking information.
        - chnl (int): Channel number to be considered.
        - lbl1 (int): Label number for the spot.
        - rot_spot_patches (numpy.ndarray): Array of rotated spot patches.
        - spot_boundary (int): Size of the boundary around the spot for patch extraction.

        Returns:
        - small_spot_patches (dict): Dictionary of extracted spot patches.
        - spot_patches_center_coords (dict): Dictionary of center coordinates for each spot patch.

        This method extracts small patches around detected spots based on the provided tracking information and spot boundaries.
        """
        small_spot_patches = {}
        spot_patches_center_coords = {}
        real_spots = single_track_copy[single_track_copy['ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_locations"].apply(lambda x: x != [])]['ch'+str(int(chnl))+'_spot_no_'+str(lbl1)+"_locations"]
        real_spots_indices = np.array(real_spots.index)
        for i in np.array(single_track_copy.index):
            if i not in real_spots_indices:
                time_point = single_track_copy['t'].loc[i]
                ind1 = np.searchsorted(real_spots_indices, i)
                if ind1==0:
                    try:
                        row,col = real_spots.loc[real_spots_indices[0]][0][0]
                    except:
                        row,col = real_spots.loc[real_spots_indices[0]][0]
                        
                if ind1==len(real_spots_indices):
                    try:
                          row,col = real_spots.loc[real_spots_indices[-1]][0][0]            
                    except:
                          row,col = real_spots.loc[real_spots_indices[-1]][0]
                          
                else:
                    try: 
                          lower_row,lower_col = real_spots.loc[real_spots_indices[ind1-1]][0][0]
                    except:
                          lower_row,lower_col = real_spots.loc[real_spots_indices[ind1-1]][0]
                         
                    try:
                          uper_row,uper_col = real_spots.loc[real_spots_indices[ind1]][0][0]
                    except:
                          uper_row,uper_col = real_spots.loc[real_spots_indices[ind1]][0]
                          
                    row = int(np.round((lower_row+uper_row)/2))
                    col = int(np.round((lower_col+uper_col)/2))

            else:
                 
                try:
                    row,col = real_spots.loc[i][0][0]
                except:
                    row,col = real_spots.loc[i][0]
       
            small_spot_patches[i]=rot_spot_patches[int(i)][int(row-spot_boundary) : int(row+spot_boundary+1), int(col-spot_boundary) : int(col+spot_boundary+1)]
            spot_patches_center_coords[i] = np.array([row,col]).reshape((1,2))

        return small_spot_patches, spot_patches_center_coords
    
    def merge_small_clusters(points, labels, max_dist, min_burst_duration):
        
        """
        Merges small clusters of points based on a distance threshold.

        Parameters:
        - points (numpy.ndarray): Array of point coordinates.
        - labels (numpy.ndarray): Array of cluster labels for each point.
        - max_dist (float): Maximum distance threshold for merging clusters.

        Returns:
        - labels (numpy.ndarray): Updated array of cluster labels after merging.

        This method merges small clusters into larger ones based on proximity, using the specified distance threshold.
        """
        if (len(labels) < 2)|(min_burst_duration==1):
            return labels

        cluster_sizes = np.bincount(labels)

        large_clusters = np.where(cluster_sizes >= min_burst_duration)[0]
        small_clusters = np.where((cluster_sizes > 0) & (cluster_sizes < min_burst_duration))[0]

        large_cluster_points = points[np.isin(labels, large_clusters)]

        for small_cluster in small_clusters:
            small_cluster_indices = np.where(labels == small_cluster)[0]

            for idx in small_cluster_indices:
                distances = cdist(points[idx].reshape(1, -1), large_cluster_points)
                closest_large_cluster_point_index = np.argmin(distances)
                closest_large_cluster_point_distance = distances[0, closest_large_cluster_point_index]

                if closest_large_cluster_point_distance <= max_dist:
                    labels[idx] = labels[large_cluster_points[closest_large_cluster_point_index]]
                else:
                    labels[idx] = 0

        return labels


    def run_clustering(points, outlier_threshold=2, max_dist=6, min_burst_duration=1):
        """
        Performs clustering on a set of points, with options for handling outliers.

        Parameters:
        - points (numpy.ndarray): Array of points to be clustered.
        - outlier_threshold (float): Threshold for determining outliers.
        - max_dist (float): Maximum distance for clustering points.

        Returns:
        - clusters (numpy.ndarray): Array of cluster labels for each point.

        This method performs hierarchical clustering on points, treats outliers based on the specified threshold, and merges small clusters.
        """
        distance_matrix = pdist(points)
        linkage_matrix = linkage(distance_matrix, method='single')
        initial_clusters = fcluster(linkage_matrix, max_dist, criterion='distance')

        unique_clusters = np.unique(initial_clusters)
        centroids = np.array([points[initial_clusters == cluster].mean(axis=0) for cluster in unique_clusters])

        cluster_std_devs = []
        for cluster, centroid in zip(unique_clusters, centroids):
            cluster_points = points[initial_clusters == cluster]
            distances = cdist(centroid.reshape(1, -1), cluster_points)
            cluster_std_devs.append(np.std(distances))

        outlier_index = np.zeros(points.shape[0], dtype=bool)
        for cluster, centroid, std_dev in zip(unique_clusters, centroids, cluster_std_devs):
            cluster_points = points[initial_clusters == cluster]
            distances = cdist(centroid.reshape(1, -1), cluster_points)
            is_outlier = distances > outlier_threshold * max_dist
            outlier_index[initial_clusters == cluster] = is_outlier.flatten()

        initial_clusters[outlier_index] = 0

        # Merge small clusters and outliers
        clusters = Tracking.merge_small_clusters(points, initial_clusters, max_dist, min_burst_duration)


        return clusters