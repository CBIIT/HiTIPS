
import numpy as np
from scipy.stats import norm, expon
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.special import expit as sigmoid
import random
from sklearn.mixture import GaussianMixture
import os
class RNABurstDetector:
    def __init__(self, image_size=100, epsilon=0.01, min_pts=3, log_file="rnaburst_log.txt"):
        """Initialize the RNABurstDetector with parameters and set up logging."""
        self.image_size = image_size
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.sigma_s = None
        self.lambda_decay = None
        self.locus_positions = {}
        self.log_file = log_file
        with open(self.log_file, 'a') as f:
            f.write("Initializing RNABurstDetector...\n")
            f.write(f"image_size: {self.image_size}\n")
            f.write(f"epsilon: {self.epsilon}\n")
            f.write(f"min_pts: {self.min_pts}\n")
            f.write(f"sigma_s: {self.sigma_s}\n")
            f.write(f"lambda_decay: {self.lambda_decay}\n")
            f.write(f"locus_positions: {self.locus_positions}\n")
            f.write(f"log_file: {self.log_file}\n")

    def estimate_parameters(self, spots):
        """Estimate spatial and temporal parameters from spot data."""
        with open(self.log_file, 'a') as f:
            f.write("Estimating parameters...\n")
            f.write(f"spots: {spots}\n")
        distances = cdist(spots[:, :2], spots[:, :2])
        with open(self.log_file, 'a') as f:
            f.write(f"distances: {distances}\n")
        np.fill_diagonal(distances, np.inf)
        with open(self.log_file, 'a') as f:
            f.write(f"distances (diagonal filled): {distances}\n")
        min_distances = np.min(distances, axis=1)
        with open(self.log_file, 'a') as f:
            f.write(f"min_distances: {min_distances}\n")
        self.sigma_s = np.median(min_distances)
        with open(self.log_file, 'a') as f:
            f.write(f"self.sigma_s: {self.sigma_s}\n")
        frame_span = np.max(spots[:, 5]) - np.min(spots[:, 5])
        with open(self.log_file, 'a') as f:
            f.write(f"np.max(spots[:, 5]): {np.max(spots[:, 5])}\n")
            f.write(f"np.min(spots[:, 5]): {np.min(spots[:, 5])}\n")
            f.write(f"frame_span: {frame_span}\n")
        self.lambda_decay = 1 / (frame_span + 1) if frame_span > 0 else 0.1
        with open(self.log_file, 'a') as f:
            f.write(f"self.lambda_decay: {self.lambda_decay}\n")
            f.write(f"Final sigma_s: {self.sigma_s}, lambda_decay: {self.lambda_decay}\n")

    def spot_validation(self, spots):
        """Validate spots based on intensity and spatiotemporal properties."""
        with open(self.log_file, 'a') as f:
            f.write("Validating spots...\n")
            f.write(f"spots: {spots}\n")
        
        I1, I2, I3 = spots[:, 2], spots[:, 3], spots[:, 4]
        with open(self.log_file, 'a') as f:
            f.write(f"I1: {I1}\n")
            f.write(f"I2: {I2}\n")
            f.write(f"I3: {I3}\n")
        midpoint, k = 1.3, 20
        with open(self.log_file, 'a') as f:
            f.write(f"midpoint: {midpoint}\n")
            f.write(f"k: {k}\n")
        P_real_I3 = sigmoid(k * (I3 - midpoint))
        with open(self.log_file, 'a') as f:
            f.write(f"I3 - midpoint: {I3 - midpoint}\n")
            f.write(f"k * (I3 - midpoint): {k * (I3 - midpoint)}\n")
            f.write(f"P_real_I3: {P_real_I3}\n")
        definitely_real = I3 >= 1.5
        definitely_noise = I3 <= 1.1
        with open(self.log_file, 'a') as f:
            f.write(f"definitely_real: {definitely_real}\n")
            f.write(f"definitely_noise: {definitely_noise}\n")
        P_real_final = P_real_I3.copy()
        with open(self.log_file, 'a') as f:
            f.write(f"P_real_final (before adjustment): {P_real_final}\n")
        P_real_final[definitely_real] = 1.0
        P_real_final[definitely_noise] = 0.0
        with open(self.log_file, 'a') as f:
            f.write(f"P_real_final (after adjustment): {P_real_final}\n")

        mu_I1, sigma_I1 = np.mean(I1), np.std(I1)
        mu_I2, sigma_I2 = np.mean(I2), np.std(I2)
        with open(self.log_file, 'a') as f:
            f.write(f"np.mean(I1): {np.mean(I1)}\n")
            f.write(f"np.std(I1): {np.std(I1)}\n")
            f.write(f"mu_I1: {mu_I1}\n")
            f.write(f"sigma_I1: {sigma_I1}\n")
            f.write(f"np.mean(I2): {np.mean(I2)}\n")
            f.write(f"np.std(I2): {np.std(I2)}\n")
            f.write(f"mu_I2: {mu_I2}\n")
            f.write(f"sigma_I2: {sigma_I2}\n")
        z_I1 = (I1 - mu_I1) / sigma_I1
        z_I2 = (I2 - mu_I2) / sigma_I2
        with open(self.log_file, 'a') as f:
            f.write(f"I1 - mu_I1: {I1 - mu_I1}\n")
            f.write(f"z_I1: {z_I1}\n")
            f.write(f"I2 - mu_I2: {I2 - mu_I2}\n")
            f.write(f"z_I2: {z_I2}\n")
        z_combined = (z_I1 + z_I2) / 2
        with open(self.log_file, 'a') as f:
            f.write(f"z_I1 + z_I2: {z_I1 + z_I2}\n")
            f.write(f"z_combined: {z_combined}\n")
        scaling_factor = 0.5 + 0.5 * sigmoid(z_combined)
        with open(self.log_file, 'a') as f:
            f.write(f"sigmoid(z_combined): {sigmoid(z_combined)}\n")
            f.write(f"0.5 * sigmoid(z_combined): {0.5 * sigmoid(z_combined)}\n")
            f.write(f"scaling_factor: {scaling_factor}\n")
        combined_probability_intensity = P_real_final * scaling_factor
        with open(self.log_file, 'a') as f:
            f.write(f"combined_probability_intensity: {combined_probability_intensity}\n")

        coords = spots[:, :2]
        times = spots[:, 5]
        with open(self.log_file, 'a') as f:
            f.write(f"coords: {coords}\n")
            f.write(f"times: {times}\n")
        spatial_dist = cdist(coords, coords)
        with open(self.log_file, 'a') as f:
            f.write(f"spatial_dist: {spatial_dist}\n")
        time_dist = np.abs(times[:, None] - times[None, :])
        with open(self.log_file, 'a') as f:
            f.write(f"times[:, None]: {times[:, None]}\n")
            f.write(f"times[None, :]: {times[None, :]}\n")
            f.write(f"times[:, None] - times[None, :]: {times[:, None] - times[None, :]}\n")
            f.write(f"time_dist: {time_dist}\n")
        delta = self.sigma_s * 5
        tau = 10
        with open(self.log_file, 'a') as f:
            f.write(f"self.sigma_s: {self.sigma_s}\n")
            f.write(f"delta: {delta}\n")
            f.write(f"tau: {tau}\n")
        density = np.sum((spatial_dist < delta) & (time_dist <= tau), axis=1)
        with open(self.log_file, 'a') as f:
            f.write(f"spatial_dist < delta: {spatial_dist < delta}\n")
            f.write(f"time_dist <= tau: {time_dist <= tau}\n")
            f.write(f"(spatial_dist < delta) & (time_dist <= tau): {(spatial_dist < delta) & (time_dist <= tau)}\n")
            f.write(f"density: {density}\n")
        p_xyt_z1_base = density / len(spots)
        with open(self.log_file, 'a') as f:
            f.write(f"len(spots): {len(spots)}\n")
            f.write(f"p_xyt_z1_base: {p_xyt_z1_base}\n")
        spatial_density = np.sum(spatial_dist < delta, axis=1) / len(spots)
        with open(self.log_file, 'a') as f:
            f.write(f"np.sum(spatial_dist < delta, axis=1): {np.sum(spatial_dist < delta, axis=1)}\n")
            f.write(f"spatial_density: {spatial_density}\n")
        time_dist_no_self = time_dist.copy()
        with open(self.log_file, 'a') as f:
            f.write(f"time_dist_no_self (before diagonal): {time_dist_no_self}\n")
        np.fill_diagonal(time_dist_no_self, np.inf)
        with open(self.log_file, 'a') as f:
            f.write(f"time_dist_no_self (after diagonal): {time_dist_no_self}\n")
        min_time_dist = np.min(time_dist_no_self, axis=1)
        with open(self.log_file, 'a') as f:
            f.write(f"min_time_dist: {min_time_dist}\n")
        w = np.ones_like(min_time_dist, dtype=float)
        with open(self.log_file, 'a') as f:
            f.write(f"w (initial): {w}\n")
        mask1 = min_time_dist <= 3
        mask2 = (min_time_dist > 3) & (min_time_dist <= 10)
        with open(self.log_file, 'a') as f:
            f.write(f"min_time_dist <= 3: {min_time_dist <= 3}\n")
            f.write(f"min_time_dist > 3: {min_time_dist > 3}\n")
            f.write(f"min_time_dist <= 10: {min_time_dist <= 10}\n")
            f.write(f"mask1: {mask1}\n")
            f.write(f"mask2: {mask2}\n")
        w[mask1] = 5
        w[mask2] = 5 - (min_time_dist[mask2] - 3) * (4 / 7)
        with open(self.log_file, 'a') as f:
            f.write(f"w[mask1] assigned: {w[mask1]}\n")
            f.write(f"min_time_dist[mask2]: {min_time_dist[mask2]}\n")
            f.write(f"min_time_dist[mask2] - 3: {min_time_dist[mask2] - 3}\n")
            f.write(f"(min_time_dist[mask2] - 3) * (4 / 7): {(min_time_dist[mask2] - 3) * (4 / 7)}\n")
            f.write(f"5 - (min_time_dist[mask2] - 3) * (4 / 7): {5 - (min_time_dist[mask2] - 3) * (4 / 7)}\n")
            f.write(f"w (adjusted): {w}\n")
        p_xyt_z1 = p_xyt_z1_base * w
        with open(self.log_file, 'a') as f:
            f.write(f"p_xyt_z1: {p_xyt_z1}\n")
        isolated_in_time = min_time_dist > tau
        with open(self.log_file, 'a') as f:
            f.write(f"min_time_dist > tau: {min_time_dist > tau}\n")
            f.write(f"isolated_in_time: {isolated_in_time}\n")
        p_xyt_z1[isolated_in_time] = spatial_density[isolated_in_time]
        with open(self.log_file, 'a') as f:
            f.write(f"spatial_density[isolated_in_time]: {spatial_density[isolated_in_time]}\n")
            f.write(f"p_xyt_z1 (after isolation adjustment): {p_xyt_z1}\n")
        total_area = self.image_size ** 2
        total_frames = np.max(times) - np.min(times) + 1
        with open(self.log_file, 'a') as f:
            f.write(f"self.image_size: {self.image_size}\n")
            f.write(f"self.image_size ** 2: {self.image_size ** 2}\n")
            f.write(f"total_area: {total_area}\n")
            f.write(f"np.max(times): {np.max(times)}\n")
            f.write(f"np.min(times): {np.min(times)}\n")
            f.write(f"np.max(times) - np.min(times): {np.max(times) - np.min(times)}\n")
            f.write(f"total_frames: {total_frames}\n")
        np.fill_diagonal(spatial_dist, np.inf)
        with open(self.log_file, 'a') as f:
            f.write(f"spatial_dist (diagonal filled): {spatial_dist}\n")
        min_distances = np.min(spatial_dist, axis=1)
        mu_min_dist = np.mean(min_distances)
        sigma_min_dist = np.std(min_distances)
        with open(self.log_file, 'a') as f:
            f.write(f"min_distances: {min_distances}\n")
            f.write(f"np.mean(min_distances): {np.mean(min_distances)}\n")
            f.write(f"mu_min_dist: {mu_min_dist}\n")
            f.write(f"np.std(min_distances): {np.std(min_distances)}\n")
            f.write(f"sigma_min_dist: {sigma_min_dist}\n")
        base_p_xyt_z0 = 1 / (total_area * total_frames)
        with open(self.log_file, 'a') as f:
            f.write(f"total_area * total_frames: {total_area * total_frames}\n")
            f.write(f"base_p_xyt_z0: {base_p_xyt_z0}\n")
        k = 8.293 / sigma_min_dist
        d0 = mu_min_dist + 0.167 * sigma_min_dist
        with open(self.log_file, 'a') as f:
            f.write(f"8.293 / sigma_min_dist: {8.293 / sigma_min_dist}\n")
            f.write(f"k: {k}\n")
            f.write(f"0.167 * sigma_min_dist: {0.167 * sigma_min_dist}\n")
            f.write(f"mu_min_dist + 0.167 * sigma_min_dist: {mu_min_dist + 0.167 * sigma_min_dist}\n")
            f.write(f"d0: {d0}\n")
        p_xyt_z0 = base_p_xyt_z0 + (1 - base_p_xyt_z0) * sigmoid(k * (min_distances - d0))
        with open(self.log_file, 'a') as f:
            f.write(f"min_distances - d0: {min_distances - d0}\n")
            f.write(f"k * (min_distances - d0): {k * (min_distances - d0)}\n")
            f.write(f"sigmoid(k * (min_distances - d0)): {sigmoid(k * (min_distances - d0))}\n")
            f.write(f"1 - base_p_xyt_z0: {1 - base_p_xyt_z0}\n")
            f.write(f"(1 - base_p_xyt_z0) * sigmoid(k * (min_distances - d0)): {(1 - base_p_xyt_z0) * sigmoid(k * (min_distances - d0))}\n")
            f.write(f"p_xyt_z0 (initial): {p_xyt_z0}\n")
        threshold = mu_min_dist + sigma_min_dist
        with open(self.log_file, 'a') as f:
            f.write(f"mu_min_dist + sigma_min_dist: {mu_min_dist + sigma_min_dist}\n")
            f.write(f"threshold: {threshold}\n")
        p_xyt_z0[min_distances >= threshold] = 1.0
        with open(self.log_file, 'a') as f:
            f.write(f"min_distances >= threshold: {min_distances >= threshold}\n")
            f.write(f"p_xyt_z0: {p_xyt_z0}\n")

        p_s_z1 = combined_probability_intensity * p_xyt_z1
        with open(self.log_file, 'a') as f:
            f.write(f"p_s_z1: {p_s_z1}\n")
        p_s_z0 = (1.0 - combined_probability_intensity) * p_xyt_z0
        with open(self.log_file, 'a') as f:
            f.write(f"1.0 - combined_probability_intensity: {1.0 - combined_probability_intensity}\n")
            f.write(f"p_s_z0: {p_s_z0}\n")
        p_z1, p_z0 = 0.8, 0.2
        with open(self.log_file, 'a') as f:
            f.write(f"p_z1: {p_z1}\n")
            f.write(f"p_z0: {p_z0}\n")
        numerator = p_s_z1 * p_z1
        with open(self.log_file, 'a') as f:
            f.write(f"numerator: {numerator}\n")
        denominator = numerator + p_s_z0 * p_z0 + 1e-12
        with open(self.log_file, 'a') as f:
            f.write(f"p_s_z0 * p_z0: {p_s_z0 * p_z0}\n")
            f.write(f"p_s_z0 * p_z0 + 1e-12: {p_s_z0 * p_z0 + 1e-12}\n")
            f.write(f"denominator: {denominator}\n")
        p_z1_s = numerator / denominator
        with open(self.log_file, 'a') as f:
            f.write(f"p_z1_s: {p_z1_s}\n")
        real_mask = p_z1_s > 0.5
        with open(self.log_file, 'a') as f:
            f.write(f"p_z1_s > 0.5: {p_z1_s > 0.5}\n")
            f.write(f"real_mask: {real_mask}\n")
            f.write(f"Number of real spots: {np.sum(real_mask)}\n")
        return real_mask

    def initialize_loci(self, spots, real_mask):
        """Initialize loci by clustering real spots using DBSCAN."""
        with open(self.log_file, 'a') as f:
            f.write("Initializing loci...\n")
            f.write(f"spots: {spots}\n")
            f.write(f"real_mask: {real_mask}\n")
        real_spots = spots[real_mask]
        with open(self.log_file, 'a') as f:
            f.write(f"real_spots: {real_spots}\n")
        assignments = np.full(len(spots), -1, dtype=int)
        with open(self.log_file, 'a') as f:
            f.write(f"len(spots): {len(spots)}\n")
            f.write(f"assignments (initial): {assignments}\n")

        if len(real_spots) < self.min_pts:
            with open(self.log_file, 'a') as f:
                f.write(f"len(real_spots): {len(real_spots)}\n")
                f.write(f"self.min_pts: {self.min_pts}\n")
                f.write("Too few real spots for clustering. Assigning all real spots as noise.\n")
            assignments[real_mask] = -1
            with open(self.log_file, 'a') as f:
                f.write(f"assignments (after few spots): {assignments}\n")
            return assignments

        coords_real = real_spots[:, :2]
        times_real = real_spots[:, 5]
        with open(self.log_file, 'a') as f:
            f.write(f"coords_real: {coords_real}\n")
            f.write(f"times_real: {times_real}\n")
        time_dist_real = np.abs(times_real[:, None] - times_real[None, :])
        with open(self.log_file, 'a') as f:
            f.write(f"times_real[:, None]: {times_real[:, None]}\n")
            f.write(f"times_real[None, :]: {times_real[None, :]}\n")
            f.write(f"time_dist_real: {time_dist_real}\n")
        np.fill_diagonal(time_dist_real, np.inf)
        with open(self.log_file, 'a') as f:
            f.write(f"time_dist_real (diagonal filled): {time_dist_real}\n")
        min_time_dist_real = np.min(time_dist_real, axis=1)
        with open(self.log_file, 'a') as f:
            f.write(f"min_time_dist_real: {min_time_dist_real}\n")
        tau = 5
        with open(self.log_file, 'a') as f:
            f.write(f"tau: {tau}\n")
        non_isolated_mask = min_time_dist_real <= tau
        isolated_mask = min_time_dist_real > tau
        with open(self.log_file, 'a') as f:
            f.write(f"min_time_dist_real <= tau: {min_time_dist_real <= tau}\n")
            f.write(f"non_isolated_mask: {non_isolated_mask}\n")
            f.write(f"min_time_dist_real > tau: {min_time_dist_real > tau}\n")
            f.write(f"isolated_mask: {isolated_mask}\n")
        alpha = self.sigma_s / 50
        with open(self.log_file, 'a') as f:
            f.write(f"self.sigma_s: {self.sigma_s}\n")
            f.write(f"self.sigma_s / 50: {self.sigma_s / 50}\n")
            f.write(f"alpha: {alpha}\n")
        labels = -1 * np.ones(len(real_spots), dtype=int)
        with open(self.log_file, 'a') as f:
            f.write(f"len(real_spots): {len(real_spots)}\n")
            f.write(f"labels (initial): {labels}\n")

        if np.any(non_isolated_mask):
            feature_non_isolated = np.hstack((coords_real[non_isolated_mask],
                                             (times_real[non_isolated_mask] * alpha).reshape(-1, 1)))
            with open(self.log_file, 'a') as f:
                f.write(f"coords_real[non_isolated_mask]: {coords_real[non_isolated_mask]}\n")
                f.write(f"times_real[non_isolated_mask]: {times_real[non_isolated_mask]}\n")
                f.write(f"times_real[non_isolated_mask] * alpha: {times_real[non_isolated_mask] * alpha}\n")
                f.write(f"(times_real[non_isolated_mask] * alpha).reshape(-1, 1): {(times_real[non_isolated_mask] * alpha).reshape(-1, 1)}\n")
                f.write(f"feature_non_isolated: {feature_non_isolated}\n")
            db_non_isolated = DBSCAN(eps=self.sigma_s * 5, min_samples=self.min_pts).fit(feature_non_isolated)
            with open(self.log_file, 'a') as f:
                f.write(f"self.sigma_s * 5: {self.sigma_s * 5}\n")
                f.write(f"self.min_pts: {self.min_pts}\n")
                f.write(f"db_non_isolated.labels_: {db_non_isolated.labels_}\n")
            labels[non_isolated_mask] = db_non_isolated.labels_
            with open(self.log_file, 'a') as f:
                f.write(f"labels (after non-isolated): {labels}\n")
                f.write(f"Non-isolated DBSCAN labels: {np.unique(db_non_isolated.labels_, return_counts=True)}\n")

        if np.any(isolated_mask):
            feature_isolated = coords_real[isolated_mask]
            with open(self.log_file, 'a') as f:
                f.write(f"feature_isolated: {feature_isolated}\n")
            db_isolated = DBSCAN(eps=self.sigma_s * 5, min_samples=self.min_pts).fit(feature_isolated)
            with open(self.log_file, 'a') as f:
                f.write(f"self.sigma_s * 5: {self.sigma_s * 5}\n")
                f.write(f"self.min_pts: {self.min_pts}\n")
                f.write(f"db_isolated.labels_: {db_isolated.labels_}\n")
            labels_isolated = db_isolated.labels_
            with open(self.log_file, 'a') as f:
                f.write(f"labels_isolated: {labels_isolated}\n")
            current_max = np.max(labels) if np.any(labels >= 0) else -1
            with open(self.log_file, 'a') as f:
                f.write(f"np.any(labels >= 0): {np.any(labels >= 0)}\n")
                f.write(f"np.max(labels): {np.max(labels) if np.any(labels >= 0) else 'N/A'}\n")
                f.write(f"current_max: {current_max}\n")
            for i, l in enumerate(labels_isolated):
                if l >= 0:
                    labels[isolated_mask][i] = l + current_max + 1 if current_max >= 0 else l
                    with open(self.log_file, 'a') as f:
                        f.write(f"i: {i}, l: {l}\n")
                        f.write(f"l >= 0: {l >= 0}\n")
                        f.write(f"current_max >= 0: {current_max >= 0}\n")
                        f.write(f"l + current_max + 1: {l + current_max + 1 if current_max >= 0 else l}\n")
                        f.write(f"labels[isolated_mask][{i}]: {labels[isolated_mask][i]}\n")
            with open(self.log_file, 'a') as f:
                f.write(f"labels (after isolated): {labels}\n")
                f.write(f"Isolated DBSCAN labels (before offset): {np.unique(labels_isolated, return_counts=True)}\n")

        valid_labels = labels[labels >= 0]
        with open(self.log_file, 'a') as f:
            f.write(f"labels >= 0: {labels >= 0}\n")
            f.write(f"valid_labels: {valid_labels}\n")
        if len(valid_labels) == 0:
            with open(self.log_file, 'a') as f:
                f.write(f"len(valid_labels): {len(valid_labels)}\n")
                f.write("No valid clusters found. Assigning all real spots as noise.\n")
            assignments[real_mask] = -1
            with open(self.log_file, 'a') as f:
                f.write(f"assignments (no clusters): {assignments}\n")
            return assignments

        unique_labels = np.unique(valid_labels)
        with open(self.log_file, 'a') as f:
            f.write(f"unique_labels: {unique_labels}\n")
        label_map = {old: new for new, old in enumerate(unique_labels)}
        with open(self.log_file, 'a') as f:
            f.write(f"enumerate(unique_labels): {list(enumerate(unique_labels))}\n")
            f.write(f"label_map: {label_map}\n")
        assignments[real_mask] = [label_map.get(l, -1) if l >= 0 else -1 for l in labels]
        with open(self.log_file, 'a') as f:
            f.write(f"[label_map.get(l, -1) if l >= 0 else -1 for l in labels]: {[label_map.get(l, -1) if l >= 0 else -1 for l in labels]}\n")
            f.write(f"assignments (final): {assignments}\n")
            f.write(f"Final unique locus labels: {np.unique(assignments[real_mask], return_counts=True)}\n")
        return assignments

    def bayesian_inference(self, real_spots, prev_assignments_real, initial_locus_positions):
        """Refine spot assignments using Bayesian inference."""
        with open(self.log_file, 'a') as f:
            f.write("\n--- Starting Bayesian Inference ---\n")
            f.write(f"real_spots shape: {real_spots.shape}\n")
            f.write(f"real_spots: {real_spots}\n")
            f.write(f"prev_assignments_real: {prev_assignments_real}\n")
            f.write(f"initial_locus_positions: {initial_locus_positions}\n")

        frames = np.unique(real_spots[:, 5])
        with open(self.log_file, 'a') as f:
            f.write(f"real_spots[:, 5]: {real_spots[:, 5]}\n")
            f.write(f"frames: {frames}\n")

        assignments = prev_assignments_real.copy()
        with open(self.log_file, 'a') as f:
            f.write(f"assignments (initial): {assignments}\n")

        sigma_s_adjusted = self.sigma_s
        temporal_threshold = 10
        with open(self.log_file, 'a') as f:
            f.write(f"self.sigma_s: {self.sigma_s}\n")
            f.write(f"sigma_s_adjusted: {sigma_s_adjusted}\n")
            f.write(f"temporal_threshold: {temporal_threshold}\n")

        self.locus_positions = initial_locus_positions.copy()
        with open(self.log_file, 'a') as f:
            f.write(f"initial_locus_positions.copy(): {initial_locus_positions.copy()}\n")
            f.write(f"self.locus_positions: {self.locus_positions}\n")

        probability_threshold = 1e-5
        new_locus_threshold = 15 * sigma_s_adjusted
        raw_prob_max_threshold = 1e-20
        with open(self.log_file, 'a') as f:
            f.write(f"probability_threshold: {probability_threshold}\n")
            f.write(f"15 * sigma_s_adjusted: {15 * sigma_s_adjusted}\n")
            f.write(f"new_locus_threshold: {new_locus_threshold}\n")
            f.write(f"raw_prob_max_threshold: {raw_prob_max_threshold}\n")

        for t in frames:
            with open(self.log_file, 'a') as f:
                f.write(f"\nProcessing frame: {t}\n")
            frame_mask = real_spots[:, 5] == t
            with open(self.log_file, 'a') as f:
                f.write(f"real_spots[:, 5]: {real_spots[:, 5]}\n")
                f.write(f"real_spots[:, 5] == t: {real_spots[:, 5] == t}\n")
                f.write(f"frame_mask: {frame_mask}\n")
            frame_spots = real_spots[frame_mask]
            with open(self.log_file, 'a') as f:
                f.write(f"frame_spots: {frame_spots}\n")
            n_spots = len(frame_spots)
            with open(self.log_file, 'a') as f:
                f.write(f"n_spots: {n_spots}\n")
            n_loci = len(self.locus_positions)
            with open(self.log_file, 'a') as f:
                f.write(f"n_loci: {n_loci}\n")
            p_s_l = np.zeros((n_spots, n_loci))
            with open(self.log_file, 'a') as f:
                f.write(f"p_s_l (initial): {p_s_l}\n")
            P_real_I3 = sigmoid(20 * (frame_spots[:, 4] - 1.3))
            with open(self.log_file, 'a') as f:
                f.write(f"frame_spots[:, 4]: {frame_spots[:, 4]}\n")
                f.write(f"frame_spots[:, 4] - 1.3: {frame_spots[:, 4] - 1.3}\n")
                f.write(f"20 * (frame_spots[:, 4] - 1.3): {20 * (frame_spots[:, 4] - 1.3)}\n")
                f.write(f"P_real_I3: {P_real_I3}\n")
            frame_indices = np.where(frame_mask)[0]
            with open(self.log_file, 'a') as f:
                f.write(f"np.where(frame_mask): {np.where(frame_mask)}\n")
                f.write(f"frame_indices: {frame_indices}\n")

            for i, s in enumerate(frame_spots):
                with open(self.log_file, 'a') as f:
                    f.write(f"\nSpot {i} in frame {t}: {s}\n")
                p_I_z1 = P_real_I3[i]
                with open(self.log_file, 'a') as f:
                    f.write(f"P_real_I3[{i}]: {P_real_I3[i]}\n")
                    f.write(f"p_I_z1: {p_I_z1}\n")
                spot_prev_assignment = prev_assignments_real[frame_mask][i]
                with open(self.log_file, 'a') as f:
                    f.write(f"prev_assignments_real[frame_mask]: {prev_assignments_real[frame_mask]}\n")
                    f.write(f"spot_prev_assignment: {spot_prev_assignment}\n")

                min_dt_to_loci = []
                for l in range(n_loci):
                    locus_frames = real_spots[assignments == l, 5]
                    with open(self.log_file, 'a') as f:
                        f.write(f"l: {l}\n")
                        f.write(f"assignments == l: {assignments == l}\n")
                        f.write(f"real_spots[assignments == l, 5]: {locus_frames}\n")
                    if len(locus_frames) > 0:
                        min_dt = np.min(np.abs(t - locus_frames))
                        with open(self.log_file, 'a') as f:
                            f.write(f"t - locus_frames: {t - locus_frames}\n")
                            f.write(f"np.abs(t - locus_frames): {np.abs(t - locus_frames)}\n")
                            f.write(f"min_dt: {min_dt}\n")
                        min_dt_to_loci.append(min_dt)
                    else:
                        min_dt_to_loci.append(np.inf)
                    with open(self.log_file, 'a') as f:
                        f.write(f"len(locus_frames): {len(locus_frames)}\n")
                        f.write(f"min_dt_to_loci (current): {min_dt_to_loci}\n")

                is_temporally_isolated = all(dt > temporal_threshold for dt in min_dt_to_loci)
                with open(self.log_file, 'a') as f:
                    f.write(f"min_dt_to_loci: {min_dt_to_loci}\n")
                    f.write(f"[dt > temporal_threshold for dt in min_dt_to_loci]: {[dt > temporal_threshold for dt in min_dt_to_loci]}\n")
                    f.write(f"is_temporally_isolated: {is_temporally_isolated}\n")

                for l in range(n_loci):
                    mean_xy = self.locus_positions[l]
                    with open(self.log_file, 'a') as f:
                        f.write(f"l: {l}\n")
                        f.write(f"self.locus_positions[{l}]: {mean_xy}\n")
                    dist = cdist([s[:2]], [mean_xy])[0][0]
                    with open(self.log_file, 'a') as f:
                        f.write(f"s[:2]: {s[:2]}\n")
                        f.write(f"[s[:2]]: {[s[:2]]}\n")
                        f.write(f"[mean_xy]: {[mean_xy]}\n")
                        f.write(f"cdist([s[:2]], [mean_xy]): {cdist([s[:2]], [mean_xy])}\n")
                        f.write(f"dist: {dist}\n")
                    p_xy_l = norm.pdf(dist, scale=sigma_s_adjusted)
                    with open(self.log_file, 'a') as f:
                        f.write(f"norm.pdf(dist, scale=sigma_s_adjusted): {p_xy_l}\n")
                        f.write(f"p_xy_l: {p_xy_l}\n")
                    if len(min_dt_to_loci) > 0 and min_dt_to_loci[l] != np.inf:
                        min_dt = min_dt_to_loci[l]
                        with open(self.log_file, 'a') as f:
                            f.write(f"min_dt_to_loci[{l}]: {min_dt_to_loci[l]}\n")
                            f.write(f"min_dt: {min_dt}\n")
                        if is_temporally_isolated:
                            p_trans = expon.pdf(min_dt, scale=1 / self.lambda_decay)
                            with open(self.log_file, 'a') as f:
                                f.write(f"1 / self.lambda_decay: {1 / self.lambda_decay}\n")
                                f.write(f"expon.pdf(min_dt, scale=1 / self.lambda_decay): {p_trans}\n")
                        else:
                            p_trans = 1e-10 if min_dt > temporal_threshold else expon.pdf(min_dt, scale=1 / self.lambda_decay)
                            with open(self.log_file, 'a') as f:
                                f.write(f"min_dt > temporal_threshold: {min_dt > temporal_threshold}\n")
                                f.write(f"p_trans: {p_trans}\n")
                    else:
                        p_trans = 1.0
                        with open(self.log_file, 'a') as f:
                            f.write(f"len(min_dt_to_loci): {len(min_dt_to_loci)}\n")
                            f.write(f"min_dt_to_loci[{l}] != np.inf: {min_dt_to_loci[l] != np.inf if len(min_dt_to_loci) > 0 else 'N/A'}\n")
                            f.write(f"p_trans: {p_trans}\n")
                    p_s_l[i, l] = p_I_z1 * p_xy_l * p_trans
                    with open(self.log_file, 'a') as f:
                        f.write(f"p_I_z1 * p_xy_l: {p_I_z1 * p_xy_l}\n")
                        f.write(f"p_s_l[{i}, {l}]: {p_s_l[i, l]}\n")

                if n_loci > 0:
                    min_dist = min(cdist([s[:2]], [self.locus_positions[l] for l in range(n_loci)]).flatten())
                    with open(self.log_file, 'a') as f:
                        f.write(f"[self.locus_positions[l] for l in range(n_loci)]: {[self.locus_positions[l] for l in range(n_loci)]}\n")
                        f.write(f"cdist([s[:2]], ...): {cdist([s[:2]], [self.locus_positions[l] for l in range(n_loci)])}\n")
                        f.write(f"cdist(...).flatten(): {cdist([s[:2]], [self.locus_positions[l] for l in range(n_loci)]).flatten()}\n")
                        f.write(f"min_dist: {min_dist}\n")
                    prob_sum = np.sum(p_s_l[i, :])
                    with open(self.log_file, 'a') as f:
                        f.write(f"p_s_l[{i}, :]: {p_s_l[i, :]}\n")
                        f.write(f"prob_sum: {prob_sum}\n")
                    max_raw_prob = np.max(p_s_l[i, :]) if prob_sum > 0 else 0
                    with open(self.log_file, 'a') as f:
                        f.write(f"np.max(p_s_l[{i}, :]): {np.max(p_s_l[i, :]) if prob_sum > 0 else 'N/A'}\n")
                        f.write(f"max_raw_prob: {max_raw_prob}\n")
                    if prob_sum > 0:
                        p_s_l_normalized = p_s_l[i, :] / prob_sum
                        with open(self.log_file, 'a') as f:
                            f.write(f"p_s_l[{i}, :] / prob_sum: {p_s_l[i, :] / prob_sum}\n")
                            f.write(f"p_s_l_normalized: {p_s_l_normalized}\n")
                        max_prob = np.max(p_s_l_normalized)
                        with open(self.log_file, 'a') as f:
                            f.write(f"np.max(p_s_l_normalized): {max_prob}\n")
                            f.write(f"max_prob: {max_prob}\n")
                    else:
                        p_s_l_normalized = np.zeros_like(p_s_l[i, :])
                        max_prob = 0
                        with open(self.log_file, 'a') as f:
                            f.write(f"p_s_l_normalized (zero prob_sum): {p_s_l_normalized}\n")
                            f.write(f"max_prob: {max_prob}\n")
                else:
                    min_dist = np.inf
                    prob_sum = 0
                    max_prob = 0
                    max_raw_prob = 0
                    p_s_l_normalized = np.zeros((n_spots,))
                    with open(self.log_file, 'a') as f:
                        f.write(f"n_loci == 0, setting min_dist: {min_dist}\n")
                        f.write(f"prob_sum: {prob_sum}\n")
                        f.write(f"max_prob: {max_prob}\n")
                        f.write(f"max_raw_prob: {max_raw_prob}\n")
                        f.write(f"p_s_l_normalized: {p_s_l_normalized}\n")

                if n_loci == 0:
                    new_locus_id = 0
                    self.locus_positions[new_locus_id] = s[:2]
                    assignments[frame_indices[i]] = new_locus_id
                    with open(self.log_file, 'a') as f:
                        f.write(f"n_loci: {n_loci}\n")
                        f.write(f"new_locus_id: {new_locus_id}\n")
                        f.write(f"s[:2]: {s[:2]}\n")
                        f.write(f"self.locus_positions[{new_locus_id}]: {self.locus_positions[new_locus_id]}\n")
                        f.write(f"frame_indices[{i}]: {frame_indices[i]}\n")
                        f.write(f"assignments[{frame_indices[i]}]: {assignments[frame_indices[i]]}\n")
                        f.write(f"Created first locus {new_locus_id} for spot {i}\n")
                elif is_temporally_isolated and min_dist > new_locus_threshold:
                    new_locus_id = max(self.locus_positions.keys()) + 1 if self.locus_positions else 0
                    self.locus_positions[new_locus_id] = s[:2]
                    assignments[frame_indices[i]] = new_locus_id
                    with open(self.log_file, 'a') as f:
                        f.write(f"min_dist > new_locus_threshold: {min_dist > new_locus_threshold}\n")
                        f.write(f"self.locus_positions.keys(): {self.locus_positions.keys()}\n")
                        f.write(f"max(self.locus_positions.keys()): {max(self.locus_positions.keys()) if self.locus_positions else 'N/A'}\n")
                        f.write(f"new_locus_id: {new_locus_id}\n")
                        f.write(f"self.locus_positions[{new_locus_id}]: {self.locus_positions[new_locus_id]}\n")
                        f.write(f"assignments[{frame_indices[i]}]: {assignments[frame_indices[i]]}\n")
                        f.write(f"Created new locus {new_locus_id} for spot {i} in frame {t}\n")
                elif n_loci > 0 and max_prob >= probability_threshold and max_raw_prob >= raw_prob_max_threshold:
                    cost = -np.log(p_s_l_normalized + 1e-10)
                    with open(self.log_file, 'a') as f:
                        f.write(f"max_prob >= probability_threshold: {max_prob >= probability_threshold}\n")
                        f.write(f"max_raw_prob >= raw_prob_max_threshold: {max_raw_prob >= raw_prob_max_threshold}\n")
                        f.write(f"p_s_l_normalized + 1e-10: {p_s_l_normalized + 1e-10}\n")
                        f.write(f"-np.log(p_s_l_normalized + 1e-10): {cost}\n")
                        f.write(f"cost: {cost}\n")
                    row_ind, col_ind = linear_sum_assignment(cost.reshape(1, -1))
                    with open(self.log_file, 'a') as f:
                        f.write(f"cost.reshape(1, -1): {cost.reshape(1, -1)}\n")
                        f.write(f"linear_sum_assignment result: row_ind={row_ind}, col_ind={col_ind}\n")
                    assignments[frame_indices[i]] = col_ind[0]
                    with open(self.log_file, 'a') as f:
                        f.write(f"col_ind[0]: {col_ind[0]}\n")
                        f.write(f"assignments[{frame_indices[i]}]: {assignments[frame_indices[i]]}\n")
                        f.write(f"Assigned spot {frame_indices[i]} to locus {col_ind[0]}\n")
                else:
                    assignments[frame_indices[i]] = -1
                    with open(self.log_file, 'a') as f:
                        f.write(f"assignments[{frame_indices[i]}]: {assignments[frame_indices[i]]}\n")
                        f.write(f"Spot {i} in frame {t} classified as noise\n")

            with open(self.log_file, 'a') as f:
                f.write(f"p_s_l (after spot loop): {p_s_l}\n")

            for l in self.locus_positions.keys():
                locus_spots = real_spots[assignments == l, :2]
                with open(self.log_file, 'a') as f:
                    f.write(f"\nUpdating locus {l}\n")
                    f.write(f"assignments == {l}: {assignments == l}\n")
                    f.write(f"locus_spots: {locus_spots}\n")
                if len(locus_spots) > 0:
                    self.locus_positions[l] = np.mean(locus_spots, axis=0)
                    with open(self.log_file, 'a') as f:
                        f.write(f"len(locus_spots): {len(locus_spots)}\n")
                        f.write(f"np.mean(locus_spots, axis=0): {np.mean(locus_spots, axis=0)}\n")
                        f.write(f"self.locus_positions[{l}]: {self.locus_positions[l]}\n")

        with open(self.log_file, 'a') as f:
            f.write("\n--- Bayesian Inference Completed ---\n")
            f.write(f"Final assignments: {assignments}\n")
            f.write(f"Final locus positions: {self.locus_positions}\n")
        return assignments

    # def is_spot_real(self, new_I1, new_I2, new_I3, reference_spots):
    #     """
    #     Determine if a new spot is real based on its intensities I1, I2, I3,
    #     using the reference spots to compute statistics. Ignores spatial and temporal information.

    #     Parameters:
    #     - new_I1 (float): Intensity I1 of the new spot.
    #     - new_I2 (float): Intensity I2 of the new spot.
    #     - new_I3 (float): Intensity I3 of the new spot.
    #     - reference_spots (np.ndarray): Array of shape (n, 6) with columns [x, y, I1, I2, I3, t]

    #     Returns:
    #     - bool: True if the spot is classified as real, False otherwise.
    #     """
    #     # Log the start of the function and input parameters
    #     with open(self.log_file, 'a') as f:
    #         f.write("\n--- Starting is_spot_real ---\n")
    #         f.write(f"new_I1: {new_I1}\n")
    #         f.write(f"new_I2: {new_I2}\n")
    #         f.write(f"new_I3: {new_I3}\n")
    #         f.write(f"reference_spots shape: {reference_spots.shape}\n")

    #     # Extract I1 and I2 from reference spots for statistics
    #     I1_ref = reference_spots[:, 2]
    #     I2_ref = reference_spots[:, 3]

    #     # Compute means and standard deviations
    #     mu_I1 = np.mean(I1_ref)
    #     sigma_I1 = np.std(I1_ref)
    #     mu_I2 = np.mean(I2_ref)
    #     sigma_I2 = np.std(I2_ref)
    #     with open(self.log_file, 'a') as f:
    #         f.write(f"mu_I1: {mu_I1}\n")
    #         f.write(f"sigma_I1: {sigma_I1}\n")
    #         f.write(f"mu_I2: {mu_I2}\n")
    #         f.write(f"sigma_I2: {sigma_I2}\n")

    #     # Compute Z-scores for the new spot, handling zero standard deviation
    #     z_I1 = (new_I1 - mu_I1) / sigma_I1 if sigma_I1 > 0 else 0
    #     z_I2 = (new_I2 - mu_I2) / sigma_I2 if sigma_I2 > 0 else 0
    #     z_combined = (z_I1 + z_I2) / 2
    #     with open(self.log_file, 'a') as f:
    #         f.write(f"z_I1: {z_I1}\n")
    #         f.write(f"z_I2: {z_I2}\n")
    #         f.write(f"z_combined: {z_combined}\n")

    #     # Compute scaling factor based on combined Z-scores
    #     scaling_factor = 0.5 + 0.5 * sigmoid(z_combined)
    #     with open(self.log_file, 'a') as f:
    #         f.write(f"scaling_factor: {scaling_factor}\n")

    #     # Compute initial probability for I3 using sigmoid
    #     midpoint = 1.3
    #     k = 20
    #     P_real_I3 = sigmoid(k * (new_I3 - midpoint))
    #     with open(self.log_file, 'a') as f:
    #         f.write(f"midpoint: {midpoint}\n")
    #         f.write(f"k: {k}\n")
    #         f.write(f"P_real_I3: {P_real_I3}\n")

    #     # Adjust probability based on definite conditions
    #     if new_I3 >= 1.5:
    #         P_real_final = 1.0
    #     elif new_I3 <= 1.1:
    #         P_real_final = 0.0
    #     else:
    #         P_real_final = P_real_I3
    #     with open(self.log_file, 'a') as f:
    #         f.write(f"P_real_final: {P_real_final}\n")

    #     # Combine probabilities from I3 and I1/I2
    #     combined_probability_intensity = P_real_final * scaling_factor
    #     with open(self.log_file, 'a') as f:
    #         f.write(f"combined_probability_intensity: {combined_probability_intensity}\n")

    #     # Define likelihoods for Bayesian update
    #     p_s_z1 = combined_probability_intensity
    #     p_s_z0 = 1.0 - combined_probability_intensity
    #     with open(self.log_file, 'a') as f:
    #         f.write(f"p_s_z1: {p_s_z1}\n")
    #         f.write(f"p_s_z0: {p_s_z0}\n")

    #     # Use fixed priors from the original function
    #     p_z1 = 0.5
    #     p_z0 = 0.5
    #     with open(self.log_file, 'a') as f:
    #         f.write(f"p_z1: {p_z1}\n")
    #         f.write(f"p_z0: {p_z0}\n")

    #     # Compute posterior probability using Bayes' theorem
    #     numerator = p_s_z1 * p_z1
    #     denominator = numerator + p_s_z0 * p_z0 + 1e-12  # Small epsilon to avoid division by zero
    #     p_z1_s = numerator / denominator
    #     with open(self.log_file, 'a') as f:
    #         f.write(f"numerator: {numerator}\n")
    #         f.write(f"denominator: {denominator}\n")
    #         f.write(f"p_z1_s: {p_z1_s}\n")

    #     # Classify the spot as real if posterior probability > 0.5
    #     is_real = p_z1_s > 0.5
    #     with open(self.log_file, 'a') as f:
    #         f.write(f"is_real: {is_real}\n")
    #         f.write("--- Ending is_spot_real ---\n")

    #     return is_real
    def is_spot_real(self, new_I1, new_I2, new_I3, reference_spots):
        """
        Determine if a new spot is real using a multivariate GMM on I1, I2, I3 intensities.
        The GMM component with the higher mean intensity is classified as 'spot', and the lower as 'noise'.

        Parameters:
        - new_I1 (float): Intensity I1 of the new spot.
        - new_I2 (float): Intensity I2 of the new spot.
        - new_I3 (float): Intensity I3 of the new spot.
        - reference_spots (np.ndarray): Array of shape (n, 6) with columns [x, y, I1, I2, I3, t]

        Returns:
        - bool: True if the spot is classified as real (spot), False if noise.
        """
        # Log the start of the function and input parameters
        with open(self.log_file, 'a') as f:
            f.write("\n--- Starting is_spot_real ---\n")
            f.write(f"new_I1: {new_I1}\n")
            f.write(f"new_I2: {new_I2}\n")
            f.write(f"new_I3: {new_I3}\n")
            f.write(f"reference_spots shape: {reference_spots.shape}\n")
            f.write(f"reference_spots: {reference_spots}\n")

        # Extract I1, I2, I3 from reference spots
        intensities_ref = reference_spots[:, 2:5]  # Columns 2, 3, 4 are I1, I2, I3
        with open(self.log_file, 'a') as f:
            f.write(f"intensities_ref shape: {intensities_ref.shape}\n")
            f.write(f"intensities_ref: {intensities_ref}\n")

        # Check if we have enough data points for GMM
        n_samples = intensities_ref.shape[0]
        with open(self.log_file, 'a') as f:
            f.write(f"n_samples: {n_samples}\n")
        if n_samples < 2:
            with open(self.log_file, 'a') as f:
                f.write("Not enough reference spots for GMM (need at least 2). Classifying as noise.\n")
                f.write("--- Ending is_spot_real ---\n")
            return False

        # Fit a 2-component multivariate GMM
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        try:
            gmm.fit(intensities_ref)
            with open(self.log_file, 'a') as f:
                f.write("GMM fitting successful.\n")
        except Exception as e:
            with open(self.log_file, 'a') as f:
                f.write(f"GMM fitting failed: {str(e)}\n")
                f.write("Classifying as noise due to fitting failure.\n")
                f.write("--- Ending is_spot_real ---\n")
            return False

        # Get the means of the two components
        means = gmm.means_
        with open(self.log_file, 'a') as f:
            f.write(f"gmm.means_: {means}\n")

        # Compute the mean intensity (average of I1, I2, I3) for each component
        mean_intensities = np.mean(means, axis=1)
        with open(self.log_file, 'a') as f:
            f.write(f"mean_intensities (average of I1, I2, I3 per component): {mean_intensities}\n")

        # Identify which component is "spot" (higher mean) and which is "noise" (lower mean)
        spot_component = np.argmax(mean_intensities)
        noise_component = np.argmin(mean_intensities)
        with open(self.log_file, 'a') as f:
            f.write(f"spot_component (higher mean): {spot_component}\n")
            f.write(f"noise_component (lower mean): {noise_component}\n")
            f.write(f"spot_component mean: {mean_intensities[spot_component]}\n")
            f.write(f"noise_component mean: {mean_intensities[noise_component]}\n")

        # Prepare the new spot's intensities for prediction
        new_spot_intensities = np.array([[new_I1, new_I2, new_I3]])
        with open(self.log_file, 'a') as f:
            f.write(f"new_spot_intensities: {new_spot_intensities}\n")

        # Predict the component for the new spot
        predicted_label = gmm.predict(new_spot_intensities)[0]
        with open(self.log_file, 'a') as f:
            f.write(f"predicted_label: {predicted_label}\n")

        # Classify as real if the predicted label matches the spot component
        is_real = (predicted_label == spot_component)
        with open(self.log_file, 'a') as f:
            f.write(f"is_real (predicted_label == spot_component): {is_real}\n")
            f.write("--- Ending is_spot_real ---\n")

        return is_real
    
    def fit(self, spots, n_iterations=5):
        """Fit the detector to the spot data over multiple iterations."""
        with open(self.log_file, 'a') as f:
            f.write("\n--- Starting Fit Method ---\n")
            f.write(f"spots: {spots}\n")
            f.write(f"n_iterations: {n_iterations}\n")

        self.estimate_parameters(spots)
        with open(self.log_file, 'a') as f:
            f.write(f"After estimate_parameters: sigma_s = {self.sigma_s}, lambda_decay = {self.lambda_decay}\n")

        self.real_spot_mask = self.spot_validation(spots)
        with open(self.log_file, 'a') as f:
            f.write(f"self.real_spot_mask: {self.real_spot_mask}\n")

        real_spots = spots[self.real_spot_mask]
        with open(self.log_file, 'a') as f:
            f.write(f"real_spots: {real_spots}\n")

        initial_assignments = self.initialize_loci(spots, self.real_spot_mask)
        with open(self.log_file, 'a') as f:
            f.write(f"initial_assignments: {initial_assignments}\n")

        unique_loci = np.unique(initial_assignments[initial_assignments >= 0])
        with open(self.log_file, 'a') as f:
            f.write(f"initial_assignments >= 0: {initial_assignments >= 0}\n")
            f.write(f"unique_loci: {unique_loci}\n")

        initial_locus_positions = {}
        for l in unique_loci:
            locus_spots = spots[initial_assignments == l, :2]
            with open(self.log_file, 'a') as f:
                f.write(f"l: {l}\n")
                f.write(f"initial_assignments == l: {initial_assignments == l}\n")
                f.write(f"locus_spots: {locus_spots}\n")
            if len(locus_spots) > 0:
                initial_locus_positions[l] = np.mean(locus_spots, axis=0)
                with open(self.log_file, 'a') as f:
                    f.write(f"len(locus_spots): {len(locus_spots)}\n")
                    f.write(f"np.mean(locus_spots, axis=0): {np.mean(locus_spots, axis=0)}\n")
                    f.write(f"initial_locus_positions[{l}]: {initial_locus_positions[l]}\n")
        with open(self.log_file, 'a') as f:
            f.write(f"initial_locus_positions: {initial_locus_positions}\n")

        assignments = initial_assignments.copy()
        with open(self.log_file, 'a') as f:
            f.write(f"assignments (initial copy): {assignments}\n")

        for iteration in range(n_iterations):
            with open(self.log_file, 'a') as f:
                f.write(f"\nIteration {iteration + 1}/{n_iterations}\n")
                f.write(f"iteration: {iteration}\n")
            prev_assignments_real = assignments[self.real_spot_mask]
            with open(self.log_file, 'a') as f:
                f.write(f"prev_assignments_real: {prev_assignments_real}\n")
            assignments_real = self.bayesian_inference(real_spots, prev_assignments_real, initial_locus_positions)
            with open(self.log_file, 'a') as f:
                f.write(f"assignments_real after bayesian_inference: {assignments_real}\n")
            assignments[self.real_spot_mask] = assignments_real
            with open(self.log_file, 'a') as f:
                f.write(f"assignments after iteration {iteration + 1}: {assignments}\n")

        full_assignments = np.full(len(spots), -1, dtype=int)
        with open(self.log_file, 'a') as f:
            f.write(f"len(spots): {len(spots)}\n")
            f.write(f"full_assignments (initialized): {full_assignments}\n")
        full_assignments[self.real_spot_mask] = assignments_real
        with open(self.log_file, 'a') as f:
            f.write(f"full_assignments (final): {full_assignments}\n")
            f.write("\n--- Fit Method Completed ---\n")

        return {
            "assignments": full_assignments,
            "log_file": self.log_file
        }

