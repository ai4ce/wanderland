#!/usr/bin/env python3
"""
Pose Estimation Evaluation Script for COLMAP Sparse Format

This script evaluates pose estimation results by comparing prediction and ground truth
poses stored in COLMAP sparse format (both .txt and .bin files supported).

Note: COLMAP stores world-to-camera poses, but this script converts them to 
camera-to-world poses for consistent evaluation using SVD-based alignment.

Evaluation metrics include:
- ATE (Absolute Trajectory Error): Translation ATE (no scale alignment), Translation ATE (scale aligned), Rotation ATE
- RTE (Relative Trajectory Error): Translation RTE (meters), Translation RTE (degrees), Rotation RTE (degrees)
- AUC@30: Area Under Curve metric using Translation RTE (degrees) and Rotation RTE (degrees)
- Reconstruction completeness ratio
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import struct
from typing import Dict, Tuple, List, Optional
import json
from dataclasses import dataclass


@dataclass
class Image:
    """COLMAP Image class"""
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read next bytes from file"""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


class COLMAPPoseReader:
    """Reader for COLMAP pose data from both .txt and .bin formats"""

    def __init__(self):
        pass

    def read_poses(self, directory: str) -> Dict[int, Dict]:
        """
        Read poses from COLMAP sparse directory
        Automatically detects and reads from either .txt or .bin format
        """
        txt_path = os.path.join(directory, 'images.txt')
        bin_path = os.path.join(directory, 'images.bin')

        if os.path.exists(bin_path):
            print(f"Reading poses from {bin_path}")
            images = read_images_binary(bin_path)
        elif os.path.exists(txt_path):
            print(f"Reading poses from {txt_path}")
            images = read_images_text(txt_path)
        else:
            raise FileNotFoundError(
                f"Neither images.bin nor images.txt found in {directory}")

        # Convert Image objects to dictionary format for compatibility
        pose_dict = {}
        for image_id, image in images.items():
            # Convert quaternion to rotation matrix
            # COLMAP uses w, x, y, z format, scipy uses x, y, z, w
            rotation = R.from_quat(
                [image.qvec[1], image.qvec[2], image.qvec[3], image.qvec[0]]).as_matrix()

            # COLMAP stores world-to-camera poses, but we need camera-to-world for evaluation
            # Construct world-to-camera transformation matrix
            w2c_T = np.eye(4)
            w2c_T[:3, :3] = rotation
            w2c_T[:3, 3] = image.tvec

            # Invert to get camera-to-world transformation
            c2w_T = np.linalg.inv(w2c_T)
            c2w_rotation = c2w_T[:3, :3]
            c2w_translation = c2w_T[:3, 3]

            pose_dict[image_id] = {
                'name': image.name,
                'camera_id': image.camera_id,
                'rotation': c2w_rotation,
                'translation': c2w_translation,
                'quaternion': image.qvec
            }

        return pose_dict


class PoseAlignment:
    """Pose alignment utilities for SE3 and SE3+scale transformations"""

    @staticmethod
    def se3_alignment(pred_poses: List[np.ndarray], gt_poses: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SE3 alignment between predicted and ground truth poses
        Returns: (rotation_matrix, translation_vector)
        """
        # Extract positions and rotations
        pred_positions = np.array([pose[:3, 3] for pose in pred_poses])
        gt_positions = np.array([pose[:3, 3] for pose in gt_poses])

        # Compute centroids
        pred_centroid = np.mean(pred_positions, axis=0)
        gt_centroid = np.mean(gt_positions, axis=0)

        # Center the positions
        pred_centered = pred_positions - pred_centroid
        gt_centered = gt_positions - gt_centroid

        # Compute rotation using SVD
        H = pred_centered.T @ gt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = gt_centroid - R @ pred_centroid

        return R, t

    @staticmethod
    def se3_scale_alignment(pred_poses: List[np.ndarray], gt_poses: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute SE3+scale alignment between predicted and ground truth poses
        Returns: (rotation_matrix, translation_vector, scale_factor)
        """
        # Extract positions and rotations
        pred_positions = np.array([pose[:3, 3] for pose in pred_poses])
        gt_positions = np.array([pose[:3, 3] for pose in gt_poses])

        pred_rotations = np.array([pose[:3, :3] for pose in pred_poses])
        gt_rotations = np.array([pose[:3, :3] for pose in gt_poses])

        # Compute centroids
        pred_centroid = np.mean(pred_positions, axis=0)
        gt_centroid = np.mean(gt_positions, axis=0)

        # Center the positions
        pred_centered = pred_positions - pred_centroid
        gt_centered = gt_positions - gt_centroid

        # Compute scale
        pred_norm = np.linalg.norm(pred_centered, axis=1)
        gt_norm = np.linalg.norm(gt_centered, axis=1)

        # Avoid division by zero
        valid_indices = (pred_norm > 1e-8) & (gt_norm > 1e-8)
        if np.sum(valid_indices) == 0:
            scale = 1.0
        else:
            scale = np.mean(gt_norm[valid_indices] / pred_norm[valid_indices])

        # Scale the predicted positions
        pred_scaled = pred_centered * scale

        # Compute rotation using SVD
        H = pred_scaled.T @ gt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = gt_centroid - R @ (pred_centroid * scale)

        return R, t, scale


class PoseMetrics:
    """Compute various pose evaluation metrics"""

    @staticmethod
    def compute_ate_translation(pred_poses: List[np.ndarray], gt_poses: List[np.ndarray]) -> float:
        """Compute Absolute Trajectory Error for translation (RMSE in meters)"""
        pred_positions = np.array([pose[:3, 3] for pose in pred_poses])
        gt_positions = np.array([pose[:3, 3] for pose in gt_poses])

        errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
        return np.sqrt(np.mean(errors**2))

    @staticmethod
    def compute_ate_rotation(pred_poses: List[np.ndarray], gt_poses: List[np.ndarray]) -> float:
        """Compute Absolute Trajectory Error for rotation (RMSE in degrees)"""
        pred_rotations = np.array([pose[:3, :3] for pose in pred_poses])
        gt_rotations = np.array([pose[:3, :3] for pose in gt_poses])

        errors = []
        for pred_R, gt_R in zip(pred_rotations, gt_rotations):
            # Compute relative rotation error
            rel_R = gt_R.T @ pred_R
            angle_error = np.arccos(np.clip((np.trace(rel_R) - 1) / 2, -1, 1))
            errors.append(np.degrees(angle_error))

        return np.sqrt(np.mean(np.array(errors)**2))

    @staticmethod
    def compute_rte_translation(pred_poses: List[np.ndarray], gt_poses: List[np.ndarray]) -> float:
        """Compute Relative Trajectory Error for translation (RMSE in meters) - all pairs"""
        pred_positions = np.array([pose[:3, 3]
                                  for pose in pred_poses])  # Shape: (n, 3)
        gt_positions = np.array([pose[:3, 3]
                                for pose in gt_poses])      # Shape: (n, 3)

        # Vectorized computation for all pairs
        # pred_positions[i] - pred_positions[j] for all i, j pairs
        pred_diff = pred_positions[:, np.newaxis, :] - \
            pred_positions[np.newaxis, :, :]  # Shape: (n, n, 3)
        gt_diff = gt_positions[:, np.newaxis, :] - \
            gt_positions[np.newaxis, :, :]        # Shape: (n, n, 3)

        # Compute errors for all pairs
        errors = np.linalg.norm(pred_diff - gt_diff, axis=2)  # Shape: (n, n)

        # Extract upper triangular part (excluding diagonal) for unique pairs
        mask = np.triu(np.ones_like(errors, dtype=bool), k=1)
        pair_errors = errors[mask]

        return np.sqrt(np.mean(pair_errors**2))

    @staticmethod
    def compute_rte_translation_angle(pred_poses: List[np.ndarray], gt_poses: List[np.ndarray]) -> Tuple[float, np.ndarray]:
        """Compute Relative Trajectory Error for translation (RMSE in degrees) - scale invariant, all pairs"""
        pred_positions = np.array([pose[:3, 3]
                                  for pose in pred_poses])  # Shape: (n, 3)
        gt_positions = np.array([pose[:3, 3]
                                for pose in gt_poses])      # Shape: (n, 3)

        # Vectorized computation for all pairs
        pred_diff = pred_positions[:, np.newaxis, :] - \
            pred_positions[np.newaxis, :, :]  # Shape: (n, n, 3)
        gt_diff = gt_positions[:, np.newaxis, :] - \
            gt_positions[np.newaxis, :, :]        # Shape: (n, n, 3)

        # Compute norms for all pairs
        pred_norms = np.linalg.norm(pred_diff, axis=2)  # Shape: (n, n)
        gt_norms = np.linalg.norm(gt_diff, axis=2)      # Shape: (n, n)

        # Create mask for valid pairs (non-zero norms)
        valid_mask = (pred_norms > 1e-8) & (gt_norms > 1e-8)

        # Normalize vectors
        pred_unit = np.zeros_like(pred_diff)
        gt_unit = np.zeros_like(gt_diff)

        pred_unit[valid_mask] = pred_diff[valid_mask] / \
            pred_norms[valid_mask, np.newaxis]
        gt_unit[valid_mask] = gt_diff[valid_mask] / \
            gt_norms[valid_mask, np.newaxis]

        # Compute dot products for all pairs
        dot_products = np.sum(pred_unit * gt_unit, axis=2)  # Shape: (n, n)
        dot_products = np.clip(dot_products, -1, 1)

        # Compute angles
        angles = np.arccos(dot_products)  # Shape: (n, n)
        angles_deg = np.degrees(angles)

        # Extract upper triangular part (excluding diagonal) for unique pairs
        mask = np.triu(np.ones_like(angles_deg, dtype=bool), k=1)
        valid_pairs = mask & valid_mask

        if np.sum(valid_pairs) == 0:
            return 0.0, np.array([])

        pair_angles = angles_deg[valid_pairs]
        return np.sqrt(np.mean(pair_angles**2)), pair_angles

    @staticmethod
    def compute_rte_rotation(pred_poses: List[np.ndarray], gt_poses: List[np.ndarray]) -> Tuple[float, np.ndarray]:
        """Compute Relative Trajectory Error for rotation (RMSE in degrees) - all pairs"""
        pred_rotations = np.array([pose[:3, :3]
                                  for pose in pred_poses])  # Shape: (n, 3, 3)
        gt_rotations = np.array([pose[:3, :3]
                                for pose in gt_poses])      # Shape: (n, 3, 3)

        n = len(pred_rotations)

        # Vectorized computation for all pairs
        # pred_rotations[i] @ pred_rotations[j].T for all i, j pairs
        pred_rel = pred_rotations[:, np.newaxis, :, :] @ pred_rotations[np.newaxis,
                                                                        # Shape: (n, n, 3, 3)
                                                                        :, :, :].transpose(0, 1, 3, 2)
        gt_rel = gt_rotations[:, np.newaxis, :, :] @ gt_rotations[np.newaxis,
                                                                  # Shape: (n, n, 3, 3)
                                                                  :, :, :].transpose(0, 1, 3, 2)

        # Compute relative rotation errors
        rel_R = gt_rel.transpose(0, 1, 3, 2) @ pred_rel  # Shape: (n, n, 3, 3)

        # Compute traces for all pairs
        traces = np.trace(rel_R, axis1=2, axis2=3)  # Shape: (n, n)
        traces = np.clip(traces, -1, 3)  # Clamp for numerical stability

        # Compute angles
        angles = np.arccos(np.clip((traces - 1) / 2, -1, 1))  # Shape: (n, n)
        angles_deg = np.degrees(angles)

        # Extract upper triangular part (excluding diagonal) for unique pairs
        mask = np.triu(np.ones_like(angles_deg, dtype=bool), k=1)
        pair_angles = angles_deg[mask]

        return np.sqrt(np.mean(pair_angles**2)), pair_angles

    @staticmethod
    def calculate_auc(r_error, t_error, max_threshold=30):
        """
        Calculate the Area Under the Curve (AUC) for the given error arrays.

        :param r_error: numpy array representing R error values (Degree).
        :param t_error: numpy array representing T error values (Degree).
        :param max_threshold: maximum threshold value for binning the histogram.
        :return: cumulative sum of normalized histogram of maximum error values.
        """
        # Concatenate the error arrays along a new axis
        error_matrix = np.concatenate(
            (r_error[:, None], t_error[:, None]), axis=1)

        # Compute the maximum error value for each pair
        max_errors = np.max(error_matrix, axis=1)

        # Define histogram bins
        bins = np.arange(max_threshold + 1)

        # Calculate histogram of maximum error values
        histogram, _ = np.histogram(max_errors, bins=bins)

        # Normalize the histogram
        num_pairs = float(len(max_errors))
        normalized_histogram = histogram.astype(float) / num_pairs

        # Compute and return the cumulative sum of the normalized histogram
        return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


class PoseEvaluator:
    """Main pose evaluation class"""

    def __init__(self):
        self.reader = COLMAPPoseReader()
        self.alignment = PoseAlignment()
        self.metrics = PoseMetrics()

    def evaluate(self, pred_dir: str, gt_dir: str, output_file: str = None) -> Dict:
        """
        Evaluate pose estimation results
        
        Args:
            pred_dir: Directory containing prediction poses (COLMAP sparse format)
            gt_dir: Directory containing ground truth poses (COLMAP sparse format)
            output_file: Optional output file for results
        
        Returns:
            Dictionary containing evaluation results
        """
        print("Reading prediction poses...")
        pred_poses = self.reader.read_poses(pred_dir)

        print("Reading ground truth poses...")
        gt_poses = self.reader.read_poses(gt_dir)

        # Find intersection of image names
        pred_names = {data['name']: image_id for image_id,
                      data in pred_poses.items()}
        gt_names = {data['name']: image_id for image_id,
                    data in gt_poses.items()}

        common_names = set(pred_names.keys()) & set(gt_names.keys())
        print(
            f"Found {len(common_names)} common images out of {len(pred_poses)} predicted and {len(gt_poses)} ground truth")

        if len(common_names) == 0:
            raise ValueError(
                "No common images found between prediction and ground truth")

        # Extract poses for common images
        pred_poses_list = []
        gt_poses_list = []

        for name in sorted(common_names):
            pred_id = pred_names[name]
            gt_id = gt_names[name]

            # Convert to 4x4 transformation matrix
            pred_data = pred_poses[pred_id]
            gt_data = gt_poses[gt_id]

            pred_T = np.eye(4)
            pred_T[:3, :3] = pred_data['rotation']
            pred_T[:3, 3] = pred_data['translation']

            gt_T = np.eye(4)
            gt_T[:3, :3] = gt_data['rotation']
            gt_T[:3, 3] = gt_data['translation']

            pred_poses_list.append(pred_T)
            gt_poses_list.append(gt_T)

        # Compute reconstruction completeness
        completeness_ratio = len(pred_poses) / len(gt_poses)

        print("Computing scale-variant metrics...")
        # Scale-variant metrics with SE3 alignment
        R_se3, t_se3 = self.alignment.se3_alignment(
            pred_poses_list, gt_poses_list)

        # Apply SE3 alignment
        pred_poses_aligned_se3 = []
        for pose in pred_poses_list:
            aligned_pose = np.eye(4)
            aligned_pose[:3, :3] = R_se3 @ pose[:3, :3]
            aligned_pose[:3, 3] = R_se3 @ pose[:3, 3] + t_se3
            pred_poses_aligned_se3.append(aligned_pose)

        # Compute scale-variant metrics
        ate_trans_se3 = self.metrics.compute_ate_translation(
            pred_poses_aligned_se3, gt_poses_list)
        ate_rot_se3 = self.metrics.compute_ate_rotation(
            pred_poses_aligned_se3, gt_poses_list)
        rte_trans_se3 = self.metrics.compute_rte_translation(
            pred_poses_aligned_se3, gt_poses_list)
        rte_trans_angle_se3, t_errors_deg = self.metrics.compute_rte_translation_angle(
            pred_poses_aligned_se3, gt_poses_list)
        rte_rot_se3, r_errors_deg = self.metrics.compute_rte_rotation(
            pred_poses_aligned_se3, gt_poses_list)

        print("Computing scale-invariant metrics...")
        # Scale-invariant metrics with SE3+scale alignment
        R_sim3, t_sim3, scale = self.alignment.se3_scale_alignment(
            pred_poses_list, gt_poses_list)

        # Apply SE3+scale alignment
        pred_poses_aligned_sim3 = []
        for pose in pred_poses_list:
            aligned_pose = np.eye(4)
            aligned_pose[:3, :3] = R_sim3 @ pose[:3, :3]
            aligned_pose[:3, 3] = R_sim3 @ (pose[:3, 3] * scale) + t_sim3
            pred_poses_aligned_sim3.append(aligned_pose)

        # Compute scale-invariant metrics
        ate_trans_sim3 = self.metrics.compute_ate_translation(
            pred_poses_aligned_sim3, gt_poses_list)

        # Compute AUC@30 metric using the error arrays from RTE computations
        print("Computing AUC@30 metric...")
        if len(t_errors_deg) > 0 and len(r_errors_deg) > 0:
            auc30, _ = self.metrics.calculate_auc(
                r_errors_deg, t_errors_deg, max_threshold=30)
        else:
            auc30 = 0.0

        # Compile results
        results = {
            'reconstruction_completeness': {
                'num_predicted': len(pred_poses),
                'num_ground_truth': len(gt_poses),
                'num_common': len(common_names),
                'ratio': completeness_ratio
            },
            'ate_metrics': {
                'translation_ate_no_scale_m': ate_trans_se3,
                'translation_ate_scale_aligned_m': ate_trans_sim3,
                'rotation_ate_deg': ate_rot_se3
            },
            'rte_metrics': {
                'translation_rte_m': rte_trans_se3,
                'translation_rte_deg': rte_trans_angle_se3,
                'rotation_rte_deg': rte_rot_se3
            },
            'auc30': auc30,
            'alignment_parameters': {
                'se3_rotation': R_se3.tolist(),
                'se3_translation': t_se3.tolist(),
                'sim3_rotation': R_sim3.tolist(),
                'sim3_translation': t_sim3.tolist(),
                'scale_factor': scale
            }
        }

        # Save results
        if output_file:
            self.save_results(results, output_file)

        return results

    def save_results(self, results: Dict, output_file: str):
        """Save results to text file"""
        with open(output_file, 'w') as f:
            f.write("Pose Estimation Evaluation Results\n")
            f.write("=" * 50 + "\n\n")

            # Reconstruction completeness
            f.write("Reconstruction Completeness:\n")
            f.write(
                f"  Number of predicted images: {results['reconstruction_completeness']['num_predicted']}\n")
            f.write(
                f"  Number of ground truth images: {results['reconstruction_completeness']['num_ground_truth']}\n")
            f.write(
                f"  Number of common images: {results['reconstruction_completeness']['num_common']}\n")
            f.write(
                f"  Completeness ratio: {results['reconstruction_completeness']['ratio']:.4f}\n\n")

            # ATE metrics
            f.write("Absolute Trajectory Error (ATE) Metrics:\n")
            f.write(
                f"  Translation ATE (no scale alignment): {results['ate_metrics']['translation_ate_no_scale_m']:.6f} m\n")
            f.write(
                f"  Translation ATE (scale aligned): {results['ate_metrics']['translation_ate_scale_aligned_m']:.6f} m\n")
            f.write(
                f"  Rotation ATE: {results['ate_metrics']['rotation_ate_deg']:.6f} deg\n\n")

            # RTE metrics
            f.write("Relative Trajectory Error (RTE) Metrics:\n")
            f.write(
                f"  Translation RTE: {results['rte_metrics']['translation_rte_m']:.6f} m\n")
            f.write(
                f"  Translation RTE: {results['rte_metrics']['translation_rte_deg']:.6f} deg\n")
            f.write(
                f"  Rotation RTE: {results['rte_metrics']['rotation_rte_deg']:.6f} deg\n\n")

            # AUC@30 metric
            f.write("AUC@30 Metric:\n")
            f.write(f"  AUC@30: {results['auc30']:.6f}\n\n")

            # Alignment parameters
            f.write("Alignment Parameters:\n")
            f.write(
                f"  Scale factor: {results['alignment_parameters']['scale_factor']:.6f}\n")
            f.write(
                f"  SE3 Translation: {results['alignment_parameters']['se3_translation']}\n")

        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pose estimation results in COLMAP sparse format')
    parser.add_argument(
        'pred_dir', help='Directory containing prediction poses (COLMAP sparse format)')
    parser.add_argument(
        'gt_dir', help='Directory containing ground truth poses (COLMAP sparse format)')
    parser.add_argument(
        '--output', '-o', help='Output file for results (default: results.txt)')

    args = parser.parse_args()

    if not os.path.exists(args.pred_dir):
        print(f"Error: Prediction directory {args.pred_dir} does not exist")
        sys.exit(1)

    if not os.path.exists(args.gt_dir):
        print(f"Error: Ground truth directory {args.gt_dir} does not exist")
        sys.exit(1)

    output_file = args.output or 'pose_evaluation_results.txt'

    evaluator = PoseEvaluator()

    try:
        results = evaluator.evaluate(args.pred_dir, args.gt_dir, output_file)

        print("\nEvaluation completed successfully!")
        print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
