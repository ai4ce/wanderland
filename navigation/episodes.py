#!/usr/bin/env python3
"""
NavScene - Navigation Scene Episode Data Container

This module provides a class for managing individual navigation episodes
in the VLN-CE Isaac dataset format.

Author: NaVILA-Bench Team
License: MIT
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
import copy


@dataclass
class Goal:
    """Represents a navigation goal with position and success radius"""
    position: List[float]  # [x, y, z]
    radius: float = 3.0

    def __post_init__(self):
        """Validate goal data"""
        if len(self.position) != 3:
            raise ValueError(f"Goal position must be [x, y, z], got {len(self.position)} elements")
        if self.radius <= 0:
            raise ValueError(f"Goal radius must be positive, got {self.radius}")

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {"position": self.position, "radius": self.radius}

    def distance_to(self, position: List[float]) -> float:
        """Calculate Euclidean distance from position to goal"""
        return np.linalg.norm(np.array(self.position) - np.array(position))

    def is_reached(self, position: List[float]) -> bool:
        """Check if position is within goal radius"""
        return self.distance_to(position) <= self.radius


class NavScene:
    """
    Navigation Scene Episode Container

    Manages data for a single VLN-CE navigation episode including:
    - Scene metadata and identifiers
    - Start pose and goal specifications
    - Natural language instruction
    - Ground truth trajectories and waypoints

    Example:
        >>> scene = NavScene.from_dict(episode_dict)
        >>> print(f"Navigate in {scene.scene_id}: {scene.instruction}")
        >>> print(f"Distance: {scene.geodesic_distance:.2f}m, Steps: {scene.gt_steps}")
        >>> scene.to_json_file("episode_001.json")
    """

    def __init__(
        self,
        episode_id: int,
        scene_id: str,
        start_position: List[float],
        start_rotation: List[float],
        geodesic_distance: float,
        goals: List[Goal],
        instruction: str,
        reference_path: List[List[float]],
        gt_locations: List[List[float]],
        gt_steps: int,
        **kwargs
    ):
        """
        Initialize a navigation scene episode

        Args:
            episode_id: Unique episode identifier
            scene_id: Scene file path/identifier
            start_position: Starting [x, y, z] position
            start_rotation: Starting quaternion [x, y, z, w]
            geodesic_distance: Shortest path distance to goal (meters)
            goals: List of Goal objects
            instruction: Natural language navigation instruction
            reference_path: High-level waypoints
            gt_locations: Dense ground truth trajectory
            gt_steps: Number of forward steps
            **kwargs: Additional custom fields
        """
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.start_position = start_position
        self.start_rotation = start_rotation
        self.geodesic_distance = geodesic_distance
        self.goals = goals
        self.instruction = instruction
        self.reference_path = reference_path
        self.gt_locations = gt_locations
        self.gt_steps = gt_steps

        # Store any additional fields
        self.extra_fields = kwargs

        # Validate data
        self._validate()

    def _validate(self):
        """Validate episode data integrity"""
        # Validate positions
        if len(self.start_position) != 3:
            raise ValueError(f"start_position must be [x, y, z], got {len(self.start_position)}")

        # Validate quaternion
        if len(self.start_rotation) != 4:
            raise ValueError(f"start_rotation must be quaternion [x, y, z, w], got {len(self.start_rotation)}")

        # Check quaternion normalization
        norm = np.linalg.norm(self.start_rotation)
        if abs(norm - 1.0) > 0.01:
            raise ValueError(f"start_rotation quaternion not normalized (norm={norm:.4f})")

        # Validate distance
        if self.geodesic_distance <= 0:
            raise ValueError(f"geodesic_distance must be positive, got {self.geodesic_distance}")

        # Validate goals
        if not self.goals or len(self.goals) == 0:
            raise ValueError("Must have at least one goal")

        # Validate paths
        if len(self.reference_path) < 2:
            raise ValueError(f"reference_path must have at least 2 waypoints, got {len(self.reference_path)}")

        if len(self.gt_locations) < 2:
            raise ValueError(f"gt_locations must have at least 2 positions, got {len(self.gt_locations)}")

        # Validate steps
        if self.gt_steps < 1:
            raise ValueError(f"gt_steps must be at least 1, got {self.gt_steps}")

        # Validate instruction
        if not self.instruction or len(self.instruction.strip()) < 5:
            raise ValueError("instruction must be a non-empty string")

    # ==================== Factory Methods ====================

    @classmethod
    def from_dict(cls, data: Dict) -> 'NavScene':
        """
        Create NavScene from dictionary

        Args:
            data: Dictionary with episode data

        Returns:
            NavScene instance

        Example:
            >>> episode_dict = json.load(open("episode.json"))
            >>> scene = NavScene.from_dict(episode_dict)
        """
        # Parse goals
        goals = [Goal(**g) for g in data['goals']]

        # Extract known fields
        known_fields = {
            'episode_id', 'scene_id', 'start_position', 'start_rotation',
            'geodesic_distance', 'goals', 'instruction', 'reference_path',
            'gt_locations', 'gt_steps'
        }

        # Separate extra fields
        kwargs = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            episode_id=data['episode_id'],
            scene_id=data['scene_id'],
            start_position=data['start_position'],
            start_rotation=data['start_rotation'],
            geodesic_distance=data['geodesic_distance'],
            goals=goals,
            instruction=data['instruction'],
            reference_path=data['reference_path'],
            gt_locations=data['gt_locations'],
            gt_steps=data['gt_steps'],
            **kwargs
        )

    @classmethod
    def from_json_file(cls, filepath: Union[str, Path]) -> 'NavScene':
        """
        Load NavScene from JSON file

        Args:
            filepath: Path to JSON file

        Returns:
            NavScene instance

        Example:
            >>> scene = NavScene.from_json_file("episode_001.json")
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dataset(cls, dataset_path: Union[str, Path], episode_id: int) -> 'NavScene':
        """
        Load specific episode from full dataset file

        Args:
            dataset_path: Path to dataset JSON file
            episode_id: Episode ID to load

        Returns:
            NavScene instance

        Raises:
            ValueError: If episode_id not found

        Example:
            >>> scene = NavScene.from_dataset("vln_ce_isaac_v1.json", episode_id=42)
        """
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        for episode in data['episodes']:
            if episode['episode_id'] == episode_id:
                return cls.from_dict(episode)

        raise ValueError(f"Episode {episode_id} not found in dataset")

    # ==================== Export Methods ====================

    def to_dict(self) -> Dict:
        """
        Convert to dictionary

        Returns:
            Dictionary representation of episode

        Example:
            >>> scene_dict = scene.to_dict()
            >>> json.dump(scene_dict, open("output.json", "w"))
        """
        data = {
            'episode_id': self.episode_id,
            'scene_id': self.scene_id,
            'start_position': self.start_position,
            'start_rotation': self.start_rotation,
            'geodesic_distance': self.geodesic_distance,
            'goals': [g.to_dict() for g in self.goals],
            'instruction': self.instruction,
            'reference_path': self.reference_path,
            'gt_locations': self.gt_locations,
            'gt_steps': self.gt_steps,
        }

        # Add extra fields
        data.update(self.extra_fields)

        return data

    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string

        Args:
            indent: Number of spaces for indentation

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_json_file(self, filepath: Union[str, Path], indent: int = 2):
        """
        Save to JSON file

        Args:
            filepath: Output file path
            indent: Number of spaces for indentation

        Example:
            >>> scene.to_json_file("episode_001.json")
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    # ==================== Data Access Methods ====================

    def get_start_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get start pose as numpy arrays

        Returns:
            Tuple of (position, rotation) as numpy arrays

        Example:
            >>> pos, rot = scene.get_start_pose()
            >>> print(f"Start at {pos}, facing {rot}")
        """
        return np.array(self.start_position), np.array(self.start_rotation)

    def get_primary_goal(self) -> Goal:
        """
        Get the primary (first) goal

        Returns:
            Primary Goal object
        """
        return self.goals[0]

    def get_goal_position(self, goal_idx: int = 0) -> np.ndarray:
        """
        Get goal position as numpy array

        Args:
            goal_idx: Goal index (default: 0 for primary goal)

        Returns:
            Goal position as numpy array [x, y, z]
        """
        return np.array(self.goals[goal_idx].position)

    def get_reference_path_array(self) -> np.ndarray:
        """
        Get reference path as numpy array

        Returns:
            Array of shape (N, 3) with waypoints
        """
        return np.array(self.reference_path)

    def get_gt_locations_array(self) -> np.ndarray:
        """
        Get ground truth locations as numpy array

        Returns:
            Array of shape (N, 3) with trajectory positions
        """
        return np.array(self.gt_locations)

    def get_start_rotation_matrix(self) -> np.ndarray:
        """
        Convert start quaternion to rotation matrix

        Returns:
            3x3 rotation matrix
        """
        return self.quaternion_to_rotation_matrix(self.start_rotation)

    def get_start_yaw(self) -> float:
        """
        Get start yaw angle in radians

        Returns:
            Yaw angle in radians
        """
        return self.quaternion_to_yaw(self.start_rotation)

    # ==================== Utility Methods ====================

    @staticmethod
    def quaternion_to_rotation_matrix(quat: List[float]) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix

        Args:
            quat: Quaternion [x, y, z, w]

        Returns:
            3x3 rotation matrix
        """
        x, y, z, w = quat
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    @staticmethod
    def quaternion_to_yaw(quat: List[float]) -> float:
        """
        Extract yaw angle from quaternion

        Args:
            quat: Quaternion [x, y, z, w]

        Returns:
            Yaw angle in radians
        """
        x, y, z, w = quat
        # Yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> List[float]:
        """
        Convert yaw angle to quaternion (rotation around z-axis)

        Args:
            yaw: Yaw angle in radians

        Returns:
            Quaternion [x, y, z, w]
        """
        return [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]

    def compute_path_length(self, use_gt: bool = True) -> float:
        """
        Compute total path length

        Args:
            use_gt: If True, use gt_locations; else use reference_path

        Returns:
            Total path length in meters
        """
        path = self.gt_locations if use_gt else self.reference_path
        path_array = np.array(path)

        if len(path_array) < 2:
            return 0.0

        # Compute sum of distances between consecutive points
        diffs = np.diff(path_array, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.sum(distances))

    def get_distance_to_goal(self, position: List[float], goal_idx: int = 0) -> float:
        """
        Calculate distance from position to goal

        Args:
            position: Current position [x, y, z]
            goal_idx: Goal index

        Returns:
            Euclidean distance in meters
        """
        return self.goals[goal_idx].distance_to(position)

    def is_goal_reached(self, position: List[float], goal_idx: int = 0) -> bool:
        """
        Check if position reaches the goal

        Args:
            position: Current position [x, y, z]
            goal_idx: Goal index

        Returns:
            True if within goal radius
        """
        return self.goals[goal_idx].is_reached(position)

    # ==================== Analysis Methods ====================

    def get_statistics(self) -> Dict:
        """
        Compute episode statistics

        Returns:
            Dictionary with statistics
        """
        return {
            'episode_id': self.episode_id,
            'geodesic_distance': self.geodesic_distance,
            'path_length': self.compute_path_length(use_gt=True),
            'reference_path_length': self.compute_path_length(use_gt=False),
            'gt_steps': self.gt_steps,
            'num_waypoints': len(self.reference_path),
            'num_gt_locations': len(self.gt_locations),
            'instruction_length': len(self.instruction),
            'num_goals': len(self.goals),
            'start_yaw_deg': np.degrees(self.get_start_yaw()),
        }

    def print_summary(self):
        """Print human-readable summary"""
        print(f"{'='*70}")
        print(f"Episode {self.episode_id}")
        print(f"{'='*70}")
        print(f"Scene:        {self.scene_id}")
        print(f"Instruction:  {self.instruction}")
        print(f"\nStart Pose:")
        print(f"  Position:   {self.start_position}")
        print(f"  Rotation:   {self.start_rotation}")
        print(f"  Yaw:        {np.degrees(self.get_start_yaw()):.1f}°")
        print(f"\nGoal:")
        goal = self.get_primary_goal()
        print(f"  Position:   {goal.position}")
        print(f"  Radius:     {goal.radius}m")
        print(f"\nMetrics:")
        print(f"  Geodesic Distance: {self.geodesic_distance:.2f}m")
        print(f"  GT Path Length:    {self.compute_path_length(True):.2f}m")
        print(f"  GT Steps:          {self.gt_steps}")
        print(f"  Waypoints:         {len(self.reference_path)}")
        print(f"  GT Locations:      {len(self.gt_locations)}")
        print(f"{'='*70}")

    # ==================== Modification Methods ====================

    def copy(self) -> 'NavScene':
        """Create a deep copy of this NavScene"""
        return NavScene.from_dict(self.to_dict())

    def set_start_pose(self, position: List[float], rotation: List[float]):
        """
        Update start pose

        Args:
            position: New start position [x, y, z]
            rotation: New start rotation quaternion [x, y, z, w]
        """
        self.start_position = position
        self.start_rotation = rotation
        self._validate()

    def set_start_yaw(self, yaw: float):
        """
        Set start orientation using yaw angle

        Args:
            yaw: Yaw angle in radians
        """
        self.start_rotation = self.yaw_to_quaternion(yaw)

    def add_goal(self, position: List[float], radius: float = 3.0):
        """
        Add a new goal

        Args:
            position: Goal position [x, y, z]
            radius: Success radius
        """
        self.goals.append(Goal(position, radius))

    def update_instruction(self, instruction: str):
        """
        Update navigation instruction

        Args:
            instruction: New instruction text
        """
        if not instruction or len(instruction.strip()) < 5:
            raise ValueError("Instruction must be non-empty")
        self.instruction = instruction

    # ==================== Visualization Support ====================

    def get_plot_data(self) -> Dict:
        """
        Get data formatted for plotting/visualization

        Returns:
            Dictionary with plottable arrays
        """
        ref_path = self.get_reference_path_array()
        gt_path = self.get_gt_locations_array()

        return {
            'start_pos': np.array(self.start_position),
            'goal_pos': self.get_goal_position(),
            'goal_radius': self.goals[0].radius,
            'reference_path_x': ref_path[:, 0],
            'reference_path_y': ref_path[:, 1],
            'gt_path_x': gt_path[:, 0],
            'gt_path_y': gt_path[:, 1],
        }

    def __repr__(self) -> str:
        """String representation"""
        return (f"NavScene(episode_id={self.episode_id}, "
                f"scene='{self.scene_id}', "
                f"distance={self.geodesic_distance:.2f}m, "
                f"steps={self.gt_steps})")

    def __str__(self) -> str:
        """User-friendly string"""
        return f"Episode {self.episode_id}: {self.instruction[:50]}..."


# ==================== Dataset Collection Class ====================

class NavSceneDataset:
    """
    Container for multiple NavScene episodes

    Provides batch loading, filtering, and iteration over episodes
    """

    def __init__(self, scenes: List[NavScene] = None):
        """
        Initialize dataset

        Args:
            scenes: List of NavScene objects
        """
        self.scenes = scenes or []

    @classmethod
    def from_json_file(cls, filepath: Union[str, Path]) -> 'NavSceneDataset':
        """
        Load entire dataset from JSON file

        Args:
            filepath: Path to dataset JSON file

        Returns:
            NavSceneDataset instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        scenes = [NavScene.from_dict(ep) for ep in data['episodes']]
        return cls(scenes)

    def to_json_file(self, filepath: Union[str, Path], indent: int = 2):
        """
        Save entire dataset to JSON file

        Args:
            filepath: Output file path
            indent: JSON indentation
        """
        data = {
            'episodes': [scene.to_dict() for scene in self.scenes]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)

    def __len__(self) -> int:
        """Number of scenes in dataset"""
        return len(self.scenes)

    def __getitem__(self, idx: int) -> NavScene:
        """Get scene by index"""
        return self.scenes[idx]

    def __iter__(self):
        """Iterate over scenes"""
        return iter(self.scenes)

    def get_by_episode_id(self, episode_id: int) -> Optional[NavScene]:
        """
        Get scene by episode ID

        Args:
            episode_id: Episode ID to find

        Returns:
            NavScene or None if not found
        """
        for scene in self.scenes:
            if scene.episode_id == episode_id:
                return scene
        return None

    def filter_by_scene(self, scene_id: str) -> 'NavSceneDataset':
        """
        Filter episodes by scene ID

        Args:
            scene_id: Scene identifier (can be partial match)

        Returns:
            New NavSceneDataset with filtered episodes
        """
        filtered = [s for s in self.scenes if scene_id in s.scene_id]
        return NavSceneDataset(filtered)

    def filter_by_distance(self, min_dist: float = 0, max_dist: float = float('inf')) -> 'NavSceneDataset':
        """
        Filter episodes by geodesic distance

        Args:
            min_dist: Minimum distance (inclusive)
            max_dist: Maximum distance (inclusive)

        Returns:
            New NavSceneDataset with filtered episodes
        """
        filtered = [s for s in self.scenes if min_dist <= s.geodesic_distance <= max_dist]
        return NavSceneDataset(filtered)

    def get_statistics(self) -> Dict:
        """
        Compute dataset-wide statistics

        Returns:
            Dictionary with statistics
        """
        if not self.scenes:
            return {}

        distances = [s.geodesic_distance for s in self.scenes]
        steps = [s.gt_steps for s in self.scenes]

        return {
            'num_episodes': len(self.scenes),
            'distance_mean': np.mean(distances),
            'distance_std': np.std(distances),
            'distance_min': np.min(distances),
            'distance_max': np.max(distances),
            'steps_mean': np.mean(steps),
            'steps_std': np.std(steps),
            'steps_min': np.min(steps),
            'steps_max': np.max(steps),
        }

    def print_summary(self):
        """Print dataset summary"""
        stats = self.get_statistics()
        print(f"Dataset Summary")
        print(f"{'='*50}")
        print(f"Total Episodes: {stats['num_episodes']}")
        print(f"\nDistance Statistics (meters):")
        print(f"  Mean: {stats['distance_mean']:.2f} ± {stats['distance_std']:.2f}")
        print(f"  Range: [{stats['distance_min']:.2f}, {stats['distance_max']:.2f}]")
        print(f"\nStep Statistics:")
        print(f"  Mean: {stats['steps_mean']:.1f} ± {stats['steps_std']:.1f}")
        print(f"  Range: [{stats['steps_min']}, {stats['steps_max']}]")
        print(f"{'='*50}")
