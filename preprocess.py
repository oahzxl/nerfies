"""
colmap should be generated under the root dir
"""


from absl import logging
from typing import Dict
import numpy as np
from nerfies.camera import Camera
import pycolmap
from pycolmap import Quaternion
import imageio
import cv2
import os


res = True
root_dir = '/home/xuanlei/nerfies/data/yellow_half'


def multi_res(picture_dir_path):
    picture_list = os.listdir(picture_dir_path)
    new_root = picture_dir_path.replace('rgb-raw', 'rgb')
    if not os.path.exists(new_root):
        os.mkdir(new_root)

    for reso in [1., 2., 4., 8.]:
        new_path = os.path.join(new_root, '%dx' % reso)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        for i in picture_list:
            if not ('png' in i or 'jpg' in i):
                continue
            img = cv2.imread(os.path.join(picture_dir_path, i))
            img = cv2.resize(img, dsize=None, fx=1/reso, fy=1/reso)
            cv2.imwrite(os.path.join(new_path, i), img)


if res:
    multi_res(os.path.join(root_dir, 'rgb-raw'))


### colmap
def convert_colmap_camera(colmap_camera, colmap_image):
    """Converts a pycolmap `image` to an SFM camera."""
    camera_rotation = colmap_image.R()
    camera_position = -(colmap_image.t @ camera_rotation)
    new_camera = Camera(
        orientation=camera_rotation,
        position=camera_position,
        focal_length=colmap_camera.fx,
        pixel_aspect_ratio=colmap_camera.fx / colmap_camera.fx,
        principal_point=np.array([colmap_camera.cx, colmap_camera.cy]),
        skew=0.0,
        image_size=np.array([colmap_camera.width, colmap_camera.height])
    )
    return new_camera


def filter_outlier_points(points, inner_percentile):
    """Filters outlier points."""
    outer = 1.0 - inner_percentile
    lower = outer / 2.0
    upper = 1.0 - lower
    centers_min = np.quantile(points, lower, axis=0)
    centers_max = np.quantile(points, upper, axis=0)
    result = points.copy()

    too_near = np.any(result < centers_min[None, :], axis=1)
    too_far = np.any(result > centers_max[None, :], axis=1)

    return result[~(too_near | too_far)]


# def average_reprojection_errors(points, pixels, cameras):
#     """Computes the average reprojection errors of the points."""
#     cam_errors = []
#     for i, camera in enumerate(cameras):
#         cam_error = reprojection_error(points, pixels[:, i], camera)
#         cam_errors.append(cam_error)
#     cam_error = np.stack(cam_errors)
#
#     return cam_error.mean(axis=1)


def _get_camera_translation(camera):
    """Computes the extrinsic translation of the camera."""
    rot_mat = camera.orientation
    return -camera.position.dot(rot_mat.T)


def _transform_camera(camera, transform_mat):
    """Transforms the camera using the given transformation matrix."""
    # The determinant gives us volumetric scaling factor.
    # Take the cube root to get the linear scaling factor.
    scale = np.cbrt(np.linalg.det(transform_mat[:, :3]))
    quat_transform = ~Quaternion.FromR(transform_mat[:, :3] / scale)

    translation = _get_camera_translation(camera)
    rot_quat = Quaternion.FromR(camera.orientation)
    rot_quat *= quat_transform
    translation = scale * translation - rot_quat.ToR().dot(transform_mat[:, 3])
    new_transform = np.eye(4)
    new_transform[:3, :3] = rot_quat.ToR()
    new_transform[:3, 3] = translation

    rotation = rot_quat.ToR()
    new_camera = camera.copy()
    new_camera.orientation = rotation
    new_camera.position = -(translation @ rotation)
    return new_camera


def _pycolmap_to_sfm_cameras(manager: pycolmap.SceneManager) -> Dict[int, Camera]:
    """Creates SFM cameras."""
    # Use the original filenames as indices.
    # This mapping necessary since COLMAP uses arbitrary numbers for the
    # image_id.
    image_id_to_colmap_id = {
        image.name.split('.')[0]: image_id
        for image_id, image in manager.images.items()
    }

    sfm_cameras = {}
    for image_id in image_id_to_colmap_id:
        colmap_id = image_id_to_colmap_id[image_id]
        image = manager.images[colmap_id]
        camera = manager.cameras[image.camera_id]
        sfm_cameras[image_id] = convert_colmap_camera(camera, image)

    return sfm_cameras


class SceneManager:
    """A thin wrapper around pycolmap."""

    @classmethod
    def from_pycolmap(cls, colmap_path, image_path, min_track_length=10):
        """Create a scene manager using pycolmap."""
        manager = pycolmap.SceneManager(str(colmap_path))
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()
        manager.filter_points3D(min_track_len=min_track_length)
        sfm_cameras = _pycolmap_to_sfm_cameras(manager)
        return cls(sfm_cameras, manager.get_filtered_points3D(), image_path)

    def __init__(self, cameras, points, image_path):
        self.image_path = image_path
        self.camera_dict = cameras
        print(len(cameras))
        self.points = points

        logging.info('Created scene manager with %d cameras', len(self.camera_dict))

    def __len__(self):
        return len(self.camera_dict)

    @property
    def image_ids(self):
        return sorted(self.camera_dict.keys())

    @property
    def camera_list(self):
        return [self.camera_dict[i] for i in self.image_ids]

    @property
    def camera_positions(self):
        """Returns an array of camera positions."""
        return np.stack([camera.position for camera in self.camera_list])

    def load_image(self, image_id):
        """Loads the image with the specified image_id."""
        path = self.image_path / f'{image_id}.png'
        with path.open('rb') as f:
            return imageio.imread(f)

    def change_basis(self, axes, center):
        """Change the basis of the scene.

    Args:
      axes: the axes of the new coordinate frame.
      center: the center of the new coordinate frame.

    Returns:
      A new SceneManager with transformed points and cameras.
    """
        transform_mat = np.zeros((3, 4))
        transform_mat[:3, :3] = axes.T
        transform_mat[:, 3] = -(center @ axes)
        return self.transform(transform_mat)

    def transform(self, transform_mat):
        """Transform the scene using a transformation matrix.

    Args:
      transform_mat: a 3x4 transformation matrix representation a
        transformation.

    Returns:
      A new SceneManager with transformed points and cameras.
    """
        if transform_mat.shape != (3, 4):
            raise ValueError('transform_mat should be a 3x4 transformation matrix.')

        points = None
        if self.points is not None:
            points = self.points.copy()
            points = points @ transform_mat[:, :3].T + transform_mat[:, 3]

        new_cameras = {}
        for image_id, camera in self.camera_dict.items():
            new_cameras[image_id] = _transform_camera(camera, transform_mat)

        return SceneManager(new_cameras, points, self.image_path)

    def filter_images(self, image_ids):
        num_filtered = 0
        for image_id in image_ids:
            if self.camera_dict.pop(image_id, None) is not None:
                num_filtered += 1

        return num_filtered


# @title Load COLMAP scene.
import plotly.graph_objs as go
import os

scene_manager = SceneManager.from_pycolmap(
    os.path.join(root_dir, 'sparse/0'),
    os.path.join(root_dir, 'images'),
    min_track_length=5)

### scene
# @title Compute near/far planes.
import pandas as pd


def estimate_near_far_for_image(scene_manager, image_id):
    """Estimate near/far plane for a single image based via point cloud."""
    points = filter_outlier_points(scene_manager.points, 0.95)
    points = np.concatenate([
        points,
        scene_manager.camera_positions,
    ], axis=0)
    camera = scene_manager.camera_dict[image_id]
    pixels = camera.project(points)
    depths = camera.points_to_local_points(points)[..., 2]

    # in_frustum = camera.ArePixelsInFrustum(pixels)
    in_frustum = (
            (pixels[..., 0] >= 0.0)
            & (pixels[..., 0] <= camera.image_size_x)
            & (pixels[..., 1] >= 0.0)
            & (pixels[..., 1] <= camera.image_size_y))
    depths = depths[in_frustum]

    in_front_of_camera = depths > 0
    depths = depths[in_front_of_camera]

    near = np.quantile(depths, 0.001)
    far = np.quantile(depths, 0.999)

    return near, far


def estimate_near_far(scene_manager):
    """Estimate near/far plane for a set of randomly-chosen images."""
    # image_ids = sorted(scene_manager.images.keys())
    image_ids = scene_manager.image_ids
    rng = np.random.RandomState(0)
    image_ids = rng.choice(
        image_ids, size=len(scene_manager.camera_list), replace=False)

    result = []
    for image_id in image_ids:
        near, far = estimate_near_far_for_image(scene_manager, image_id)
        result.append({'image_id': image_id, 'near': near, 'far': far})
    result = pd.DataFrame.from_records(result)
    return result


near_far = estimate_near_far(scene_manager)
print('Statistics for near/far computation:')
print(near_far.describe())
print()

near = near_far['near'].quantile(0.001) / 0.8
far = near_far['far'].quantile(0.999) * 1.2
print('Selected near/far values:')
print(f'Near = {near:.04f}')
print(f'Far = {far:.04f}')


# @title Compute scene center and scale.

def get_bbox_corners(points):
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return np.stack([lower, upper])


points = filter_outlier_points(scene_manager.points, 0.95)
bbox_corners = get_bbox_corners(
    np.concatenate([points, scene_manager.camera_positions], axis=0))

scene_center = np.mean(bbox_corners, axis=0)
scene_scale = 1.0 / np.sqrt(np.sum((bbox_corners[1] - bbox_corners[0]) ** 2))

print(f'Scene Center: {scene_center}')
print(f'Scene Scale: {scene_scale}')

# @title Define Utilities.
_EPSILON = 1e-5


def points_bound(points):
    """Computes the min and max dims of the points."""
    min_dim = np.min(points, axis=0)
    max_dim = np.max(points, axis=0)
    return np.stack((min_dim, max_dim), axis=1)


def points_centroid(points):
    """Computes the centroid of the points from the bounding box."""
    return points_bound(points).mean(axis=1)


def points_bounding_size(points):
    """Computes the bounding size of the points from the bounding box."""
    bounds = points_bound(points)
    return np.linalg.norm(bounds[:, 1] - bounds[:, 0])


def look_at(camera,
            camera_position: np.ndarray,
            look_at_position: np.ndarray,
            up_vector: np.ndarray):
    look_at_camera = camera.copy()
    optical_axis = look_at_position - camera_position
    norm = np.linalg.norm(optical_axis)
    if norm < _EPSILON:
        raise ValueError('The camera center and look at position are too close.')
    optical_axis /= norm

    right_vector = np.cross(optical_axis, up_vector)
    norm = np.linalg.norm(right_vector)
    if norm < _EPSILON:
        raise ValueError('The up-vector is parallel to the optical axis.')
    right_vector /= norm

    # The three directions here are orthogonal to each other and form a right
    # handed coordinate system.
    camera_rotation = np.identity(3)
    camera_rotation[0, :] = right_vector
    camera_rotation[1, :] = np.cross(optical_axis, right_vector)
    camera_rotation[2, :] = optical_axis

    look_at_camera.position = camera_position
    look_at_camera.orientation = camera_rotation
    return look_at_camera


import math
from scipy import interpolate
from plotly.offline import iplot
import plotly.graph_objs as go


def compute_camera_rays(points, camera):
    origins = np.broadcast_to(camera.position[None, :], (points.shape[0], 3))
    directions = camera.pixels_to_rays(points.astype(np.float32))
    endpoints = origins + directions
    return origins, endpoints


from tensorflow_graphics.geometry.representation.ray import triangulate as ray_triangulate


def triangulate_rays(origins, directions):
    origins = origins[np.newaxis, ...].astype('float32')
    directions = directions[np.newaxis, ...].astype('float32')
    weights = np.ones(origins.shape[:2], dtype=np.float32)
    points = np.array(ray_triangulate(origins, origins + directions, weights))
    return points.squeeze()


ref_cameras = [c for c in scene_manager.camera_list]
origins = np.array([c.position for c in ref_cameras])
directions = np.array([c.optical_axis for c in ref_cameras])
look_at = triangulate_rays(origins, directions)
print('look_at', look_at)

avg_position = np.mean(origins, axis=0)
print('avg_position', avg_position)

up = -np.mean([c.orientation[..., 1] for c in ref_cameras], axis=0)
print('up', up)

bounding_size = points_bounding_size(origins) / 2
x_scale = 0.75  # @param {type: 'number'}
y_scale = 0.75  # @param {type: 'number'}
xs = x_scale * bounding_size
ys = y_scale * bounding_size
radius = 0.75  # @param {type: 'number'}
num_frames = 100  # @param {type: 'number'}

origin = np.zeros(3)

ref_camera = ref_cameras[0]
print(ref_camera.position)
z_offset = -0.1

angles = np.linspace(0, 2 * math.pi, num=num_frames)
positions = []
for angle in angles:
    x = np.cos(angle) * radius * xs
    y = np.sin(angle) * radius * ys
    # x = xs * radius * np.cos(angle) / (1 + np.sin(angle) ** 2)
    # y = ys * radius * np.sin(angle) * np.cos(angle) / (1 + np.sin(angle) ** 2)

    position = np.array([x, y, z_offset])
    # Make distance to reference point constant.
    position = avg_position + position
    positions.append(position)

positions = np.stack(positions)

orbit_cameras = []
for position in positions:
    camera = ref_camera.look_at(position, look_at, up)
    orbit_cameras.append(camera)

camera_paths = {'orbit-mild': orbit_cameras}

# @title Save scene information to `scene.json`.
from pprint import pprint
import json

scene_json_path = os.path.join(root_dir, 'scene.json')
with open(scene_json_path, 'w+') as f:
    json.dump({
        'scale': scene_scale,
        'center': scene_center.tolist(),
        'bbox': bbox_corners.tolist(),
        'near': near * scene_scale,
        'far': far * scene_scale,
    }, f, indent=2)

print(f'Saved scene information to {scene_json_path}')

# @title Save dataset split to `dataset.json`.

all_ids = scene_manager.image_ids
val_ids = all_ids[::20]
train_ids = sorted(set(all_ids) - set(val_ids))
dataset_json = {
    'count': len(scene_manager),
    'num_exemplars': len(train_ids),
    'ids': scene_manager.image_ids,
    'train_ids': train_ids,
    'val_ids': val_ids,
}

dataset_json_path = os.path.join(root_dir, 'dataset.json')
with open(dataset_json_path, 'w+') as f:
    json.dump(dataset_json, f, indent=2)

print(f'Saved dataset information to {dataset_json_path}')

# @title Save metadata information to `metadata.json`.
import bisect

metadata_json = {}
for i, image_id in enumerate(train_ids):
    metadata_json[image_id] = {
        'warp_id': i,
        'appearance_id': i,
        'camera_id': 0,
    }
for i, image_id in enumerate(val_ids):
    i = bisect.bisect_left(train_ids, image_id)
    metadata_json[image_id] = {
        'warp_id': i,
        'appearance_id': i,
        'camera_id': 0,
    }

metadata_json_path = os.path.join(root_dir, 'metadata.json')
with open(metadata_json_path, 'w+') as f:
    json.dump(metadata_json, f, indent=2)

print(f'Saved metadata information to {metadata_json_path}')

# @title Save cameras.
camera_dir = os.path.join(root_dir, 'camera')
if not os.path.exists(camera_dir):
    os.mkdir(camera_dir)
for item_id, camera in scene_manager.camera_dict.items():
    camera_path = os.path.join(camera_dir, f'{item_id}.json')
    print(f'Saving camera to {camera_path!s}')
    with open(camera_path, 'w+') as f:
        json.dump(camera.to_json(), f, indent=2)

# @title Save test cameras.

import json

test_camera_dir = os.path.join(root_dir, 'camera-paths')
if not os.path.exists(test_camera_dir):
    os.mkdir(test_camera_dir)
for test_path_name, test_cameras in camera_paths.items():
    out_dir = os.path.join(test_camera_dir, test_path_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, camera in enumerate(test_cameras):
        camera_path = os.path.join(out_dir, f'{i:06d}.json')
        print(f'Saving camera to {camera_path!s}')
        with open(camera_path, 'w+') as f:
            json.dump(camera.to_json(), f, indent=2)
