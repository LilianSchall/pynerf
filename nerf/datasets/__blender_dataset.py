import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

import numpy as np

import os
import json

# TODO: see if you can requires_grad = False
__trans_t = lambda t: torch.Tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()  # type: ignore

# TODO: see if you can requires_grad = False
__rot_phi = lambda phi: torch.Tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()  # type: ignore

# TODO: see if you can requires_grad = False
__rot_theta = lambda th: torch.Tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()  # type: ignore


def pose_spherical_to_cartesian(
    theta: float, phi: float, radius: float
) -> torch.Tensor:
    """
    Convert spherical coordinates to cartesian coordinates

    Args:
        @param theta (float): azimuthal angle in degrees
        @param phi (float): polar angle in degrees
        @param radius (float): radius
    """
    camera2world: torch.Tensor = __trans_t(radius)
    camera2world = __rot_phi(phi / 180.0 * np.pi) @ camera2world
    camera2world = __rot_theta(theta / 180.0 * np.pi) @ camera2world
    # TODO: see if you can requires_grad = False
    camera2world = (
        torch.Tensor(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        ).float()
        @ camera2world
    )  # type: ignore

    return camera2world


class BlenderDataset(Dataset):
    images: torch.Tensor
    poses: torch.Tensor
    H: int
    W: int
    focal: float
    render_poses: list[torch.Tensor]
    index: int
    near: float
    far: float
    needs_ndc: bool

    def __init__(
        self, root_dir: str, dataset_type: str, white_bkgd: bool, half_res: bool = True
    ) -> None:
        self.near = 2.0
        self.far = 6.0
        self.needs_ndc = False
        with open(
            os.path.join(root_dir, f"transforms_{dataset_type}.json"), "r"
        ) as file:
            meta = json.load(file)

        images: list[torch.Tensor] = []
        poses: list[torch.Tensor] = []

        for frame in meta["frames"]:
            filename = os.path.join(root_dir, frame["file_path"] + ".png")
            images.append(read_image(filename).float() / 255.0)
            # TODO: see if you can requires_grad = False
            poses.append(torch.Tensor(frame["transform_matrix"]).float())  # type: ignore

        self.poses = torch.stack(poses, dim=0)

        self.H, self.W = images[0].shape[-2:]
        camera_angle_x = meta["camera_angle_x"]
        self.focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)

        # I use numpy instead of torch because I don't want to load this data on the GPU
        self.render_poses = [
            pose_spherical_to_cartesian(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ]

        if half_res:
            print(
                "Half resolution needed, values before are the following:"
                + f" H = {self.H}, W = {self.W}, focal = {self.focal}"
            )
            self.H //= 2
            self.W //= 2
            self.focal /= 2.0

            imgs_half_res: list[torch.Tensor] = []
            for img in images:
                img: torch.Tensor = torch.nn.functional.interpolate(
                    img.unsqueeze(0), (self.H, self.W)
                )
                imgs_half_res.append(img.squeeze(0))
            images = imgs_half_res

        for i in range(len(images)):
            if white_bkgd:
                images[i] = images[i][:3, ...] * images[i][-1:, ...] + (
                    1.0 - images[i][-1:, ...]
                )
            else:
                images[i] = images[i][:3, ...]
        self.images = torch.stack(images, dim=0)
        self.images = torch.permute(self.images, (0, 2, 3, 1))

        print(f"Shape of images: {self.images.shape}")
        print(f"H = {self.H}, W = {self.W}, focal={self.focal}")
        print(f"Shape of poses: {self.poses.shape}")
        print(f"Nb render poses: {len(self.render_poses)}")
        print(f"Shape of render pose: {self.render_poses[0].shape}")

        self.index = 0

    def __len__(self) -> int:
        return len(self.images)

    def __item__(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, float, float]]:
        t = (
            self.images[self.index],
            self.poses[self.index],
            self.render_poses[self.index],
            (self.H, self.W, self.focal),
        )
        self.index = (self.index + 1) % len(self.images)

        return t
