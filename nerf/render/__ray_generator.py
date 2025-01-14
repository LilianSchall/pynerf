import torch
import numpy as np


class RayGenerator:
    """
    Ray generator class to generate rays in camera space and normalized device coordinates (NDC).
    """
    def __init__(self) -> None:
        pass

    def torch_rays(
        self, H: int, W: int, K: np.ndarray, c2w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays in camera space.

        Args:
            H (int): Height of the image.
            W (int): Width of the image.
            K (np.ndarray): Camera intrinsics.
            c2w (torch.Tensor): Camera-to-world matrix.
        """
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        i, j = i.t(), j.t()

        dirs: torch.Tensor = torch.stack(
            [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1
        )

        # added np.newaxis with None
        rays_d: torch.Tensor = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
        rays_o: torch.Tensor = c2w[:3, -1].expand(rays_d.shape)

        return rays_o, rays_d

    def ndc_rays(
        self,
        H: int,
        W: int,
        focal: float,
        near: float,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert rays from camera space to normalized device coordinates (NDC).
        For more explanation on what this function does: https://github.com/bmild/nerf/issues/18

        Args:
            H (int): Height of the image.
            W (int): Width of the image.
            focal (float): Focal length of the camera.
            near (float): Near plane of the camera.
            rays_o (torch.Tensor): Origin of the rays.
            rays_d (torch.Tensor): Direction of the rays.
        """
        t: torch.Tensor = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        o0: torch.Tensor = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1: torch.Tensor = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2: torch.Tensor = 1.0 + 2.0 * near / rays_o[..., 2]

        d0: torch.Tensor = (
            -1.0
            / (W / (2.0 * focal))
            * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        )
        d1: torch.Tensor = (
            -1.0
            / (H / (2.0 * focal))
            * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        )
        d2: torch.Tensor = -2.0 * near / rays_o[..., 2]

        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)

        return rays_o, rays_d
