import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import imageio

from nerf.models import Model
from nerf.datasets import BlenderDataset
from nerf.render import Renderer


class TestSession:
    chunk: int
    save_path: str

    def __init__(
        self, chunk: int, base_dir: str, experiment_name: str, render_factor: int
    ) -> None:
        self.chunk = chunk
        self.save_path = os.path.join(base_dir, experiment_name, "rendering")
        os.makedirs(self.save_path, exist_ok=True)
        self.render_factor = render_factor

    def run(self, model: Model, dataset: BlenderDataset, renderer: Renderer):
        K: np.ndarray = np.array(
            [
                [dataset.focal, 0, dataset.W // 2],
                [0, dataset.focal, dataset.H // 2],
                [0, 0, 1],
            ]
        )

        with torch.no_grad():
            model.eval()

            rgbs, _ = renderer.render_path(
                dataset.render_poses,
                H=dataset.H,
                W=dataset.W,
                focal=dataset.focal,
                K=K,
                chunk=self.chunk,
                near=dataset.near,
                far=dataset.far,
                save_dir=self.save_path,
                render_factor=self.render_factor,
            )

            imageio.mimwrite(os.path.join(self.save_path, "video.mp4"), renderer.to8b(rgbs), fps=30, quality=8)  # type: ignore
