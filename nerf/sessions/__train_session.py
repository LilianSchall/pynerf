import os
import torch
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import imageio

from nerf.models import Model
from nerf.datasets import BlenderDataset
from nerf.render import Renderer


class TrainSession:
    chunk: int
    save_path: str

    n_rand: int
    use_batching: bool
    learning_rate_decay: float

    def __init__(
        self,
        chunk: int,
        base_dir: str,
        experiment_name: str,
        render_factor: int,
        n_rand: int,
        n_iters: int,
        use_batching: bool,
        learning_rate_decay: float,
    ) -> None:
        self.chunk = chunk
        self.save_path = os.path.join(base_dir, experiment_name, "rendering")
        os.makedirs(self.save_path, exist_ok=True)
        self.render_factor = render_factor
        self.n_rand = n_rand
        self.use_batching = use_batching
        self.n_iters = n_iters
        self.learning_rate_decay = learning_rate_decay

    def prepare_data(
        self, K: np.ndarray, dataset: BlenderDataset, renderer: Renderer, device: str
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rays_rgb: torch.Tensor | None = None
        if self.use_batching:
            rays: torch.Tensor = torch.stack(
                [
                    torch.stack(
                        renderer.ray_generator.torch_rays(
                            H=dataset.H, W=dataset.W, K=K, c2w=pose
                        ),
                        dim=0,
                    )
                    for pose in dataset.poses[:, :3, :4]
                ],
                dim=0,
            ).to(device)
            # TODO: check dimensions of dataset.images
            rays_rgb = torch.concatenate([rays, dataset.images[None, :]], dim=1)
            rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)
            rays_rgb = rays_rgb.reshape(-1, 3, 3)
            idx: torch.Tensor = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[idx]
            rays_rgb = rays_rgb.to(device)

            images: torch.Tensor = dataset.images.to(device)
        else:
            images: torch.Tensor = dataset.images

        return images, rays_rgb
    
    def new_data_per_epoch(self, images: torch.Tensor, rays_rgb: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def img2mse(self, img: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((img - target) ** 2)

    def run(self, model: Model, dataset: BlenderDataset, renderer: Renderer):
        K: np.ndarray = np.array(
            [
                [dataset.focal, 0, dataset.W // 2],
                [0, dataset.focal, dataset.H // 2],
                [0, 0, 1],
            ]
        )
        model.train()

        images, rays_rgb = self.prepare_data(K, dataset, renderer, model.device)

        for i in trange(model.start, self.n_iters):
            batch_rays, target_s = self.new_data_per_epoch(images, rays_rgb)
            return_list, extras = renderer.render(
                H=dataset.H,
                W=dataset.W,
                K=K,
                rays=batch_rays,
                retraw=True,
                with_ndc=dataset.needs_ndc,
                near=dataset.near,
                far=dataset.far,
            )

            rgb: torch.Tensor = return_list[0]

            model.optimizer.zero_grad()
            img_loss = self.img2mse(rgb, target_s)

            if "rgb0" in extras:
                img_loss0 = self.img2mse(extras["rgb0"], target_s)
                img_loss += img_loss0

            img_loss.backward()
            model.optimizer.step()

            decay_rate: float = 0.1
            decay_steps: float = self.learning_rate_decay * 1000
            new_learning_rate: float = model.learning_rate * (decay_rate ** (i / decay_steps))
            for param_group in model.optimizer.param_groups:
                param_group["lr"] = new_learning_rate
