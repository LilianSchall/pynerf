import os
import torch
import numpy as np
from torch.utils import checkpoint
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import imageio

from nerf.models import Model
from nerf.datasets import BlenderDataset
from nerf.render import Renderer


class TrainSession:
    chunk: int
    save_path: str

    batch_size: int
    use_batching: bool
    learning_rate_decay: int
    precrop_iters: int
    precrop_frac: float
    checkpoint_freq: int

    def __init__(
        self,
        chunk: int,
        base_dir: str,
        experiment_name: str,
        render_factor: int,
        batch_size: int,
        n_iters: int,
        use_batching: bool,
        learning_rate_decay: int,
        precrop_iters: int,
        precrop_frac: float,
        checkpoint_freq: int,
    ) -> None:
        self.chunk = chunk
        self.save_path = os.path.join(base_dir, experiment_name, "rendering")
        os.makedirs(self.save_path, exist_ok=True)
        self.render_factor = render_factor
        self.batch_size = batch_size
        self.use_batching = use_batching
        self.n_iters = n_iters
        self.learning_rate_decay = learning_rate_decay
        self.precrop_iters = precrop_iters
        self.precrop_frac = precrop_frac
        self.checkpoint_freq = checkpoint_freq

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

    def new_data_per_epoch(
        self,
        K: np.ndarray,
        renderer: Renderer,
        dataset: BlenderDataset,
        images: torch.Tensor,
        rays_rgb: torch.Tensor | None,
        i_batch: int,
        i: int,
        start: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.use_batching and rays_rgb is not None:
            batch: torch.Tensor = rays_rgb[i_batch : i_batch + self.batch_size]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += self.batch_size

            if i_batch >= rays_rgb.shape[0]:
                rand_idx = torch.randperm(images.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else:
            index: torch.Tensor = torch.randint(images.shape[0], size=(1,), device="cpu")
            target: torch.Tensor = images[index]
            target = target.squeeze(0).to(device)

            pose: torch.Tensor = dataset.poses[index, :3, :4].squeeze(0)

            rays_o, rays_d = renderer.ray_generator.torch_rays(
                H=dataset.H, W=dataset.W, K=K, c2w=pose
            )

            if i < self.precrop_iters:
                dH = int(dataset.H // 2 * self.precrop_frac)
                dW = int(dataset.W // 2 * self.precrop_frac)

                coords: torch.Tensor = torch.stack(
                    torch.meshgrid(
                        torch.linspace(
                            dataset.H // 2 - dH, dataset.H // 2 + dH - 1, 2 * dH
                        ),
                        torch.linspace(
                            dataset.W // 2 - dW, dataset.W // 2 + dW - 1, 2 * dW
                        ),
                    ),
                    -1,
                )

                if i == start:
                    print(
                        f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {self.precrop_iters}"
                    )
            else:
                coords: torch.Tensor = torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, dataset.H - 1, dataset.H),
                        torch.linspace(0, dataset.W - 1, dataset.W),
                    ),
                    -1,
                )

            coords = torch.reshape(coords, [-1, 2])
            select_inds: torch.Tensor = torch.randint(
                coords.shape[0], size=(self.batch_size,)
            )

            select_coords: torch.Tensor = coords[select_inds].long()

            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
            batch_rays: torch.Tensor = torch.stack([rays_o, rays_d], 0)
            target_s: torch.Tensor = target[select_coords[:, 0], select_coords[:, 1]]

        return batch_rays, target_s, i_batch

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
        i_batch: int = 0

        for i in trange(model.start, self.n_iters):
            batch_rays, target_s, i_batch = self.new_data_per_epoch(
                K,
                renderer,
                dataset,
                images,
                rays_rgb,
                i_batch,
                i,
                model.start,
                model.device,
            )
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
            new_learning_rate: float = model.learning_rate * (
                decay_rate ** (i / decay_steps)
            )
            for param_group in model.optimizer.param_groups:
                param_group["lr"] = new_learning_rate

            if i % self.checkpoint_freq == 0:
                path = os.path.join(self.save_path, f"model_{i}.tar")
                torch.save(
                    {
                        "global_step": i,
                        "network_fn_state_dict": model.model.state_dict(),
                        "network_fine_state_dict": (
                            model.model_fine.state_dict()
                            if model.model_fine is not None
                            else None
                        ),
                        "optimizer_state_dict": model.optimizer.state_dict(),
                    },
                    path,
                )

            tqdm.write(f"[TRAINING] Iter: {i} Loss: {img_loss.item()}")
