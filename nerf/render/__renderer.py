import torch
import numpy as np
from tqdm import tqdm
import imageio
import os

from .__ray_generator import RayGenerator
from nerf.models import Model


class Renderer:
    ray_generator: RayGenerator
    model: Model

    def __init__(self: "Renderer", model: Model):
        self.ray_generator = RayGenerator()
        self.model = model
        pass

    def render(
        self: "Renderer",
        H: int,
        W: int,
        K: np.ndarray,
        rays: torch.Tensor | None = None,
        chunk: int = 1024 * 32,
        c2w: torch.Tensor | None = None,
        with_ndc: bool = False,
        near: float = 0.0,
        far: float = 1.0,
        c2w_staticcam=None,
        retraw: bool = False,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:

        if c2w is not None:
            rays_o, rays_d = self.ray_generator.torch_rays(H, W, K, c2w)
        elif rays is not None:
            rays_o, rays_d = rays
        else:
            raise ValueError("Either c2w or rays must be provided")

        viewdirs: torch.Tensor | None = None

        if self.model.use_viewdirs:
            viewdirs = rays_d
            if c2w_staticcam is not None:
                rays_o, rays_d = self.ray_generator.torch_rays(H, W, K, c2w_staticcam)

            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # type: ignore

        sh = rays_d.shape

        if with_ndc:
            rays_o, rays_d = self.ray_generator.ndc_rays(
                H, W, K[0][0], 1.0, rays_o, rays_d
            )

        rays_o: torch.Tensor = torch.reshape(rays_o, [-1, 3]).float()
        rays_d: torch.Tensor = torch.reshape(rays_d, [-1, 3]).float()

        near_tensor, far_tensor = near * torch.ones_like(
            rays_d[..., :1]
        ), far * torch.ones_like(rays_d[..., :1])

        rays = torch.cat([rays_o, rays_d, near_tensor, far_tensor], -1)

        if self.model.use_viewdirs and viewdirs is not None:
            rays = torch.cat([rays, viewdirs], -1)

        all_ret = self.batchify_rays(rays, chunk, retraw)

        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ["rgb_map", "disp_map", "acc_map"]
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

        return ret_list, ret_dict

    def batchify_rays(
        self: "Renderer", rays: torch.Tensor, chunk: int, retraw: bool
    ) -> dict[str, torch.Tensor]:
        """
        Batchify rays to avoid memory issues.
        """
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i : i + chunk], retraw)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        for k in all_ret:
            all_ret[k] = torch.cat(all_ret[k], 0)

        return all_ret

    def render_rays(
        self: "Renderer", rays: torch.Tensor, retraw: bool
    ) -> dict[str, torch.Tensor]:

        n_rays: int = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]

        viewdirs: torch.Tensor | None = rays[:, -3:] if rays.shape[-1] > 8 else None

        bounds: torch.Tensor = torch.reshape(rays[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]

        t_vals: torch.Tensor = torch.linspace(0.0, 1.0, steps=self.model.n_samples)

        if not self.model.lindisp:
            z_vals: torch.Tensor = near * (1.0 - t_vals) + far * t_vals
        else:
            z_vals: torch.Tensor = 1.0 / (
                1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals
            )

        z_vals = z_vals.expand([n_rays, self.model.n_samples])

        if self.model.perturb > 0.0:
            mids: torch.Tensor = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper: torch.Tensor = torch.cat([mids, z_vals[..., -1:]], -1)
            lower: torch.Tensor = torch.cat([z_vals[..., :1], mids], -1)
            t_rand: torch.Tensor = torch.rand(z_vals.shape)

            z_vals = lower + (upper - lower) * t_rand

        pts: torch.Tensor = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )

        raw: torch.Tensor = self.model.run(pts, viewdirs)
        rgb_map, disp_map, acc_map, weights, depth_map = self.model.raw2outputs(
            raw, z_vals, rays_d
        )

        if self.model.n_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid: torch.Tensor = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples: torch.Tensor = self.ray_generator.sample_pdf(
                z_vals_mid,
                weights[..., 1:-1],
                self.model.n_importance,
                deterministic=(self.model.perturb == 0.0),
            )
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

            raw = self.model.run(pts, viewdirs, use_fine=True)
            rgb_map, disp_map, acc_map, weights, depth_map = self.model.raw2outputs(
                raw, z_vals, rays_d
            )

        ret: dict[str, torch.Tensor] = {
            "rgb_map": rgb_map,
            "disp_map": disp_map,
            "acc_map": acc_map,
        }

        if retraw:
            ret["raw"] = raw

        if self.model.n_importance > 0:
            ret["rgb0"] = rgb_map_0  # type: ignore
            ret["disp0"] = disp_map_0  # type: ignore
            ret["acc0"] = acc_map_0  # type: ignore
            ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # type: ignore

        for k in ret:
            if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
                print(f"NaN/inf in {k}")

        return ret

    def render_path(
        self,
        render_poses: list[torch.Tensor],
        H: int,
        W: int,
        focal: float,
        K: np.ndarray,
        chunk: int,
        near: float,
        far: float,
        with_ndc: bool,
        save_dir: str | None = None,
        render_factor: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:

        if render_factor != 0:
            H = H // render_factor
            W = W // render_factor
            focal = focal / render_factor

        rgbs: list[np.ndarray] = []
        disps: list[np.ndarray] = []

        for i, c2w in enumerate(tqdm(render_poses)):
            render_list, _ = self.render(
                H,
                W,
                K,
                near=near,
                far=far,
                c2w=c2w[:3, :4],
                with_ndc=with_ndc,
                chunk=chunk,
            )
            assert len(render_list) == 3
            rgb, disp, acc = render_list

            rgbs.append(rgb.cpu().numpy())
            disps.append(disp.cpu().numpy())

            if save_dir is not None:
                rgb8 = self.to8b(rgbs[-1])
                filename = os.path.join(save_dir, f"{i:03d}.png")

                imageio.imwrite(filename, rgb8)

        return np.stack(rgbs, 0), np.stack(disps, 0)

    def to8b(self, x: np.ndarray) -> np.ndarray:
        return (255 * np.clip(x, 0, 1)).astype(np.uint8)
