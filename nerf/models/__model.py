from .__encoder import Encoder
from .__nerf import NeRF

import torch
import torch.nn.functional as F
import os
import typing as t


class Model:
    start: int
    encoder: Encoder
    encoder_views: Encoder | None
    model: NeRF
    model_fine: NeRF | None
    optimizer: torch.optim.Optimizer
    learning_rate: float
    perturb: float
    n_importance: int
    device: str
    use_viewdirs: bool
    white_bkgd: bool
    raw_noise_std: float
    n_samples: int
    lindisp: bool

    def __init__(
        self: "Model",
        device: str,
        base_dir: str,
        exp_name: str,
        multires: int,
        net_depth: int,
        net_width: int,
        use_viewdirs: bool,
        learning_rate: float,
        multires_views: int | None = None,
        n_importance: int = 0,
        net_depth_fine: int = 0,
        net_width_fine: int = 0,
        perturb: float = 1.0,
        n_samples: int = 64,
        white_bkgd: bool = False,
        raw_noise_std: float = 0.0,
        lindisp: bool = False,
    ):
        skips: list[int] = [4]
        self.start = 0

        self.encoder = Encoder(3, multires - 1, multires, True)
        self.encoder_views = None
        self.use_viewdirs = use_viewdirs
        self.n_importance = n_importance
        self.perturb = perturb
        self.n_samples = n_samples
        self.white_bkgd = white_bkgd
        self.raw_noise_std = raw_noise_std
        self.device = device
        self.lindisp = lindisp

        if use_viewdirs and multires_views is not None:
            self.encoder_views = Encoder(3, multires_views - 1, multires_views, True)

        input_channel = self.encoder.out_dim
        output_channel = 5 if n_importance > 0 else 4

        self.model = NeRF(
            D=net_depth,
            W=net_width,
            input_channel=input_channel,
            input_channel_views=(
                0 if self.encoder_views is None else self.encoder_views.out_dim
            ),
            output_channel=output_channel,
            skips=skips,
            use_viewdirs=use_viewdirs,
        ).to(device)

        grad_vars = list(self.model.parameters())

        self.model_fine: NeRF | None = None

        if self.n_importance > 0:
            self.model_fine = NeRF(
                D=net_depth_fine,
                W=net_width_fine,
                input_channel=input_channel,
                input_channel_views=(
                    0 if self.encoder_views is None else self.encoder_views.out_dim
                ),
                output_channel=output_channel,
                skips=skips,
                use_viewdirs=use_viewdirs,
            ).to(device)
            grad_vars += list(self.model_fine.parameters())

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            params=grad_vars, lr=learning_rate, betas=(0.9, 0.999)
        )

        os.makedirs(os.path.join(base_dir, exp_name), exist_ok=True)

        checkpoints = [
            os.path.join(base_dir, exp_name, f)
            for f in sorted(os.listdir(os.path.join(base_dir, exp_name)))
            if f.endswith(".tar")
        ]

        if len(checkpoints) > 0:
            print(f"Reloading checkpoint for model: {checkpoints[-1]}")
            checkpoint = torch.load(checkpoints[-1])
            self.start = checkpoint["global_step"]
            self.model.load_state_dict(checkpoint["network_fn_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.model_fine is not None:
                print(f"Reloading checkpoint for model fine: {checkpoints[-1]}")
                self.model_fine.load_state_dict(checkpoint["network_fine_state_dict"])

    def batchify(
        self: "Model", model: NeRF, encoded: torch.Tensor, netchunk: int
    ) -> torch.Tensor:
        return torch.cat(
            [
                model(encoded[i : i + netchunk])
                for i in range(0, encoded.shape[0], netchunk)
            ],
            0,
        )

    def run(
        self: "Model",
        inputs: torch.Tensor,
        viewdirs: torch.Tensor | None,
        netchunk: int | None = 1024 * 64,
        use_fine: bool = False,
    ) -> torch.Tensor:
        inputs_flat: torch.Tensor = torch.reshape(inputs, (-1, inputs.shape[-1]))

        encoded = self.encoder.encode(inputs_flat)

        if viewdirs is not None and self.encoder_views is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, (-1, input_dirs.shape[-1]))
            encoded_dirs = self.encoder_views.encode(input_dirs_flat)
            encoded = torch.cat([encoded, encoded_dirs], dim=-1)

        if use_fine and self.model_fine is not None:
            model: NeRF = self.model_fine
        else:
            model: NeRF = self.model

        if netchunk is None:
            outputs_flat: torch.Tensor = model(encoded)
        else:
            outputs_flat: torch.Tensor = self.batchify(model, encoded, netchunk)

        outputs: torch.Tensor = torch.reshape(
            outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        )

        return outputs

    def raw2outputs(
        self: "Model", raw: torch.Tensor, z_vals: torch.Tensor, rays_d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw2alpha: t.Callable = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(
            -act_fn(raw) * dists
        )

        dists: torch.Tensor = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1
        )

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb: torch.Tensor = torch.sigmoid(raw[..., :3])

        noise: torch.Tensor | float = 0.0

        if self.raw_noise_std > 0.0:
            noise = torch.randn(raw[..., 3].shape) * self.raw_noise_std

        alpha: torch.Tensor = raw2alpha(raw[..., 3] + noise, dists)

        weights: torch.Tensor = (
            alpha
            * torch.cumprod(
                torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1),
                -1,
            )[:, :-1]
        )
        rgb_map: torch.Tensor = torch.sum(weights[..., None] * rgb, -2)

        depth_map: torch.Tensor = torch.sum(weights * z_vals, -1)
        disp_map: torch.Tensor = 1.0 / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
        )

        acc_map: torch.Tensor = torch.sum(weights, -1)

        if self.white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    def eval(self) -> None:
        self.model.eval()
        if self.model_fine is not None:
            self.model_fine.eval()

    def train(self) -> None:
        self.model.train()
        if self.model_fine is not None:
            self.model_fine.train()
