from .__embedder import Embedder
from .__nerf import NeRF

import torch
import os


def build_model(
    device: str,
    base_dir: str,
    exp_name: str,
    multires: int,
    net_depth: int,
    net_width: int,
    input_channel: int,
    output_channel: int,
    use_viewdirs: bool,
    learning_rate: float,
    multires_views: int | None = None,
    n_importance: int = 0,
    net_depth_fine: int = 0,
    net_width_fine: int = 0,
):
    skips: list[int] = [4]

    embedder: Embedder = Embedder(3, multires - 1, multires, True)
    embedder_views: Embedder | None = None

    if use_viewdirs and multires_views is not None:
        embedder_views = Embedder(3, multires_views - 1, multires_views, True)

    model = NeRF(
        D=net_depth,
        W=net_width,
        input_channel=input_channel,
        input_channel_views=(0 if embedder_views is None else embedder_views.out_dim),
        output_channel=output_channel,
        skips=skips,
        use_viewdirs=use_viewdirs,
    ).to(device)

    grad_vars = list(model.parameters())

    model_fine: NeRF | None = None

    if n_importance > 0:
        model_fine = NeRF(
            D=net_depth_fine,
            W=net_width_fine,
            input_channel=input_channel,
            input_channel_views=0 if embedder_views is None else embedder_views.out_dim,
            output_channel=output_channel,
            skips=skips,
            use_viewdirs=use_viewdirs,
        ).to(device)
        grad_vars += list(model_fine.parameters())

    optimizer = torch.optim.Adam(params=grad_vars, lr=learning_rate, betas=(0.9, 0.999))

    checkpoints = [
        os.path.join(base_dir, exp_name, f)
        for f in sorted(os.listdir(os.path.join(base_dir, exp_name)))
        if f.endswith(".tar")
    ]

    if len(checkpoints) > 0:
        checkpoint = torch.load(checkpoints[-1])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if model_fine is not None:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
