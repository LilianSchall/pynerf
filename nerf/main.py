from nerf.config import Config
from nerf.models import Model
from nerf.render import Renderer
import torch

def main(config: Config) -> None:
    model: Model = Model(
        device="cuda" if torch.cuda.is_available() else "cpu",
        base_dir=config.base_dir,
        exp_name=config.exp_name,
        multires=config.multires,
        net_depth=config.net_depth,
        net_width=config.net_width,
        use_viewdirs=config.use_viewdirs,
        learning_rate=config.learning_rate,
        multires_views=config.multires_views,
        n_importance=config.n_importance,
        net_depth_fine=config.net_depth_fine,
        net_width_fine=config.net_width_fine,
        perturb=config.perturb,
        n_samples=config.n_samples,
        white_bkgd=config.white_bkgd,
        raw_noise_std=config.raw_noise_std,
        lindisp=config.lindisp,
    )

    renderer: Renderer = Renderer(model=model)

if __name__ == "__main__":
    config = Config()
    main(config)
