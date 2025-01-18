from nerf.config import Config
from nerf.models import Model
from nerf.render import Renderer
from nerf.sessions import TestSession
from nerf.datasets import BlenderDataset
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
    dataset: BlenderDataset = BlenderDataset(
        root_dir=config.data_dir,
        dataset_type="test",
        white_bkgd=config.white_bkgd,
        half_res=config.half_res,
    )

    if config.train:
        pass
    else:
        session: TestSession = TestSession(
            chunk=config.chunk,
            base_dir=config.base_dir,
            experiment_name=config.exp_name,
            render_factor=config.render_factor,
        )
        session.run(model=model, dataset=dataset, renderer=renderer)


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    config = Config()
    main(config)
