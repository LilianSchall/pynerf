import torch
import numpy as np

class Renderer:
    def __init__(self):
        pass

    def render(
        self,
        H: int,
        W: int,
        K: np.ndarray,
        chunk: int=1024*32,
        rays: torch.Tensor | None = None,
        c2w: torch.Tensor | None = None,
        ndc:bool=True,
        near:float=0.,
        far:float=1.,
        use_viewdirs:bool=False,
        c2w_staticcam=None,
    ) -> None:
        pass


    def render_path(
        self,
        render_poses: list[torch.Tensor],
        H: int,
        W: int,
        focal: float,
        K: np.ndarray,
        chunk: int,
        save_dir: str | None = None,
        render_factor: int = 0,
    ) -> None:
        pass

