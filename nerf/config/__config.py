import configargparse


class Config:
    train: bool

    base_dir: str
    exp_name: str
    multires: int
    net_depth: int
    net_width: int
    use_viewdirs: bool
    learning_rate: float
    multires_views: int
    n_importance: int
    net_depth_fine: int
    net_width_fine: int
    perturb: float
    n_samples: int
    white_bkgd: bool
    raw_noise_std: float
    lindisp: bool
    chunk: int
    render_factor: int

    data_dir: str
    half_res: bool

    use_batching: bool
    batch_size: int
    n_iters: int
    learning_rate_decay: int
    checkpoint_freq: int
    precrop_frac: float
    precrop_iters: int

    def __init__(self):
        parser = configargparse.ArgumentParser()

        parser = configargparse.ArgumentParser()
        parser.add_argument("--config", is_config_file=True, help="config file path")
        parser.add_argument("--exp_name", type=str, help="experiment name")
        parser.add_argument(
            "--base_dir",
            type=str,
            default="./logs/",
            help="where to store ckpts and logs",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default="./data/llff/fern",
            help="input data directory",
        )

        # training options
        parser.add_argument(
            "--net_depth", type=int, default=8, help="layers in network"
        )
        parser.add_argument(
            "--net_width", type=int, default=256, help="channels per layer"
        )
        parser.add_argument(
            "--net_depth_fine", type=int, default=8, help="layers in fine network"
        )
        parser.add_argument(
            "--net_width_fine",
            type=int,
            default=256,
            help="channels per layer in fine network",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32 * 32 * 4,
            help="batch size (number of random rays per gradient step)",
        )
        parser.add_argument(
            "--learning_rate", type=float, default=5e-4, help="learning rate"
        )
        parser.add_argument(
            "--learning_rate_decay",
            type=int,
            default=250,
            help="exponential learning rate decay (in 1000 steps)",
        )
        parser.add_argument(
            "--chunk",
            type=int,
            default=1024 * 32,
            help="number of rays processed in parallel, decrease if running out of memory",
        )
        parser.add_argument(
            "--netchunk",
            type=int,
            default=1024 * 64,
            help="number of pts sent through network in parallel, decrease if running out of memory",
        )
        parser.add_argument(
            "--no_batching",
            action="store_true",
            help="only take random rays from 1 image at a time",
        )
        # rendering options
        parser.add_argument(
            "--n_samples", type=int, default=64, help="number of coarse samples per ray"
        )
        parser.add_argument(
            "--n_importance",
            type=int,
            default=0,
            help="number of additional fine samples per ray",
        )
        parser.add_argument(
            "--perturb",
            type=float,
            default=1.0,
            help="set to 0. for no jitter, 1. for jitter",
        )
        parser.add_argument(
            "--use_viewdirs",
            action="store_true",
            help="use full 5D input instead of 3D",
        )
        parser.add_argument(
            "--multires",
            type=int,
            default=10,
            help="log2 of max freq for positional encoding (3D location)",
        )
        parser.add_argument(
            "--multires_views",
            type=int,
            default=4,
            help="log2 of max freq for positional encoding (2D direction)",
        )
        parser.add_argument(
            "--raw_noise_std",
            type=float,
            default=0.0,
            help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
        )
        parser.add_argument(
            "--render_factor",
            type=int,
            default=0,
            help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
        )

        # training options
        parser.add_argument(
            "--precrop_iters",
            type=int,
            default=0,
            help="number of steps to train on central crops",
        )
        parser.add_argument(
            "--precrop_frac",
            type=float,
            default=0.5,
            help="fraction of img taken for central crops",
        )

        # dataset options
        parser.add_argument(
            "--dataset_type",
            type=str,
            default="llff",
            help="options: llff / blender / deepvoxels",
        )

        ## blender flags
        parser.add_argument(
            "--white_bkgd",
            action="store_true",
            help="set to render synthetic data on a white bkgd (always use for dvoxels)",
        )
        parser.add_argument(
            "--half_res",
            action="store_true",
            help="load blender synthetic data at 400x400 instead of 800x800",
        )

        ## llff flags
        parser.add_argument(
            "--factor", type=int, default=8, help="downsample factor for LLFF images"
        )
        parser.add_argument(
            "--no_ndc",
            action="store_true",
            help="do not use normalized device coordinates (set for non-forward facing scenes)",
        )
        parser.add_argument(
            "--lindisp",
            action="store_true",
            help="sampling linearly in disparity rather than depth",
        )
        parser.add_argument(
            "--spherify", action="store_true", help="set for spherical 360 scenes"
        )
        parser.add_argument(
            "--llffhold",
            type=int,
            default=8,
            help="will take every 1/N images as LLFF test set, paper uses 8",
        )

        # logging/saving options
        parser.add_argument(
            "--i_print",
            type=int,
            default=100,
            help="frequency of console printout and metric loggin",
        )
        parser.add_argument(
            "--checkpoint_freq",
            type=int,
            default=10000,
            help="frequency of weight ckpt saving",
        )
        parser.add_argument(
            "--i_testset", type=int, default=50000, help="frequency of testset saving"
        )
        parser.add_argument(
            "--i_video",
            type=int,
            default=50000,
            help="frequency of render_poses video saving",
        )

        parser.add_argument(
            "--train", action="store_true", help="Launch training session"
        )

        args = parser.parse_args()

        self.train = args.train

        # data config
        self.base_dir = args.base_dir
        self.exp_name = args.exp_name
        self.data_dir = args.data_dir
        self.half_res = args.half_res

        # model config
        self.multires = args.multires
        self.net_depth = args.net_depth
        self.net_width = args.net_width
        self.use_viewdirs = args.use_viewdirs
        self.learning_rate = args.learning_rate
        self.multires_views = args.multires_views
        self.n_importance = args.n_importance
        self.net_depth_fine = args.net_depth_fine
        self.net_width_fine = args.net_width_fine
        self.perturb = args.perturb
        self.n_samples = args.n_samples
        self.white_bkgd = args.white_bkgd
        self.raw_noise_std = args.raw_noise_std
        self.lindisp = args.lindisp

        # session config
        self.chunk = args.chunk
        self.render_factor = args.render_factor

        self.batch_size = args.batch_size
        self.n_iters = 200_000 + 1
        self.use_batching = not args.no_batching
        self.learning_rate_decay = args.learning_rate_decay
        self.checkpoint_freq = args.checkpoint_freq
        self.precrop_frac = args.precrop_frac
        self.precrop_iters = args.precrop_iters
