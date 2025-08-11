import argparse

HPARAMS_REGISTRY = {}


class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


def add_arguments(parser):
    """Add command line arguments to parser"""
    parser.add_argument("--hps", type=str, default="morphomnist", help="Hyperparameter set")
    parser.add_argument("--exp_name", type=str, default="flow_matching_exp", help="Experiment name")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--bs", type=int, default=None, help="Batch size override")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--wd", type=float, default=None, help="Weight decay override")
    parser.add_argument("--eval_freq", type=int, default=None, help="Evaluation frequency override")
    parser.add_argument("--parents_x", type=str, nargs='+', default=None, help="Parent variables list")
    return parser


def setup_hparams(parser):
    """Setup hyperparameters from command line arguments"""
    args = parser.parse_args()
    
    # Get base hyperparameters
    if args.hps not in HPARAMS_REGISTRY:
        raise ValueError(f"Unknown hyperparameter set: {args.hps}")
    
    hparams = HPARAMS_REGISTRY[args.hps]
    
    # Store the hyperparameter set name for downstream use
    hparams.hps = args.hps
    
    # Override with command line arguments if provided
    if args.exp_name:
        hparams.exp_name = args.exp_name
    if args.resume:
        hparams.resume = args.resume
    if args.data_dir:
        hparams.data_dir = args.data_dir
    if args.lr is not None:
        hparams.lr = args.lr
    if args.bs is not None:
        hparams.bs = args.bs
    if args.epochs is not None:
        hparams.epochs = args.epochs
    if args.seed is not None:
        hparams.seed = args.seed
    if args.wd is not None:
        hparams.wd = args.wd
    if args.eval_freq is not None:
        hparams.eval_freq = args.eval_freq
    if args.parents_x is not None:
        hparams.parents_x = args.parents_x
        
    # Set default values for required attributes
    if not hasattr(hparams, 'resume'):
        hparams.resume = None
    if not hasattr(hparams, 'data_dir'):
        hparams.data_dir = None
    if not hasattr(hparams, 'exp_name'):
        hparams.exp_name = "flow_matching_exp"
    
    return hparams


morphomnist = Hparams()
# Flow Matching specific parameters
morphomnist.lr = 1e-4
morphomnist.bs = 32
morphomnist.wd = 0.01
morphomnist.betas = [0.9, 0.999]
morphomnist.lr_warmup_steps = 100
morphomnist.num_samples = 32

# Model architecture
morphomnist.base_channels = 32
morphomnist.time_emb_dim = 32
morphomnist.input_res = 32
morphomnist.pad = 4
# Flow matching parameters
morphomnist.num_steps = 50  # Integration steps for sampling
morphomnist.cfg_scale = 1.0  # Classifier-free guidance scale

# Data parameters
morphomnist.parents_x = ["thickness", "intensity", "digit"]
morphomnist.concat_pa = False  # Use separate format for flow matching
morphomnist.context_norm = "[-1,1]"
morphomnist.context_dim = 12

# Training parameters
morphomnist.epochs = 100
morphomnist.eval_freq = 5
morphomnist.viz_freq = 1000
morphomnist.accu_steps = 1
morphomnist.grad_clip = 1.0
morphomnist.grad_skip = 100.0

# Required for trainer initialization
morphomnist.start_epoch = 0
morphomnist.best_loss = float('inf')
morphomnist.iter = 0

# EMA and other training parameters
morphomnist.ema_rate = 0.999
morphomnist.seed = 42
morphomnist.deterministic = True

HPARAMS_REGISTRY["morphomnist"] = morphomnist