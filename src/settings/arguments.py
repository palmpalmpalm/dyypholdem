import os
import sys

import torch

from utils.timer import Timer
import utils.pseudo_random as pseudo_random
import utils.output as output


"""Section Data Management and Paths"""
# the directory for data files
data_directory = '../data/'
# folders for data per street
street_folders = {0: "preflop-aux", 1: "preflop/", 2: "flop/", 3: "turn/", 4: "river/"}
# names of streets for printing
street_names = {1: "Pre-flop", 2: "Flop", 3: "Turn", 4: "River"}
# path to the solved poker situation data used to train the neural net
training_data_path = '../../data/training_samples/'
# path to the models during training
training_model_path = '../../data/models/'
# folder for raw training files
training_data_raw = 'raw/'
# folder for converted / bucketed training files
training_data_converted = 'bucketed/'
# file patter for input files
inputs_extension = ".inputs"
# file extension for targets files
targets_extension = ".targets"
# path to the neural net models
model_path = '../data/models/'
# the name of a neural net file
value_net_name = 'final'
# the extension of a neural net file
value_net_extension = '.tar'
# flag whether to use sqlite database for bucketing information (otherwise use data files)
use_sqlite = False


"""Section CFR Iterations"""
# the number of iterations that DyypHoldem runs CFR for
def _positive_int_env(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value < 1:
        raise RuntimeError(f"{name} must be at least 1")
    return value


cfr_iters = _positive_int_env("DYYPHOLDEM_CFR_ITERS", 2000)
# the number of preliminary CFR iterations which DyypHoldem doesn't factor into the average strategy (included in cfr_iters)
cfr_skip_iters = _positive_int_env("DYYPHOLDEM_CFR_SKIP_ITERS", cfr_iters // 2)
if cfr_skip_iters >= cfr_iters:
    raise RuntimeError(
        "DYYPHOLDEM_CFR_SKIP_ITERS must be smaller than DYYPHOLDEM_CFR_ITERS"
    )


"""Section Data Generation"""
# how many poker situations are solved simultaneously during data generation
gen_batch_size = 10
# how many solved poker situations are generated for use as training examples
gen_data_count = 100000


"""Section Training"""
# how many poker situations are used in each neural net training batch - has to be a multiple of gen_batch_size !
train_batch_size = 1000
# how many epochs to train for
epoch_count = 200
# how often to save the model during training
save_epoch = 10
# automatically save best epoch as final model
save_best_epoch = True
# learning rate for neural net training
learning_rate = 0.001
# resume training if a final model already exists
resume_training = False


"""Section Torch"""
# flag to use GPU for calculations
runtime_device = os.environ.get("DYYPHOLDEM_DEVICE", "cuda").strip().lower()
if runtime_device not in ("cuda", "cpu"):
    raise RuntimeError("DYYPHOLDEM_DEVICE must be cuda or cpu")
use_gpu = runtime_device == "cuda"
# CUDA Graph replay is the default after four bitwise RTX 4090 gates on every
# public node (2026-09-02) and 5,425 graphed decisions across four concurrent
# sessions with no error. "auto" falls back before computation when a solve is
# ineligible, so a CPU host or an unsupported build runs the eager loop;
# "required" fails instead of silently using it, and "off" restores it.
cuda_graph_mode = os.environ.get(
    "DYYPHOLDEM_CUDA_GRAPHS", "auto"
).strip().lower()
if cuda_graph_mode not in ("off", "auto", "required"):
    raise RuntimeError(
        "DYYPHOLDEM_CUDA_GRAPHS must be off, auto, or required"
    )
cuda_graph_eager_warmups = 3
# Regret update variant. "cfr+" is the legacy loop. "dcfr" additionally
# discounts accumulated (non-negative) regrets after iteration t by
# t^alpha / (t^alpha + 1) with alpha = 1.5 (Brown & Sandholm 2019), including
# the CFR-D gadget regrets; averaging keeps the skip-based uniform window.
cfr_variant = os.environ.get("DYYPHOLDEM_CFR_VARIANT", "cfr+").strip().lower()
if cfr_variant not in ("cfr+", "dcfr"):
    raise RuntimeError("DYYPHOLDEM_CFR_VARIANT must be cfr+ or dcfr")
dcfr_alpha = 1.5
# default tensor types
if not use_gpu:
    Tensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    value_net_name = value_net_name + "_cpu"
else:
    Tensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    value_net_name = value_net_name + "_gpu"
    # flag to use new tensor cores on Ampere based GPUs - set to 'False' for reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
# device name for torch
device = torch.device('cpu') if not use_gpu else torch.device('cuda')


"""Section Random"""
# flag to choose between official or internal random number generator - set to 'True' for reproducibility
use_pseudo_random = False
if use_pseudo_random:
    pseudo_random.manual_seed(123)


"""Section Global Objects"""
# global logger
use_loguru = True
if use_loguru:
    try:
        import loguru
    except ImportError:
        use_loguru = False

if use_loguru:
    logger = loguru.logger
    logger.remove(0)
    logger.level("LOADING", no=8, color="<fg #944100><bold>", icon="@")
    logger.level("TIMING", no=15, color="<fg #b88210><bold>", icon="@")
    logger.level("TRACE", color="<fg #717171><bold>")
    log_format_stderr = "| <level>{level: <8}</level> | <level>{message}</level>"
    log_format_file = "{time:YYYY-MM-DD HH:mm:ss.SS} | {level: <8} | {message}"
    logging_level_stderr = "TRACE"
    logging_level_file = "TRACE"
    logger.add(sys.stderr, format=log_format_stderr, level=logging_level_stderr)
    logger.add("../../logs/dyypholdem.log", format=log_format_file, level=logging_level_file, rotation="10 MB")
else:
    logger = output.DummyLogger("TRACE")

# a global timer used to measure loading and calculation times
timer = Timer(logger)

logger.info("Environment setup complete - initializing...")
