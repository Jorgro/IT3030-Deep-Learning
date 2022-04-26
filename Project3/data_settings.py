COLUMNS_TO_CLAMP = ["y"]
COLUMNS_TO_NORMALIZE = [
    "hydro",
    "micro",
    "thermal",
    "wind",
    "total",
    "y",
    "sys_reg",
    "flow",
    "structural_imbalance",
    "interpolation",
    "sum"
]
COLUMNS_TO_DROP = ["start_time", "river"]
AVOID_STRUCTURAL_IMBALANCE = False
ONE_HOT_ENCODE_TIME = False  # Else use Cos encoding
GAUSSIAN_NOISE=0
