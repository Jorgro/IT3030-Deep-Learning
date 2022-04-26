SEQUENCE_LENGTH = 50
LR = 0.00013
DROPOUT_RATE = 0.1
DATASET = "train_dataset.csv"
COLUMNS_TO_USE = [
    "hydro",
    "micro",
    "thermal",
    "wind",
    "total",
    "sys_reg",
    "flow",
    "time_of_week_cos",
    "time_of_year_cos",
    "time_of_day_cos",
    "time_of_hour_cos",
    "y_yesterday",
    "structural_imbalance",
    "y_prev_24h",

    "y_prev"
]

