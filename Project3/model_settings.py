SEQUENCE_LENGTH = 50
LR = 0.00013
DROPOUT_RATE = 0.15
DATASET = "train_dataset_noisy_2.csv"
TEST_DATASET = "test_dataset_4.csv"
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
    "time_of_week_sin",
    "time_of_year_sin",
    "time_of_day_sin",
    "time_of_hour_sin",
    "y_yesterday",
    "structural_imbalance",
    "y_prev_24h",
    "y_prev"
]
EPOCHS=2
