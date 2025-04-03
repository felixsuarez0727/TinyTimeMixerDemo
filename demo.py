import matplotlib.pyplot as plt
import pandas as pd
import os

from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TinyTimeMixerForPrediction,
)
from tsfm_public.toolkit.visualization import plot_predictions


# Obtain directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the path to the data file
DATA_FILE_PATH = os.path.join(BASE_DIR, "archive", "energy_dataset.csv")


timestamp_column = "time"
target_columns = ["total load actual"]
context_length = 512

print("📌 Loading data...")
input_df = pd.read_csv(
  DATA_FILE_PATH,
  parse_dates=[timestamp_column],
)

print("📌 Filling missing data...")
input_df = input_df.ffill()
input_df = input_df.iloc[-context_length:,]

print("📌 Showing last rows in dataset...")
print(input_df.tail())

print("📌 Creating charts...")
fig, axs = plt.subplots(len(target_columns), 1, figsize=(10, 2 * len(target_columns)), squeeze=False)
for ax, target_column in zip(axs, target_columns):
    ax[0].plot(input_df[timestamp_column], input_df[target_column])
plt.show()  # Shows chart

print("📌 Loading model...")
zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
  "ibm-granite/granite-timeseries-ttm-r1", #you may switch model here: ibm-granite/granite-timeseries-ttm-r2
  num_input_channels=len(target_columns)
)
print("✅ Loaded model")

print("📌 Creating pipeline...")
pipeline = TimeSeriesForecastingPipeline(
    zeroshot_model,
    timestamp_column=timestamp_column,
    id_columns=[],
    target_columns=target_columns,
    explode_forecasts=True,
    freq="h",
    device="cpu",
)
print("✅ Pipeline done!")

print("📌 Doing forecast...")
zeroshot_forecast = pipeline(input_df)

print("📌 Last rows of forecast:")
print(zeroshot_forecast.tail())

print("📌 Showing forecast charts...")
plot_predictions(
    input_df=input_df,
    exploded_predictions_df=zeroshot_forecast,
    freq="h",
    timestamp_column=timestamp_column,
    channel=target_columns[0],
    indices=[-1],
    num_plots=1,
)
plt.show()  # Shows forecast prediction