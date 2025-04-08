import h5py
import sys
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime, timedelta
from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TinyTimeMixerForPrediction,
)

import numpy as np


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-mod", default="pulsed",
                        help="Modulation options: pulsed,fmcw,bpsk,amdsb,amssb,ask")
    parser.add_argument("-sig", default="Airborne-detection",
                        help="Signal type options: Airborne-detection,Airborne-range,Air-Ground-MTI,Ground mapping,Radar-Altimeter,Satcom,AM radio,short-range")
    parser.add_argument("-snr", type=int, default=10,
                        help="SNR: -20 to 18 in 2 step increments")
    parser.add_argument("-num", type=int, default=0,
                        help="Which sample to pick. 0 to 699")
    return parser.parse_args()

def plot_raw_signals(key, real, imag):
    # Plot and visualize the selected sample
    plt.figure(figsize=[8, 6])
    plt.plot(real, '-go', label="Real - Data")
    plt.plot(imag, '-bo', label="Imaginary - Data")
    plt.title(str(key), fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def forecast_and_plot_complex(real, imag):
    # 🔹 1. Normalization
    real_mean, real_std = np.mean(real), np.std(real)
    imag_mean, imag_std = np.mean(imag), np.std(imag)
    real_norm = (real - real_mean) / real_std
    imag_norm = (imag - imag_mean) / imag_std

    # 🔹 2. Create combined DataFrame
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i) for i in range(len(real))]
    input_df = pd.DataFrame({
        "time": timestamps,
        "real": real_norm,
        "imag": imag_norm
    })

    # 🔹 3. Load model
    print("📌 Loading model...")
    try:
        model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r2",
            num_input_channels=2  # because we're using real + imag
        )
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load the model: {e}")
        return

    # 🔹 4. Run combined pipeline
    print("📌 Running forecasting pipeline...")
    try:
        pipeline = TimeSeriesForecastingPipeline(
            model,
            timestamp_column="time",
            id_columns=[],
            target_columns=["real", "imag"],
            explode_forecasts=True,
            freq="s",
            device="cpu",
        )
        forecast = pipeline(input_df)
        print("✅ Forecasting pipeline executed.")
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        return

    # 🔹 5. Denormalize predictions
    forecast["real_prediction"] = forecast["real_prediction"] * real_std + real_mean
    forecast["imag_prediction"] = forecast["imag_prediction"] * imag_std + imag_mean

    # 🔹 6. RMSE (optional, uncomment if needed)
    # from sklearn.metrics import root_mean_squared_error
    # rmse_real = root_mean_squared_error(real, forecast["real_prediction"])
    # rmse_imag = root_mean_squared_error(imag, forecast["imag_prediction"])
    # print(f"📏 RMSE Real: {rmse_real:.4f}, RMSE Imag: {rmse_imag:.4f}")

    # 🔹 7. Visualization - Time Domain
    print("📌 Plotting time-domain predictions...")
    plt.figure(figsize=[12, 7])
    plt.plot(timestamps, real, '-go', label="Real - Data")
    plt.plot(timestamps, imag, '-bo', label="Imaginary - Data")
    plt.plot(forecast["time"], forecast["real_prediction"], '--g', label="Real - Prediction")
    plt.plot(forecast["time"], forecast["imag_prediction"], '--b', label="Imaginary - Prediction")
    plt.title("Complex Signal and its Predictions (Time Domain)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE_PATH = os.path.join(BASE_DIR, "RadComOta2.45GHz", "RadComOta2.45GHz.hdf5")

    with h5py.File(DATA_FILE_PATH, 'r') as f:
        key = (str(args.mod), str(args.sig), str(args.snr), str(args.num))
        waveform = f[str(key)][:]
        real = waveform[:128]
        imag = waveform[128:]

    plot_raw_signals(key, real, imag)

    # Call the complex forecast function
    forecast_and_plot_complex(real, imag)

if __name__ == "__main__":
    sys.exit(not main())
