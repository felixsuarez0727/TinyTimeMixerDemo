# Time Series Forecasting Demo with TinyTimeMixer

This project demonstrates time series forecasting using the `tsfm_public` library and the TinyTimeMixer model. It loads energy consumption data, performs preprocessing, generates forecasts, and visualizes the results.

## What is TinyTimeMixer?
It is one of the many pretrained models from the family of "Granite Time Series". This models are developed by IBM.
This models performs on principles of zero/few-shot learning, hence, it is used to provide predictions even with limited training data. This model is lightweight meaning it could run with minimial hardware requirements.
This model is useful in scenarios where series data is scarce or diverse.

[Further details here!](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1)

[Check this academic paper!](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1)

## Prerequisites

Before running the code, ensure you have the following installed:

-   Python 3.x
-   `venv` (virtual environment module)
-   `pip` (Python package installer)

## Setup

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**

    -   On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

    -   On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

3.  **Install the necessary dependencies:**

    ```bash
    pip install "tsfm_public[notebooks] @ git+[https://github.com/ibm-granite/granite-tsfm.git@v0.2.12](https://github.com/ibm-granite/granite-tsfm.git@v0.2.12)"
    ```

## Running the Demo

**Execute the `demo.py` script:**

    ```bash
    python demo.py
    ```
This script will:

  -   Load the energy consumption dataset (`energy_dataset.csv`).
  -   Fill missing data using forward fill.
  -   Display the last few rows of the preprocessed data.
  -   Generate a plot of the target time series.
  -   Load the TinyTimeMixer forecasting model (`ibm-granite/granite-timeseries-ttm-r1`).
  -   Create a forecasting pipeline.
  -   Generate forecasts for the target time series.
  -   Display the last few rows of the forecast.
  -   Generate and display a plot comparing the actual data with the forecast.

## Showing chart made with loaded data
The data loaded is used to build a chart.
![Data Preview](images/preview.jpg)

## Showing forecast chart
Once the data is loaded and the forescast is done base on the mentioned data, the new chart is generated to show the forecast data.
![Forecast chart view](images/forecast.jpg)


## Dataset

The dataset used in this demo is `energy_dataset.csv`, which contains hourly energy consumption data. It should be placed in the `archive` directory within the project root.

## Dependencies

-   `matplotlib`
-   `pandas`
-   `tsfm_public` (from `git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.12`)

## Notes

-   You can experiment with different TinyTimeMixer models by changing the model name in the `from_pretrained` function.
-   The `context_length` parameter controls the number of historical data points used for forecasting.
-   Feel free to modify the script to explore different forecasting scenarios and datasets.