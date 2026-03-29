# Financial Time Series Forecasting using STFT + CNN
**Name:** Meera Nandita S  
**Registration Number:** TCR24CS045  
**Deployed Project Link:** https://financial-time-series-forecasting-ctptuyb2ytsfblfe84ltpy.streamlit.app/
---

## Overview

This project implements a **Pattern Recognition pipeline for Financial Time Series Forecasting**. It fetches historical stock price data, transforms it into STFT-based spectrograms, and trains a **CNN (Convolutional Neural Network)** model to predict future stock prices. The entire pipeline is presented through an interactive **Streamlit** web application.

---

## Project Structure

```
Financial Time Series Forecasting/
├── app.py            # Streamlit web application (main entry point)
├── data.py           # Data loading, normalization, and windowing
├── model.py          # Spectrogram computation and CNN model definition
└── requirements.txt  # Python dependencies
```

---

## How It Works

1. **Data Collection** — Downloads historical closing prices from Yahoo Finance for selected stocks (e.g., Reliance, TCS, Infosys).
2. **Preprocessing** — Normalizes the data using MinMaxScaler and creates sliding windows for time series input.
3. **Spectrogram Generation** — Applies Short-Time Fourier Transform (STFT) on each window to generate multi-channel spectrograms.
4. **CNN Training** — Trains a 2D CNN model on the spectrograms to predict the next day's closing price.
5. **Evaluation & Visualization** — Displays FFT spectrum, spectrogram images, training loss curves, and actual vs. predicted price plots.

---

## Visualizations Produced

| Step | Output |
|------|--------|
| 1 | Time Series Line Chart |
| 2 | FFT Frequency Spectrum |
| 3 | STFT Spectrogram |
| 4 | CNN Architecture Summary |
| 5 | Training & Validation Loss Curve |
| 6 | Actual vs. Predicted Price Plot |
| 7 | Prediction Table (first 20 entries) |
| 8 | Interpretation of frequency components |

---

## CNN Architecture

```
Input → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout(0.3) → Dense(64) → Output(1)
```

- **Optimizer:** Adam  
- **Loss Function:** Mean Squared Error (MSE)  
- **Early Stopping:** Patience of 5 epochs on validation loss

---

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
python -m streamlit run app.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application UI |
| `numpy` | Numerical computations |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting and visualization |
| `scipy` | STFT computation |
| `yfinance` | Stock data download |
| `tensorflow` | CNN model (Keras API) |
| `scikit-learn` | Normalization and metrics |

---

## Supported Tickers

- `RELIANCE.NS` — Reliance Industries
- `TCS.NS` — Tata Consultancy Services
- `INFY.NS` — Infosys
- `^BSESN` — BSE Sensex Index
- `INR=X` — USD/INR Exchange Rate
