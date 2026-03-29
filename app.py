import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date

from data import (
    load_stock_data,
    normalize_data,
    create_sliding_windows,
    inverse_transform_target
)

from model import (
    prepare_spectrogram_dataset,
    build_cnn_model,
    get_early_stopping
)

st.set_page_config(page_title="Financial Time Series Forecasting", layout="wide")

st.title("Pattern Recognition for Financial Time Series Forecasting")
st.write("STFT + Spectrogram + CNN")

st.sidebar.header("Input Settings")

ticker_options = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "^BSESN", "INR=X"]

selected_tickers = st.sidebar.multiselect(
    "Select at least 3 time series",
    ticker_options,
    default=["RELIANCE.NS", "TCS.NS", "INFY.NS"]
)

start_date = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date(2025, 1, 1))

window_size = st.sidebar.slider("Window Size", 32, 128, 64, 16)
epochs = st.sidebar.slider("Epochs", 5, 30, 10, 5)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32], index=1)

target_ticker = st.sidebar.selectbox(
    "Target stock to predict",
    selected_tickers if selected_tickers else ticker_options
)

run_button = st.sidebar.button("Run Project")

if run_button:
    if len(selected_tickers) < 3:
        st.error("Please select at least 3 time series.")
    else:
        try:
            data = load_stock_data(selected_tickers, str(start_date), str(end_date))
            st.subheader("1. Time Series Data")
            st.line_chart(data)

            scaled_df, scaler = normalize_data(data)
            target_index = list(scaled_df.columns).index(target_ticker)

            X_windows, y, dates = create_sliding_windows(
                scaled_df,
                window_size=window_size,
                target_col=target_index
            )

            st.write("Windowed input shape:", X_windows.shape)
            st.write("Target shape:", y.shape)

            st.subheader("2. Frequency Spectrum")
            sample_signal = X_windows[0][:, target_index]
            fft_vals = np.fft.rfft(sample_signal)
            fft_freqs = np.fft.rfftfreq(len(sample_signal), d=1)

            fig_fft, ax_fft = plt.subplots(figsize=(8, 4))
            ax_fft.plot(fft_freqs, np.abs(fft_vals))
            ax_fft.set_title(f"FFT Spectrum - {target_ticker}")
            ax_fft.set_xlabel("Frequency")
            ax_fft.set_ylabel("Magnitude")
            ax_fft.grid(True)
            st.pyplot(fig_fft)

            st.subheader("3. Spectrogram")
            X_spec = prepare_spectrogram_dataset(X_windows, nperseg=16, noverlap=8)

            fig_spec, ax_spec = plt.subplots(figsize=(8, 4))
            ax_spec.imshow(X_spec[0][:, :, target_index], aspect="auto", origin="lower")
            ax_spec.set_title(f"Spectrogram - {target_ticker}")
            ax_spec.set_xlabel("Time Bins")
            ax_spec.set_ylabel("Frequency Bins")
            st.pyplot(fig_spec)

            split_index = int(0.8 * len(X_spec))
            X_train, X_test = X_spec[:split_index], X_spec[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            test_dates = dates[split_index:]

            st.subheader("4. CNN Architecture")
            st.code(
                "Input -> Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Flatten -> Dense(128) -> Dropout -> Dense(64) -> Output(1)"
            )

            st.subheader("5. Train CNN Model")
            model = build_cnn_model(X_train.shape[1:])
            early_stopping = get_early_stopping()

            history = model.fit(
                X_train,
                y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )

            st.success("Model training completed.")

            fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
            ax_loss.plot(history.history["loss"], label="Train Loss")
            ax_loss.plot(history.history["val_loss"], label="Validation Loss")
            ax_loss.set_title("Training History")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(True)
            st.pyplot(fig_loss)

            st.subheader("6. Prediction")
            y_pred = model.predict(X_test).flatten()

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            col1, col2 = st.columns(2)
            col1.metric("MSE", f"{mse:.6f}")
            col2.metric("MAE", f"{mae:.6f}")

            real_y_test = inverse_transform_target(y_test, scaler, scaled_df.shape[1], target_index)
            real_y_pred = inverse_transform_target(y_pred, scaler, scaled_df.shape[1], target_index)

            fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
            ax_pred.plot(test_dates, real_y_test, label="Actual Price")
            ax_pred.plot(test_dates, real_y_pred, label="Predicted Price")
            ax_pred.set_title(f"Actual vs Predicted Price - {target_ticker}")
            ax_pred.set_xlabel("Date")
            ax_pred.set_ylabel("Price")
            ax_pred.legend()
            ax_pred.grid(True)
            st.pyplot(fig_pred)

            st.subheader("7. Prediction Table")
            st.dataframe({
                "Date": test_dates[:20],
                "Actual Price": real_y_test[:20],
                "Predicted Price": real_y_pred[:20]
            })

            st.subheader("8. Interpretation")
            st.write("Low-frequency parts represent long-term trend.")
            st.write("High-frequency parts represent short-term fluctuations.")
            st.write("Spectrogram converts non-stationary financial data into image-like form for CNN learning.")

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Select inputs from sidebar and click Run Project.")