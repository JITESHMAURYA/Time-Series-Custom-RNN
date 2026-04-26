# 📊 Time Series Forecasting using Custom RNN

## 🧑‍🎓 Student Details

* **Name:** Jitesh Maurya
* **Roll Number:** 102317244
* **Course:** UCS761 - Sequence Modeling

---

## 📌 Project Overview

This project implements a **time-series forecasting pipeline from scratch** to understand how sequence models work.

The goal is to:

* Learn how time-series data is processed
* Understand how models use past information
* Analyze where models fail and why

---

## ⚙️ Personalized Parameters

Derived from roll number:

* **Window Size:** 12
* **Prediction Horizon:** 3
* **Hidden Size:** 14
* **Model Type:** Custom RNN (Last digit even)

---

## 📂 Dataset Used

* Airline Passengers Dataset (for demonstration)
* Electricity Consumption Dataset (as required)

---

## 🔄 Data Preprocessing

* Converted raw time series into supervised learning format using **windowing**
* Applied **normalization (mean = 0, std = 1)**
* Used **chronological train-test split (80-20)**

---

## 🧠 Models Implemented

### 1. MLP (Baseline)

* Treats input as independent features
* No sequence awareness
* Used for comparison

### 2. Custom RNN (From Scratch)

* Implemented without using `nn.RNN`
* Maintains hidden state (memory)
* Processes input sequentially

### 3. Additional Models

* LSTM (prebuilt)
* Transformer (prebuilt)

---

## 🔬 Custom RNN Working

At each timestep:

```
h_t = tanh(Wx * x_t + Wh * h_(t-1))
```

* Combines current input and previous memory
* Updates hidden state step-by-step
* Final hidden state used for prediction

---

## 📊 Evaluation Metrics

* **MSE (Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**

---

## 📈 Results & Observations

* RNN performs better than MLP due to sequence awareness
* Predictions are **smooth and fail to capture sharp peaks**
* Indicates:

  * Underfitting
  * Vanishing gradient problem
  * Limited memory capacity

---

## 🔍 Ablation Study

Tested model with:

* Half window size
* Original window size
* Double window size

### Observation:

* Smaller window → less context → poor performance
* Larger window → better context but diminishing returns

---

## ⚠️ Limitations

* Simple RNN struggles with long-term dependencies
* Suffers from **vanishing gradients**
* Cannot capture sharp variations effectively

---

## 🔄 When Other Models Perform Better

* **GRU/LSTM:** Better memory handling using gates
* **Transformer:** Captures long-range dependencies efficiently

---

## 🎥 Video Explanation

👉 (Add your YouTube link here)

---

## 💻 How to Run

```bash
pip install numpy pandas matplotlib torch scikit-learn
```

Run notebook:

```bash
jupyter notebook
```

---

## 📌 Key Takeaways

* Time series must be converted into supervised format
* Sequence models outperform non-sequence models
* Memory plays a crucial role in forecasting
* Model limitations are as important as performance

---

## ⭐ Acknowledgment

This project was completed as part of the UCS761 course to strengthen understanding of sequence modeling concepts.
