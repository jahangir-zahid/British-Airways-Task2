# 🎯 Task 2: Predicting Customer Bookings

**British Airways Data Science Virtual Experience (Forage)**

A machine learning model that predicts whether a customer will complete a flight booking, and shows which factors drive that decision.

---

## 📌 Project Overview

British Airways wants to understand what makes a customer complete a booking. This project:

1. explores and prepares a customer booking dataset,
2. engineers new features,
3. trains a **RandomForest** classifier,
4. evaluates it with **5-fold cross-validation** and a hold-out test set, and
5. summarises the findings on a **single slide** for a manager.

---

## 📊 Dataset

| Item | Detail |
|---|---|
| File | `customer_booking.csv` |
| Rows | 50,000 customer booking attempts |
| Columns | 14 |
| Target | `booking_complete` (1 = booked, 0 = did not book) |
| Class balance | About 15% booked, 85% did not (**imbalanced**) |
| Missing values | None |

**Main columns:** `num_passengers`, `sales_channel`, `trip_type`, `purchase_lead`, `length_of_stay`, `flight_hour`, `flight_day`, `route`, `booking_origin`, `wants_extra_baggage`, `wants_preferred_seat`, `wants_in_flight_meals`, `flight_duration`.

---

## 🧹 Data Preparation

- Converted `flight_day` (Mon to Sun) to numbers 1 to 7.
- Grouped the rare values of `route` (799 values) and `booking_origin` (104 values): kept the **top 10** and labelled the rest "Other".
- One-hot encoded `sales_channel`, `trip_type`, `booking_origin` and `route`, giving **41 features** in total.

### ✨ Engineered features

| Feature | Meaning |
|---|---|
| `total_extras` | Baggage + preferred seat + meals (0 to 3) |
| `lead_per_pax` | Purchase lead time ÷ number of passengers |
| `stay_x_pax` | Length of stay × number of passengers |
| `is_weekend` | 1 if the flight is on Saturday or Sunday |
| `is_night_flight` | 1 if the flight departs between 22:00 and 05:59 |

---

## 🤖 Model

- **Algorithm:** `RandomForestClassifier` (300 trees, `min_samples_leaf=5`)
- **Imbalance handling:** `class_weight="balanced_subsample"`
- **Split:** 80% train / 20% test, stratified to keep the 15% booking rate
- **Validation:** 5-fold stratified cross-validation on the training set

---

## 📈 Results

| Metric | 5-fold CV (mean ± std) | Hold-out test |
|---|---|---|
| **AUC** | 0.766 ± 0.005 | 0.778 |
| Accuracy | 0.783 ± 0.004 | 0.787 |
| Precision | 0.344 ± 0.009 | 0.358 |
| Recall | 0.497 ± 0.013 | 0.532 |
| F1 | 0.407 ± 0.010 | 0.428 |

**Reading the results:**
- CV and test scores are very close, so the model is **not overfitting**.
- The model has **moderate predictive power**. It catches about half of the real bookers, and about a third of the customers it flags go on to book.
- Accuracy alone would mislead: always predicting "no booking" already scores about 85%. That is why AUC, precision and recall are reported.

---

## 🔍 What Drives Bookings?

![Feature importance](feature_importance.png)

**Top drivers:** booking origin (Australia, Malaysia), purchase lead time, length of stay, flight hour and flight duration. Engineered features such as lead time per passenger and stay × passengers also rank highly.

### 💡 Key insights

- 🌏 **Market matters most:** about 34% of Malaysia-origin customers book, versus about 5% from Australia.
- 🗓️ **Trip planning:** 3 to 7 day stays convert at about 19%, versus about 10% for 14 to 30 day stays.
- 🧳 **Extras signal intent:** about 19% book when all three extras are chosen, versus about 11% with none.

> ⚠️ Feature importance shows **association, not cause**.

### ✅ Recommendation

Target high-converting markets and promote extras early in the booking journey.

---

## 📁 Files

| File | Description |
|---|---|
| `Booking_Model.ipynb` | Full analysis: exploration, preparation, model, evaluation |
| `customer_booking.csv` | Source dataset (Forage) |
| `feature_importance.csv` | Importance score for every feature |
| `feature_importance.png` | Feature importance chart |
| `Booking_Summary_Slide.pptx` | One-slide summary for the manager |

---

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
```

1. Place `customer_booking.csv` in the same folder as the notebook.
2. Open `Booking_Model.ipynb` in VS Code or Jupyter.
3. Choose a Python kernel and select **Run All**.

Results are reproducible because `random_state=42` is set throughout.

---

## 🔮 Possible Improvements

- Try gradient boosting (for example XGBoost or LightGBM) and compare against the RandomForest.
- Tune hyperparameters with a grid or randomised search.
- Adjust the decision threshold to trade precision against recall, depending on the business goal.
- Add SHAP values for a clearer view of how each variable pushes predictions up or down.

---

## 🛠️ Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter, VS Code, PowerPoint.

---

## 👤 Author

*Jahangir Ali*, Data Science Virtual Experience, British Airways (Forage)
