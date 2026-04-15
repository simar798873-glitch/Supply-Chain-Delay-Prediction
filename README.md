# Supply-Chain-Delay-Prediction
# ============================================================
# 🚚 SUPPLY CHAIN DELAY PREDICTION
# ============================================================

# 📌 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 📌 Step 2: Create Synthetic Dataset
np.random.seed(42)

data = pd.DataFrame({
    "Distance_km": np.random.randint(5, 500, 100),
    "Traffic": np.random.choice(["Low", "Medium", "High"], 100),
    "Weather": np.random.choice(["Clear", "Rainy", "Foggy"], 100),
    "Vehicle_Capacity": np.random.randint(50, 200, 100),
    "Driver_Experience": np.random.randint(1, 10, 100),
    "Fuel_Status": np.random.choice(["Low", "Medium", "Full"], 100),
    "Road_Type": np.random.choice(["Highway", "City"], 100),
    "Time_of_Day": np.random.choice(["Morning", "Evening", "Night"], 100)
})

# 📌 Step 3: Create Target Variable
data["Delay"] = np.where(
    (data["Traffic"] == "High") |
    (data["Weather"] != "Clear") |
    (data["Distance_km"] > 300),
    1, 0
)

# 📌 Step 4: Encode Categorical Data
le = LabelEncoder()

for col in ["Traffic", "Weather", "Fuel_Status", "Road_Type", "Time_of_Day"]:
    data[col] = le.fit_transform(data[col])

# 📌 Step 5: Split Data
X = data.drop("Delay", axis=1)
y = data["Delay"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 📊 MODEL TRAINING
# ============================================================

# 🌳 Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# 🌲 Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 📉 Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# ============================================================
# 📈 MODEL EVALUATION
# ============================================================

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

print("\nBest Model: Random Forest")

# ============================================================
# 🔍 FEATURE IMPORTANCE (Random Forest)
# ============================================================

importance = rf.feature_importances_
features = X.columns

plt.figure()
plt.bar(features, importance)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()

# ============================================================
# 🔮 PREDICTION SYSTEM
# ============================================================

print("\n🔮 Test a New Delivery Scenario")

distance = float(input("Enter Distance (km): "))
traffic = int(input("Traffic (0=Low,1=Medium,2=High): "))
weather = int(input("Weather (0=Clear,1=Foggy,2=Rainy): "))
capacity = float(input("Vehicle Capacity: "))
experience = float(input("Driver Experience: "))
fuel = int(input("Fuel Status (0=Full,1=Low,2=Medium): "))
road = int(input("Road Type (0=City,1=Highway): "))
time = int(input("Time (0=Evening,1=Morning,2=Night): "))

new_data = np.array([[distance, traffic, weather, capacity, experience, fuel, road, time]])

prediction = rf.predict(new_data)

if prediction[0] == 1:
    print("🚨 Delay Expected")
else:
    print("✅ On-Time Delivery")
