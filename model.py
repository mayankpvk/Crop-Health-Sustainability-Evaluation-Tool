import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

file_path = "farmer_advisor_labeled.csv"
original_df = pd.read_csv(file_path)
original_df_encoded = pd.get_dummies(original_df, columns=['Crop_Type'])
X = original_df_encoded.drop(columns=["Farm_ID", "Plant_Health", "Sustainability_Score"])
feature_columns = X.columns.tolist()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(original_df['Plant_Health'])
print("Class distribution before balancing:")
print(original_df['Plant_Health'].value_counts())
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
X_balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
y_balanced_labels = label_encoder.inverse_transform(y_balanced)
print("\nClass distribution after balancing:")
print(pd.Series(y_balanced_labels).value_counts())
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel performance after balancing:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
joblib.dump(model, "plant_health_rf_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("\nRetrained model and label encoder saved.")

def calculate_sustainability(input_dict):
    score = 100
    if input_dict['Soil_Moisture'] < 50:
        score -= 10
    if input_dict['Temperature_C'] < 20:
        score -= 15
    if input_dict['Rainfall_mm'] < 100:
        score -= 20
    if input_dict['Fertilizer_Usage_kg'] > 100:
        score -= 10
    if input_dict['Pesticide_Usage_kg'] > 5:
        score -= 15
    return max(0, score)

def get_float_input(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a value between {min_val}-{max_val}")
        except ValueError:
            print("Please enter a valid number")

def get_user_input():
    print("\n🌱 Farmer Advisor - Plant Health Prediction 🌱")
    print("Please enter your farm's details below:\n")
    print("Crop Types Available:")
    print("1. Rice 🌾")
    print("2. Corn 🌽")
    print("3. Soybean 🟫")
    print("4. Wheat 🌾")
    while True:
        crop_choice = input("\nSelect crop (1-4): ")
        if crop_choice in ['1', '2', '3', '4']:
            crop_map = {'1': 'Rice', '2': 'Corn', '3': 'Soybean', '4': 'Wheat'}
            crop_type = crop_map[crop_choice]
            break
        print("Please enter a number between 1-4")
    print("\n🌱 Soil Conditions")
    soil_ph = get_float_input("Soil pH (5.0-8.0): ", 5.0, 8.0)
    soil_moisture = get_float_input("Soil Moisture % (10-100): ", 10, 100)
    print("\n☀️ Weather Conditions")
    temperature = get_float_input("Temperature (°C): ", -10, 50)
    rainfall = get_float_input("Rainfall (mm): ", 0, 500)
    print("\n🧑‍🌾 Farm Practices")
    fertilizer = get_float_input("Fertilizer Usage (kg/hectare): ", 0, 300)
    pesticide = get_float_input("Pesticide Usage (kg/hectare): ", 0, 30)
    yield_val = get_float_input("Expected Crop Yield (tons): ", 0, 20)
    return {
        'Soil_pH': soil_ph,
        'Soil_Moisture': soil_moisture,
        'Temperature_C': temperature,
        'Rainfall_mm': rainfall,
        'Crop_Type': crop_type,
        'Fertilizer_Usage_kg': fertilizer,
        'Pesticide_Usage_kg': pesticide,
        'Crop_Yield_ton': yield_val
    }

def give_feedback(row, crop_type):
    feedback = []
    if crop_type == 'Rice':
        if row['Soil_pH'] < 5.5 or row['Soil_pH'] > 6.5:
            feedback.append("Soil pH is not optimal ⚖️ for Rice (ideal: 5.5-6.5).")
        if row['Soil_Moisture'] < 60:
            feedback.append("Soil moisture is low 💧 for Rice (ideal: 60-80%).")
        elif row['Soil_Moisture'] > 80:
            feedback.append("Soil is too wet 🌊 for Rice (ideal: 60-80%).")
        if row['Temperature_C'] < 20:
            feedback.append("Temperature is low ❄️ for Rice (ideal: 20-30°C).")
        elif row['Temperature_C'] > 30:
            feedback.append("Temperature is high 🔥 for Rice (ideal: 20-30°C).")
        if row['Fertilizer_Usage_kg'] < 50 or row['Fertilizer_Usage_kg'] > 100:
            feedback.append("Fertilizer usage is not optimal ⚠️ for Rice (ideal: 50-100 kg).")
        if row['Pesticide_Usage_kg'] < 2 or row['Pesticide_Usage_kg'] > 5:
            feedback.append("Pesticide usage is not optimal ⚠️ for Rice (ideal: 2-5 kg).")
    elif crop_type == 'Corn':
        if row['Soil_pH'] < 5.8 or row['Soil_pH'] > 6.8:
            feedback.append("Soil pH is not optimal ⚖️ for Corn (ideal: 5.8-6.8).")
        if row['Soil_Moisture'] < 55:
            feedback.append("Soil moisture is low 💧 for Corn (ideal: 55-70%).")
        elif row['Soil_Moisture'] > 70:
            feedback.append("Soil is too wet 🌊 for Corn (ideal: 55-70%).")
        if row['Temperature_C'] < 18:
            feedback.append("Temperature is low ❄️ for Corn (ideal: 18-28°C).")
        elif row['Temperature_C'] > 28:
            feedback.append("Temperature is high 🔥 for Corn (ideal: 18-28°C).")
        if row['Fertilizer_Usage_kg'] < 60 or row['Fertilizer_Usage_kg'] > 120:
            feedback.append("Fertilizer usage is not optimal ⚠️ for Corn (ideal: 60-120 kg).")
        if row['Pesticide_Usage_kg'] < 2 or row['Pesticide_Usage_kg'] > 5:
            feedback.append("Pesticide usage is not optimal ⚠️ for Corn (ideal: 2-5 kg).")
    elif crop_type == 'Soybean':
        if row['Soil_pH'] < 6.0 or row['Soil_pH'] > 7.0:
            feedback.append("Soil pH is not optimal ⚖️ for Soybean (ideal: 6.0-7.0).")
        if row['Soil_Moisture'] < 50:
            feedback.append("Soil moisture is low 💧 for Soybean (ideal: 50-65%).")
        elif row['Soil_Moisture'] > 65:
            feedback.append("Soil is too wet 🌊 for Soybean (ideal: 50-65%).")
        if row['Temperature_C'] < 20:
            feedback.append("Temperature is low ❄️ for Soybean (ideal: 20-30°C).")
        elif row['Temperature_C'] > 30:
            feedback.append("Temperature is high 🔥 for Soybean (ideal: 20-30°C).")
        if row['Fertilizer_Usage_kg'] < 40 or row['Fertilizer_Usage_kg'] > 80:
            feedback.append("Fertilizer usage is not optimal ⚠️ for Soybean (ideal: 40-80 kg).")
        if row['Pesticide_Usage_kg'] < 1 or row['Pesticide_Usage_kg'] > 4:
            feedback.append("Pesticide usage is not optimal ⚠️ for Soybean (ideal: 1-4 kg).")
    elif crop_type == 'Wheat':
        if row['Soil_pH'] < 6.0 or row['Soil_pH'] > 7.0:
            feedback.append("Soil pH is not optimal ⚖️ for Wheat (ideal: 6.0-7.0).")
        if row['Soil_Moisture'] < 40:
            feedback.append("Soil moisture is low 💧 for Wheat (ideal: 40-60%).")
        elif row['Soil_Moisture'] > 60:
            feedback.append("Soil is too wet 🌊 for Wheat (ideal: 40-60%).")
        if row['Temperature_C'] < 10:
            feedback.append("Temperature is low ❄️ for Wheat (ideal: 10-25°C).")
        elif row['Temperature_C'] > 25:
            feedback.append("Temperature is high 🔥 for Wheat (ideal: 10-25°C).")
        if row['Fertilizer_Usage_kg'] < 50 or row['Fertilizer_Usage_kg'] > 100:
            feedback.append("Fertilizer usage is not optimal ⚠️ for Wheat (ideal: 50-100 kg).")
        if row['Pesticide_Usage_kg'] < 1 or row['Pesticide_Usage_kg'] > 4:
            feedback.append("Pesticide usage is not optimal ⚠️ for Wheat (ideal: 1-4 kg).")
    if crop_type == 'Rice' and (row['Rainfall_mm'] < 150 or row['Rainfall_mm'] > 300):
        feedback.append("Rainfall is not optimal ☁️ for Rice (ideal: 150-300 mm).")
    elif crop_type == 'Corn' and (row['Rainfall_mm'] < 100 or row['Rainfall_mm'] > 250):
        feedback.append("Rainfall is not optimal ☁️ for Corn (ideal: 100-250 mm).")
    elif crop_type == 'Soybean' and (row['Rainfall_mm'] < 100 or row['Rainfall_mm'] > 200):
        feedback.append("Rainfall is not optimal ☁️ for Soybean (ideal: 100-200 mm).")
    elif crop_type == 'Wheat' and (row['Rainfall_mm'] < 80 or row['Rainfall_mm'] > 150):
        feedback.append("Rainfall is not optimal ☁️ for Wheat (ideal: 80-150 mm).")
    return feedback

def predict_plant_health(input_dict):
    model = joblib.load("plant_health_rf_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    input_df = pd.DataFrame([input_dict])
    input_df_encoded = pd.get_dummies(input_df, columns=['Crop_Type'])
    for col in feature_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[feature_columns]
    prediction = model.predict(input_df_encoded)[0]
    proba = model.predict_proba(input_df_encoded)[0]
    confidence = np.max(proba)
    label = label_encoder.inverse_transform([prediction])[0]
    crop_type = input_dict['Crop_Type']
    sustainability_score = calculate_sustainability(input_dict)
    feedback = give_feedback(input_dict, crop_type)
    health_emojis = {'Happy': '😊', 'Worried': '😟', 'Sad': '😞'}
    if sustainability_score >= 80:
        score_color = '\033[92m'
    elif sustainability_score >= 50:
        score_color = '\033[93m'
    else:
        score_color = '\033[91m'
    print(f"\n🪴 Predicted Plant Health: {label} {health_emojis.get(label, '')}")
    print(f"🔐 Confidence: {confidence * 100:.2f}%")
    print(f"♻️ Sustainability Score: {score_color}{sustainability_score:.2f}\033[0m")
    if feedback:
        print("\n🔍 Feedback:")
        for tip in feedback:
            print(f"✅ {tip}")
    else:
        print("✅ All conditions are optimal! 🌱")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("FARMER ADVISOR - PLANT HEALTH PREDICTION SYSTEM")
    print("="*50)
    input_data = get_user_input()
    print("\n" + "="*50)
    print("ANALYZING YOUR FARM CONDITIONS...")
    print("="*50)
    predict_plant_health(input_data)
