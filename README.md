# Crop-Health-Sustainability-Evaluation-Tool
Crop Health &amp; Sustainability Evaluation Tool is a machine learning-based system that predicts crop health (Happy, Worried, Sad) with 84% accuracy and calculates a custom sustainability score based on soil, weather, and farm practice inputs. It provides confidence-backed results and actionable feedback for better farming decisions.
<br><br>

Key Features
<ul>
<li>High-Accuracy Predictions: Trained with Random Forest on real-world data, achieving 84% accuracy with class-wise precision up to 0.90.</li>

<li>Custom Sustainability Index (0–100): Penalizes suboptimal values in rainfall, temperature, fertilizer, pesticide, and soil moisture to assess environmental sustainability.</li>

<li>Interactive Input Pipeline: Collects 8+ features including crop type, weather, and soil parameters via CLI.</li>

<li>Personalized Agronomic Feedback: Over 20 crop-specific rules generate insights tailored for Rice, Corn, Soybean, and Wheat.</li>

<li>Model Serialization: Includes .pkl files for easy model reuse and integration in production environments.</li>
</ul>

Technologies Used
<ul>
  <li>Python, Pandas, NumPy</li>
  <li>Scikit-learn (Random Forest, LabelEncoder)</li>
  <li>Imbalanced-learn (SMOTE)</li>
  <li>Joblib (Model Persistence)</li>
</ul>
