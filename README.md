# ❤️ Heart Failure Prediction App

A **Machine Learning web app** built with **Streamlit** that predicts the risk of heart disease based on patient health data. The model has been trained on a heart disease dataset and is deployed for live use.

🔗 **Live Demo:** [Click Here](https://heart-failure-prediction-bjahe4tlm6jtuvdwqxvzpg.streamlit.app/)  
📂 **GitHub Repository:** https://github.com/pathakaditya424-wq/heart-failure-prediction

---

## 🚀 Features
- User-friendly web interface using Streamlit
- Input patient details (age, blood pressure, cholesterol, ECG, etc.)
- Real-time prediction (High Risk / Low Risk)
- Probability score provided for better interpretability
- Model built with **TensorFlow & scikit-learn**
- Deployed for free via **Streamlit Cloud**

---

## 🧠 Machine Learning Pipeline
1. **Data Preprocessing**
   - Handled invalid values (`RestingBP=0`, `Cholesterol=0`)
   - Applied One-Hot Encoding for categorical features
   - Used StandardScaler for numerical features

2. **Model Training**
   - Neural Network implemented with TensorFlow (Keras Sequential API)
   - Two hidden layers with ReLU activation
   - Sigmoid output for binary classification

3. **Evaluation**
   - Accuracy: **86%** on test data
   - Balanced Precision and Recall
   - Confusion Matrix visualization

---

## 📦 Installation & Usage

Clone the repository:
```bash
git clone https://github.com/pathakaditya424-wq/heart-failure-prediction.git
cd heart-failure-prediction
```
Install dependencies: pip install -r requirements.txt
Run locally:streamlit run heart_app.py
<img width="1920" height="1080" alt="Screenshot (147)" src="https://github.com/user-attachments/assets/1698d101-3118-4e4a-a9a8-25cb074f738e" />
<img width="1920" height="1080" alt="Screenshot (148)" src="https://github.com/user-attachments/assets/0b8322ab-7fed-41bb-b4d4-1177a6f67ee0" />
<img width="1920" height="1080" alt="Screenshot (149)" src="https://github.com/user-attachments/assets/ebbc6344-d091-4962-a838-cfd448aa75b5" />
<img width="1920" height="1080" alt="Screenshot (150)" src="https://github.com/user-attachments/assets/17c2147d-f378-4054-a8ee-342e39478ab1" />



