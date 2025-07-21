Telco Customer Churn Prediction

This project aims to predict customer churn using machine learning in Python and visualize insights using an interactive Power BI dashboard.

 Project Structure

Telco-Churn-Project/
│
├── app.py # Streamlit app for churn prediction
├── churn_model.pkl # Trained machine learning model (Random Forest)
├── TelcoCustomerChurn.csv # Dataset used for training and dashboard
├── TelcoDashboard.pbix # Power BI report file
└── README.md # Project documentation


Tools & Technologies

-Python: pandas, scikit-learn, imbalanced-learn, pickle
-Streamlit: user-friendly churn prediction interface
-Power BI: dynamic dashboard with slicers, visuals & insights
-Git & GitHub: version control and project sharing

Key Features

Machine Learning
- Binary classification using RandomForestClassifier
- SMOTE used for class imbalance
- Feature encoding, preprocessing, and model saving done in Python

Streamlit App
- Interactive form for real-time churn prediction
- Displays churn probability based on input features

Power BI Dashboard
- KPIs: Total Customers, Churn Rate, Average Tenure
- Interactive visuals: Churn by Contract, Internet Service, Monthly  Charges
- Dynamic slicers for gender, payment method, etc.
- Insight box and customer segment table for storytelling



How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/adikutte/Telco-Churn-Project.git
   cd Telco-Churn-Project

2. Launch the Streamlit app:
   streamlit run app.py

3. Open TelcoDashboard.pbix in Power BI to explore the dashboard.

Credits
Dataset: Telco Customer Churn
Project by: Aditya Kutte
