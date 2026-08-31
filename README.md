# 📊 ChurnAI — Customer Churn Prediction

> A Machine Learning web application that predicts whether a telecom customer is likely to churn and helps identify customers who may require proactive retention attention.

<p align="center">

[🚀 Live Demo](https://customer-churn-prediction-as.streamlit.app/)

</p>

---

## 🌐 Live Application

### 🚀 Try ChurnAI Online

**👉 [Open the Live Streamlit App](https://customer-churn-prediction-as.streamlit.app/)**

Enter a customer's information and get an instant churn-risk prediction.

---

## 📌 About the Project

Customer churn is a major challenge for telecom companies. Losing existing customers can be costly, so identifying customers who are likely to leave can help businesses take action before churn happens.

**ChurnAI** is an end-to-end Machine Learning project that predicts the probability of customer churn using customer demographic, service, contract, and billing information.

The application provides:

- Customer churn probability
- High / Low churn-risk classification
- Decision threshold comparison
- Customer information summary
- Visual risk analysis
- Business-oriented retention recommendations

The project combines a trained Machine Learning model with an interactive Streamlit dashboard.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Predict whether a customer is likely to churn.
- Estimate the probability of customer churn.
- Identify potentially high-risk customers.
- Present predictions through an easy-to-understand dashboard.
- Provide useful information that can support customer-retention decisions.

---

## 🧠 How It Works

The application follows this basic workflow:

```text
Customer Information
        ↓
Feature Processing
        ↓
Trained Machine Learning Model
        ↓
Churn Probability
        ↓
Decision Threshold
        ↓
Churn Risk Classification
        ↓
Visual Results & Business Interpretation

The user enters customer information through the Streamlit interface.

The trained model calculates the probability that the customer will churn.

That probability is then compared with the project's decision threshold.

The application finally displays the prediction and explains what the result means.

📊 Customer Information Used

The application accepts information from several categories.

👤 Customer Information
Gender
Senior Citizen
Partner
Dependents
Customer Tenure
📡 Telecom Services
Phone Service
Multiple Lines
Internet Service
Online Security
Online Backup
Device Protection
Tech Support
Streaming TV
Streaming Movies
💳 Contract & Billing
Contract Type
Paperless Billing
Payment Method
Monthly Charges
Total Charges

These inputs are passed to the trained model to generate the prediction.

🔮 Prediction Output

After entering the customer's information, ChurnAI provides a detailed prediction page.

The result includes:

Churn Probability

Shows the estimated probability that the customer will leave the telecom service.

Risk Classification

The customer is classified as either:

🔴 Potential Churner
🟢 Likely to Stay

Decision Threshold

The probability is compared against the selected decision threshold to determine the final classification.

Customer Tenure

Displays how long the customer has been with the company.

📈 Visual Risk Analysis

The prediction page includes visualizations to make the result easier to understand.

Customer Risk Distribution

Shows the relationship between:

Churn Risk
Stay Probability
Probability vs Decision Threshold

Shows the customer's predicted churn probability alongside the decision threshold.

This makes it easier to understand why the model classified the customer as a potential churner or likely to stay.

💼 Business Interpretation

The application is designed not only to provide a Machine Learning prediction but also to make the prediction useful from a business perspective.

For customers identified as potential churners, the application suggests possible retention actions such as:

Personalized offers
Proactive customer support
Contract incentives
Service improvements

The goal is to help a business identify customers who may need attention before they decide to leave.

🤖 Machine Learning

The project uses a trained Logistic Regression model for customer churn prediction.

The model produces a probability of churn rather than only returning a yes/no prediction.

This probability is then evaluated against the project's decision threshold.

For example:

Predicted Probability = 0.39
Decision Threshold     = 0.30

0.39 >= 0.30
        ↓
Potential Churner

This approach makes the prediction easier to interpret and allows the application to prioritize customers based on their estimated churn risk.

🛠️ Technologies Used
Technology	Purpose
Python	Programming language
Pandas	Data manipulation
NumPy	Numerical operations
Scikit-learn	Machine Learning
Joblib	Model serialization
Matplotlib	Data visualization
Streamlit	Interactive web application
Git	Version control
GitHub	Source code hosting
Streamlit Cloud	Application deployment
📂 Project Structure
customer-churn-prediction/
│
├── 📄 app.py
│   └── Streamlit application and prediction interface
│
├── 📄 train.py
│   └── Machine Learning model training
│
├── 📁 data/
│   └── Telco-Customer-Churn.csv
│
├── 📁 models/
│   ├── churn_model.pkl
│   └── churn_threshold.pkl
│
├── 📄 requirements.txt
│   └── Python dependencies
│
├── 📄 .gitignore
│   └── Files excluded from Git tracking
│
└── 📄 README.md
    └── Project documentation
🚀 Run the Project Locally
1. Clone the repository
git clone https://github.com/Aayesha2103/customer-churn-prediction.git
2. Open the project directory
cd customer-churn-prediction
3. Create a virtual environment
python -m venv venv
4. Activate the virtual environment
Windows
venv\Scripts\activate
5. Install dependencies
pip install -r requirements.txt
6. Run the Streamlit application
streamlit run app.py

The application will open in your browser.

☁️ Deployment

The application is deployed using Streamlit Cloud.

🚀 Live Application

👉 Launch ChurnAI

The deployed application allows users to interact with the Machine Learning model directly through the browser without setting up the project locally.

📊 Example Prediction

A customer may enter information such as:

Contract:          Month-to-month
Internet Service:  DSL
Monthly Charges:   70.00
Total Charges:     840.00
Tenure:            12 months
Payment Method:    Electronic check

The model then calculates a churn probability.

For example:

Churn Probability: 39.53%

Decision Threshold: 0.30

Result:
Potential Churner

The application then presents visual analysis and a business-oriented interpretation of the result.

💡 Why This Project Matters

Customer churn prediction is a practical Machine Learning problem with direct business applications.

Instead of waiting until customers leave, businesses can use predictive analytics to identify customers who may be at risk and take proactive action.

This project demonstrates how Machine Learning can be transformed into a usable application rather than remaining only as a notebook or model.

🔑 Key Features
🎯 Customer churn prediction
📊 Probability-based risk assessment
🔴 High-risk customer identification
🟢 Low-risk customer identification
📈 Interactive visualizations
👤 Customer profile analysis
💳 Contract and billing analysis
💼 Business-oriented recommendations
🌐 Deployed Streamlit application
📱 Responsive dashboard interface
🧪 Project Highlights

This project demonstrates practical experience with:

Data-driven problem solving
Machine Learning classification
Probability-based predictions
Model deployment
Interactive dashboard development
Data visualization
Python application development
Git and GitHub
Cloud deployment
👩‍💻 Author
Aayesha Singh

Machine Learning & Data Science Project

GitHub:
github.com/Aayesha2103
