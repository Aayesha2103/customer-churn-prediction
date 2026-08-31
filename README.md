# 📊 ChurnAI — Customer Churn Prediction

<p align="center">

### 🚀 Machine Learning • Customer Analytics

A Machine Learning web application that predicts whether a telecom customer is likely to churn and helps identify customers who may require proactive retention attention.

<br>

<a href="https://customer-churn-prediction-as.streamlit.app/">
  <strong>🚀 Try ChurnAI Live</strong>
</a>

</p>

---

## 🌐 Live Demo

### 🚀 [Open ChurnAI — Live Application](https://customer-churn-prediction-as.streamlit.app/)

Enter customer information and get an instant churn-risk prediction through the interactive Streamlit application.

---

## 📌 About the Project

Customer churn is a major challenge for telecom companies. Losing existing customers can be costly, so identifying customers who are likely to leave can help businesses take action before churn happens.

**ChurnAI** is an end-to-end Machine Learning project that predicts the probability of customer churn using customer demographic, telecom service, contract, and billing information.

The application transforms the Machine Learning prediction into an interactive dashboard where users can enter customer information and understand the resulting churn risk.

### The application provides:

- 🎯 Customer churn probability
- 🔴 Potential churner identification
- 🟢 Likely-to-stay identification
- 📊 Visual risk analysis
- 📈 Probability and threshold comparison
- 👤 Customer profile summary
- 💳 Contract and billing information
- 💼 Business-oriented retention suggestions

---

## 🎯 Project Objectives

The main objectives of this project are:

- Predict whether a customer is likely to churn.
- Estimate the probability of customer churn.
- Identify customers who may require retention attention.
- Present Machine Learning predictions through an interactive dashboard.
- Make the prediction easier to understand for non-technical users.
- Connect Machine Learning predictions with practical business decisions.

---

## 🧠 How It Works

The application follows this workflow:

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
```

### Prediction Process

1. The user enters the customer's information through the Streamlit interface.
2. The entered information is processed into the format required by the trained model.
3. The trained Machine Learning model calculates the probability that the customer will churn.
4. The predicted probability is compared with the project's decision threshold.
5. The customer is classified according to the prediction.
6. The application displays the result through an interactive prediction page.
7. The result is presented using visualizations and business-oriented explanations.

---

## 📊 Customer Information Used

The application accepts information from several categories.

### 👤 Customer Information

- Gender
- Senior Citizen
- Partner
- Dependents
- Customer Tenure

### 📡 Telecom Services

- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### 💳 Contract & Billing

- Contract Type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

---

## 🔮 Prediction Output

After entering the customer's information, ChurnAI generates a detailed prediction result.

### 📈 Churn Probability

The application displays the estimated probability that the customer will leave the telecom service.

For example:

```text
Churn Probability: 39.53%
```

### 🚦 Risk Classification

The customer is classified based on the model's prediction and the decision threshold.

```text
🔴 Potential Churner

or

🟢 Likely to Stay
```

### ⚖️ Decision Threshold

The predicted probability is compared against the decision threshold to determine the final classification.

For example:

```text
Predicted Probability = 0.3953
Decision Threshold     = 0.30

0.3953 >= 0.30

Result → Potential Churner
```

### 👤 Customer Tenure

The prediction result also displays the customer's tenure with the company.

---

## 📈 Visual Risk Analysis

The prediction page uses visual elements to make the Machine Learning result easier to understand.

### Customer Risk Distribution

The application presents the customer's churn and stay probabilities in a visual format.

This allows the user to quickly understand how strong the predicted churn risk is.

### Probability vs Decision Threshold

The application also shows the relationship between the customer's predicted churn probability and the decision threshold.

This helps explain why the final classification was produced.

---

## 💼 Business Interpretation

ChurnAI is designed to go beyond simply returning a Machine Learning prediction.

If a customer is identified as a potential churner, the application highlights that the customer may require proactive retention attention.

Possible business actions include:

- Personalized offers
- Proactive customer support
- Contract incentives
- Service improvements

The purpose is to help businesses identify potentially at-risk customers early and take appropriate retention actions.

---

## 🤖 Machine Learning

The Machine Learning model predicts the likelihood of customer churn from the customer's profile, services, contract, and billing information.

Instead of producing only a simple yes/no answer, the model produces a **churn probability**.

That probability is then evaluated against the project's decision threshold to produce the final customer-risk classification.

### Example

```text
Customer Information
        ↓
Machine Learning Model
        ↓
Churn Probability = 39.53%
        ↓
Compare with Threshold = 30%
        ↓
39.53% > 30%
        ↓
Potential Churner
```

This makes the prediction more informative because the application can show **how likely** the customer is to churn rather than only displaying a binary result.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| 🐍 Python | Application and Machine Learning development |
| 🐼 Pandas | Data manipulation and preprocessing |
| 🔢 NumPy | Numerical operations |
| 🤖 Scikit-learn | Machine Learning |
| 💾 Joblib | Saving and loading trained models |
| 📊 Matplotlib | Data visualization |
| 🌐 Streamlit | Interactive web application |
| 🔧 Git | Version control |
| 🐙 GitHub | Source code hosting |
| ☁️ Streamlit Cloud | Application deployment |

---

## 📂 Project Structure

```text
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
```

---

## 🚀 Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Aayesha2103/customer-churn-prediction.git
```

### 2. Navigate to the Project

```bash
cd customer-churn-prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Deployment

ChurnAI is deployed using **Streamlit Cloud**.

### 🚀 Live Application

**[Launch ChurnAI](https://customer-churn-prediction-as.streamlit.app/)**

The deployed application allows users to interact with the trained Machine Learning model directly through a web browser.

---

## 📊 Example Prediction

A customer may enter information such as:

```text
Contract:          Month-to-month
Internet Service:  DSL
Monthly Charges:   70.00
Total Charges:     840.00
Tenure:             12 months
Payment Method:    Electronic check
```

The model then calculates the customer's churn probability.

Example:

```text
Churn Probability: 39.53%

Decision Threshold: 0.30

Final Result:
Potential Churner
```

The application then provides a visual representation of the prediction along with an explanation of what the result means.

---

## 💡 Why This Project Matters

Customer churn prediction is a practical Machine Learning problem with direct business applications.

Instead of waiting until customers leave, businesses can use predictive analytics to identify customers who may be at risk and take proactive action.

This project demonstrates how a Machine Learning model can be transformed into a complete interactive application rather than remaining only as a dataset, notebook, or standalone model.

---

## ⭐ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 Churn Prediction | Predicts potential customer churn |
| 📊 Probability Score | Shows estimated churn probability |
| 🚦 Risk Classification | Identifies potential churners |
| ⚖️ Decision Threshold | Explains the classification decision |
| 📈 Visual Analysis | Makes predictions easier to understand |
| 👤 Customer Profile | Displays customer characteristics |
| 📡 Telecom Analysis | Includes customer service information |
| 💳 Billing Analysis | Includes contract and billing information |
| 💼 Retention Suggestions | Provides possible business actions |
| 🌐 Web Deployment | Accessible through a live web application |

---

## 🧪 Skills Demonstrated

This project demonstrates practical experience with:

- Python
- Machine Learning
- Classification
- Customer churn prediction
- Data preprocessing
- Probability-based prediction
- Model deployment
- Streamlit application development
- Interactive data visualization
- Git and GitHub
- Cloud deployment

---

## 🔄 End-to-End Project Pipeline

```text
Raw Customer Dataset
        ↓
Data Preparation
        ↓
Feature Processing
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Trained Model
        ↓
Model Saved
        ↓
Streamlit Application
        ↓
Customer Input
        ↓
Churn Prediction
        ↓
Visual Results
        ↓
Business Interpretation
```

---

## 🌐 Project Links

### 🚀 Live Application

**[Open ChurnAI](https://customer-churn-prediction-as.streamlit.app/)**

### 💻 GitHub Repository

**[View Source Code](https://github.com/Aayesha2103/customer-churn-prediction)**

---

## 👩‍💻 Author

### Aayesha Singh

**Machine Learning & Data Science Project**

GitHub: [@Aayesha2103](https://github.com/Aayesha2103)

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

<p align="center">

## 🚀 Predict. Understand. Retain.

### [Try ChurnAI Live →](https://customer-churn-prediction-as.streamlit.app/)

</p>
