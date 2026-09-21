# Student Exam Performance Prediction

A Machine Learning web application that predicts a student's **Mathematics Score** based on demographic information, reading and writing scores, lunch type, and test preparation status.

This project demonstrates an end-to-end Machine Learning workflow, from data ingestion and preprocessing to model training, prediction, and deployment through a Flask web application.

---

## 1. Project Overview

Student academic performance can be influenced by several demographic and educational factors.

This project uses Machine Learning to predict a student's **Math Score** using information such as:

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch
- Test Preparation Course
- Reading Score
- Writing Score

The trained model is integrated with a **Flask web application**, allowing users to enter student information and receive a predicted Mathematics Score.

---

## 2. Project Objective

The main objectives of this project are:

- Build an end-to-end Machine Learning pipeline.
- Perform data ingestion and preprocessing.
- Transform categorical and numerical features.
- Train and evaluate Machine Learning models.
- Select a suitable model for prediction.
- Create a Flask-based web application.
- Provide an easy-to-use interface for making predictions.

---

## 3. Key Features

- Predicts student Mathematics Score.
- Machine Learning-based prediction.
- Complete data processing pipeline.
- Data preprocessing and transformation.
- Model evaluation and performance tracking.
- Modular project structure.
- Flask web application.
- Interactive prediction form.
- Responsive user interface.

---

## 4. Technologies Used

### Programming Language

- Python

### Data Science and Machine Learning

- Pandas
- NumPy
- Scikit-learn

### Web Development

- Flask
- HTML
- CSS

### Development Tools

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## 5. Machine Learning Workflow

**The project follows an end-to-end Machine Learning workflow:**

Dataset
   |
   v
Data Ingestion
   |
   v
Data Validation
   |
   v
Data Transformation
   |
   v
Model Training
   |
   v
Model Evaluation
   |
   v
Best Model Selection
   |
   v
Prediction Pipeline
   |
   v
Flask Web Application
   |
   v
Predicted Math Score

---

## 6. Dataset and Features

The project uses student performance data containing demographic information and academic scores.

### Input Features

| Feature | Description |
|---|---|
| Gender | Student gender |
| Race/Ethnicity | Student race or ethnicity group |
| Parental Level of Education | Highest education level of the student's parent |
| Lunch | Type of lunch received by the student |
| Test Preparation Course | Whether the student completed the preparation course |
| Reading Score | Student reading score |
| Writing Score | Student writing score |

### Target Variable

**Math Score**

The trained Machine Learning model predicts the student's Mathematics Score based on the input features.


## 7. Model Performance

The project evaluates trained Machine Learning models and selects a suitable model based on performance.

### Model Evaluation

**R² Score: `0.88064`**

The R² score indicates how well the model explains the variation in the target Mathematics Score on the evaluation data.

> Note: Model performance can vary depending on the dataset, preprocessing, training configuration, and evaluation split.

---

## 8. Project Structure

mlproject/
│
├── .ebextensions/
│
├── artifacts/
│   ├── data_transformation/
│   └── model_trainer/
│
├── logs/
│
├── notebook/
│
├── src/
│   ├── components/
│   ├── exception/
│   ├── pipeline/
│   └── utils/
│
├── templates/
│   ├── home.html
│   └── index.html
│
├── .gitignore
├── app.py
├── application.py
├── README.md
├── requirements.txt
└── stud.csv

---

## 9. Web Application

The project includes a Flask-based web application with two main pages.

### Landing Page

The landing page introduces the project and provides access to the prediction application.

### Prediction Page

Users can enter:

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch
- Test Preparation Course
- Reading Score
- Writing Score

The application processes the input through the trained Machine Learning pipeline and displays the predicted Mathematics Score.

---

## 10. Installation

### Clone the Repository

git clone https://github.com/skhumera0202/mlproject.git

**Navigate to the Project:**

cd mlproject

**Create a Virtual Environment:**

python -m venv venv

**Activate the Virtual Environment:**

**Windows PowerShell:**

.\venv\Scripts\Activate.ps1

**Install Dependencies:**

pip install -r requirements.txt

---

## 11. Running the Application

**Start the Flask application:**

python app.py

**The application will be available at:**

http://127.0.0.1:5000

Open the address in your browser and select Start Prediction to use the application.

---

## 12. Future Improvements

**Possible future improvements include:**

- Add additional Machine Learning models.
- Improve model accuracy through feature engineering.
- Add interactive data visualizations.
- Add model comparison metrics.
- Add automated model retraining.
- Improve prediction result visualization.
- Add user authentication.
- Deploy the application using a cloud platform.
- Add automated testing and CI/CD.

---

## 13. Learning Outcomes

**Through this project, I practiced:**

- Python programming
- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- Machine Learning model training
- Model evaluation
- Hyperparameter tuning
- Flask application development
- Building prediction pipelines
- Git and GitHub
- Project documentation

---

## 14. Author

**Humera Shaikh**

GitHub:

https://github.com/skhumera0202/mlproject

---

## License

This project is created for learning and educational purposes.