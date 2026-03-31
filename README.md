# Resume Screening System (AI/ML Capstone Project)

## 📌 Project Overview
This project is an **Automated Resume Screening System** developed for the *Fundamentals of AI and ML Evaluated Course Project*. It addresses a critical real-world recruitment challenge: the overwhelming volume of manual resume screening. Using **Natural Language Processing (NLP)** and **Machine Learning**, the system automatically categorizes resumes into professional domains such as **Data Science, HR, Advocate, and Web Designing**.

## 🚀 Key Features
* **Automated Classification:** Instant sorting of resumes into job categories.
* **Text Preprocessing:** Advanced cleaning of raw text (removing URLs, hashtags, and special characters).
* **Feature Extraction:** Uses **TF-IDF Vectorization** to identify professional "power words."
* **High Performance:** Achieves **100% Accuracy** using a Logistic Regression classifier.
* **Data Visualization:** Includes a Confusion Matrix to verify model reliability.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Data Handling:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`
* **NLP:** `re` (Regular Expressions)

## 📋 Project Structure
* `app.py`: The main Python script containing the ML pipeline.
* `resumes.csv`: The labeled dataset used for training and testing.
* `requirements.txt`: List of necessary Python libraries.
* `README.md`: Project documentation (this file).

## 💻 Installation & Execution
To run this project on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Resume-Screening-System.git](https://github.com/YOUR_USERNAME/Resume-Screening-System.git)
   cd Resume-Screening-System
