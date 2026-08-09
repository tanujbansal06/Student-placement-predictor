# 🎓 Student Placement Predictor

A Machine Learning based web application that predicts a student's **placement probability** based on academic performance, internships, projects, aptitude score, soft skills, training, and other factors.

The application uses **Logistic Regression** as the Machine Learning model and is deployed using **Flask + Render**.

---

## 🚀 Live Demo

🌐 **Live Application:**  
https://student-placement-predictor-j0mq.onrender.com/

---

## ✨ Features

- 🎓 Student placement probability prediction
- 📊 Uses multiple academic and skill-based parameters
- 🤖 Logistic Regression Machine Learning model
- 📈 Displays estimated placement chances in percentage
- 💡 Provides feedback based on placement probability
- 🖥️ User-friendly web interface
- 🌐 Deployed using Render
- ⚡ Flask backend for model prediction

---

## 🧠 Machine Learning Model

The project uses **Logistic Regression** for binary classification.

### Input Features

| Feature | Description |
|---|---|
| CGPA | Student's CGPA |
| Internships | Number of internships completed |
| Projects | Number of projects completed |
| Workshops/Certifications | Number of workshops/certifications |
| Aptitude Test Score | Aptitude score out of 100 |
| Soft Skills Rating | Soft skills rating out of 5 |
| Extracurricular Activities | Whether the student participates in extracurricular activities |
| Placement Training | Whether the student has completed placement training |
| SSC Marks | Class 10 marks |
| HSC Marks | Class 12 marks |

---

## 📊 Prediction

The model returns an estimated probability of placement.

### Example

```text
Estimated Placement Chance: 78.42%

Good — You have a good chance of getting placed.
