# Heart Disease Prediction Application

## Overview

The Heart Disease Prediction Application is a web-based tool designed to predict the likelihood of heart disease based on user inputs. It utilizes a machine learning model to analyze various health parameters and provide a prediction. This project leverages modern web development technologies and machine learning techniques to deliver an interactive and user-friendly experience.

-Tech Stack

1.Frontend
- **HTML/CSS**: Used for structuring and styling the web pages.
- **Tailwind CSS**: A utility-first CSS framework that allows for rapid UI development and customization.
- **JavaScript**: Handles form submission and interaction with the backend API.

2.Backend
- **Flask**: A lightweight WSGI web application framework in Python that serves as the backend for processing predictions and serving the web pages.
- **Python**: The programming language used to implement the backend logic and machine learning model.

3.Machine Learning
- **Scikit-learn**: A Python library used to build and evaluate the logistic regression model for heart disease prediction.

4.Other Technologies
- **Fetch API**: Utilized for making asynchronous requests to the backend server.
- **JSON**: Data format used for sending and receiving data between the frontend and backend.

5.Features

- User Form: A user-friendly form that allows users to input various health parameters such as age, gender, chest pain type, blood pressure, cholesterol levels, etc.
- Prediction Button: Submits the form data to the backend and retrieves the prediction result.
- Result Display: Shows the prediction result on the same page without reloading, and provides a personalized message based on the prediction outcome.

6.Machine Learning Model

The heart disease prediction model is built using Logistic Regression, which is a statistical model used for binary classification problems. The model has been trained to classify whether a person has heart disease based on multiple input features.

7.Dataset Used

The model was trained on the Cleveland Heart Disease dataset, which is a publicly available dataset containing various health metrics of patients. The dataset includes attributes such as:
- Age
- Gender
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise induced angina
- Old peak
- ST slope



