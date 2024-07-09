# BeatCare
**Heart Disease Prediction Web Application

This is a Flask-based web application that predicts the likelihood of heart disease based on patient data. It uses machine learning algorithms to analyze various health metrics and provide a prediction.

**Table of Contents**

[TOCM]

[TOC]

## Features

- Data input through web forms
- CSV file import for bulk data entry
- Data visualization and analysis
- Machine learning model for heart disease prediction
- MySQL database integration for data storage

## Requirements

- Python 3.7+
- Flask
- MySQL
- pandas
- scikit-learn
- matplotlib

## Setup

### 1. Clone the repository:
`git clone https://github.com/aliattia2/BeatCare.git cd BeatCare`
### 2. Install the required packages:

`pip install -r requirements.txt`

### 3. Set up your MySQL database
Make sure you have MySQL and MySQL Server  installed and running.


#### Log in to MySQL:
open MySQL Command Line Client
Enter the password you set during installation.

Or
open cmd
`cd C:\Program Files\MySQL\ && dir`

then open the workbench by writing cd + mysql workbench [version] ce
then open the server by typing the following command then type the password

`mysql -u root -p`

#### Create a new database:
Create a new database named beatcare (or use another name, but update your config accordingly).
`CREATE DATABASE beatcare;`

#### Create a new user and grant privileges:
`CREATE USER 'your_username'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON beatcare2.* TO 'your_username'@'localhost';
FLUSH PRIVILEGES;`

### 4. Update the connection details in the `app.py` file:

db = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="beatcare"
)


### 5. Run the Flask application:

`python app.py`


Open a web browser and navigate to http://localhost:5000

use the following credentials
patient1
password123
patient

doctor1
password456
doctor

# Usage

1. Home Page: Navigate through different features of the application.
2. Data Entry: Enter individual patient data through a web form.
3. CSV Import: Import bulk data using a CSV file.
4. Data Visualization: View visualizations of the dataset.
5. Prediction: Enter patient data to get a heart disease prediction.
6. Analysis: View model performance metrics and feature importances.

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
any inquiries contact me directly by e-mail
aliattia2@gmail.com
# License
MIT


![](https://img.shields.io/github/stars/pandao/editor.md.svg) ![](https://img.shields.io/github/forks/pandao/editor.md.svg) ![](https://img.shields.io/github/tag/pandao/editor.md.svg) ![](https://img.shields.io/github/release/pandao/editor.md.svg) ![](https://img.shields.io/github/issues/pandao/editor.md.svg) ![](https://img.shields.io/bower/v/editor.md.svg)



### End
