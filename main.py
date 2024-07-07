from flask import Flask, render_template, request, redirect, url_for
import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64
import csv
from io import StringIO
app = Flask(__name__)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# MySQL Configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1111",
    database="beatcare2"
)
cursor = db.cursor()

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS heart_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        age INT,
        sex INT,
        cp INT,
        trtbps INT,
        chol INT,
        fbs INT,
        restecg INT,
        thalachh INT,
        exng INT,
        oldpeak FLOAT,
        slp INT,
        caa INT,
        thall INT,
        output INT
    )
""")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/make_prediction', methods=['GET', 'POST'])
def make_prediction():
    if request.method == 'POST':
        # Get data from form
        input_data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trtbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalachh']),
            float(request.form['exng']),
            float(request.form['oldpeak']),
            float(request.form['slp']),
            float(request.form['caa']),
            float(request.form['thall'])
        ]

        # Fetch all data from the database to train the model
        cursor.execute("SELECT * FROM heart_data")
        data = cursor.fetchall()

        # Convert to pandas DataFrame
        df = pd.DataFrame(data, columns=['id', 'age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output'])

        # Prepare the features and target
        X = df.drop(['id', 'output'], axis=1)
        y = df['output']

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the model
        model = LogisticRegression()
        model.fit(X_scaled, y)

        # Scale the input data
        input_data_scaled = scaler.transform([input_data])

        # Make prediction
        prediction = model.predict(input_data_scaled)

        return render_template('prediction_result.html', prediction=prediction[0])

    return render_template('make_prediction.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Fetch all data from the database
        cursor.execute("SELECT * FROM heart_data")
        data = cursor.fetchall()

        # Convert to pandas DataFrame
        df = pd.DataFrame(data, columns=['id', 'age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output'])

        # Prepare the features and target
        X = df.drop(['id', 'output'], axis=1)
        y = df['output']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Get classification report
        report = classification_report(y_test, y_pred)

        # Get feature importances
        importances = pd.DataFrame({'feature': X.columns, 'importance': np.abs(model.coef_[0])})
        importances = importances.sort_values('importance', ascending=False)

        # Prepare data for the prediction form
        feature_names = X.columns.tolist()

        return render_template('predict.html', accuracy=accuracy, report=report, importances=importances, feature_names=feature_names)

    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Get data from form
        data = (
            request.form['age'],
            request.form['sex'],
            request.form['cp'],
            request.form['trtbps'],
            request.form['chol'],
            request.form['fbs'],
            request.form['restecg'],
            request.form['thalachh'],
            request.form['exng'],
            request.form['oldpeak'],
            request.form['slp'],
            request.form['caa'],
            request.form['thall'],
            request.form['output']
        )

        # Insert data into MySQL
        sql = """
            INSERT INTO heart_data 
            (age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall, output) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, data)
        db.commit()

        return redirect(url_for('index'))

@app.route('/data')
def data():
    cursor.execute("SELECT * FROM heart_data")
    data = cursor.fetchall()
    return render_template('data.html', data=data)
@app.route('/import_csv', methods=['GET', 'POST'])
def import_csv():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'csv_file' not in request.files:
            return "No file part in the request"

        csv_file = request.files['csv_file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if csv_file.filename == '':
            return "No file selected"

        if csv_file:
            try:
                csv_data = StringIO(csv_file.read().decode('utf-8'))
                csv_reader = csv.DictReader(csv_data)

                for row in csv_reader:
                    sql = """
                        INSERT INTO heart_data 
                        (age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall, output) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    data = (
                        row['age'], row['sex'], row['cp'], row['trtbps'], row['chol'],
                        row['fbs'], row['restecg'], row['thalachh'], row['exng'],
                        row['oldpeak'], row['slp'], row['caa'], row['thall'], row['output']
                    )
                    cursor.execute(sql, data)

                db.commit()
                return redirect(url_for('data'))
            except Exception as e:
                db.rollback()
                return f"An error occurred: {str(e)}"

    return render_template('import_csv.html')
@app.route('/analyze')
def analyze():
    # Fetch data from MySQL
    cursor.execute("SELECT * FROM heart_data")
    data = cursor.fetchall()

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=['id', 'age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output'])

    # Features and labels
    X = df.drop(['id', 'output'], axis=1)
    y = df['output']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Multi-Layer Perceptron classifier
    mlp = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=100, alpha=0.0001,
                        solver='adam', verbose=10, random_state=21, tol=0.000000001)
    mlp.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Create a simple plot
    plt.figure(figsize=(10, 6))
    df['age'].hist(bins=20)
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')

    return render_template('analysis.html', accuracy=accuracy, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
