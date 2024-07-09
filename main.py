from flask import Flask, render_template, request, redirect, url_for, session
import mysql.connector
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import csv
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from functools import wraps


# MySQL Configuration
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# ... (database connection setup)
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1111",
    database="beatcare2"
)
cursor = db.cursor()


def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or session['user_type'] != 'doctor':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
# ... (other routes and functions)
def patient_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or session['user_type'] != 'patient':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_type = request.form['user_type']

        # Check credentials in the database
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s AND user_type = %s",
                       (username, password, user_type))
        user = cursor.fetchone()

        if user:
            session['logged_in'] = True
            session['username'] = username
            session['user_type'] = user_type
            if user_type == 'patient':
                return redirect(url_for('patient_dashboard'))
            else:
                return redirect(url_for('doctor_dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('user_type', None)
    return redirect(url_for('login'))

@app.route('/patient_dashboard')
@patient_required
def patient_dashboard():
    if 'logged_in' in session and session['user_type'] == 'patient':
        return render_template('patient_dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/doctor_dashboard')
@doctor_required
def doctor_dashboard():
    if 'logged_in' in session and session['user_type'] == 'doctor':
        return render_template('doctor_dashboard.html', username=session['username'])
    return redirect(url_for('login'))
# Decorator function to check if user is a logged-in doctor

@app.route('/patient_submit', methods=['GET', 'POST'])
@patient_required
def patient_submit():
    if request.method == 'POST':
        # Get data from form
        data = (
            session['username'],  # Use username as patient ID
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
            (patient_id, age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall, output) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, data)
        db.commit()

        return redirect(url_for('patient_dashboard'))

    return render_template('patient_submit_form.html')


@app.route('/patient_view_data')
@patient_required
def patient_view_data():
    # Fetch the patient's data from the database
    cursor.execute("""
        SELECT age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall, output 
        FROM heart_data 
        WHERE patient_id = %s 
        ORDER BY id DESC
    """, (session['username'],))

    data = cursor.fetchall()

    # Convert the data to a list of dictionaries for easier handling in the template
    columns = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']
    patient_data = [dict(zip(columns, row)) for row in data]

    return render_template('patient_view_data.html', data=patient_data)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/make_prediction', methods=['GET', 'POST'])
@doctor_required
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

        # Print out the structure of the first row
        print("Database row structure:")
        print(data[0])
        print("Number of columns in database:", len(data[0]))

        # Get column names from the database
        cursor.execute("DESCRIBE heart_data")
        columns = [column[0] for column in cursor.fetchall()]
        print("Columns from database:", columns)
        print("Number of columns:", len(columns))

        # Convert to pandas DataFrame
        df = pd.DataFrame(data, columns=columns)

        print("DataFrame columns:", df.columns)
        print("DataFrame shape:", df.shape)

        # Prepare the features and target
        X = df.drop(['id', 'patient_id', 'output'], axis=1, errors='ignore')
        y = df['output']

        print("Features (X) columns:", X.columns)
        print("Features (X) shape:", X.shape)

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
@doctor_required
def predict():
    if request.method == 'POST':
        # Fetch all data from the database
        cursor.execute("SELECT * FROM heart_data")
        data = cursor.fetchall()

        # Print out the structure of the first row
        print("Database row structure:")
        print(data[0])
        print("Number of columns in database:", len(data[0]))

        # Get column names from the database
        cursor.execute("DESCRIBE heart_data")
        columns = [column[0] for column in cursor.fetchall()]
        print("Columns from database:", columns)
        print("Number of columns:", len(columns))

        # Convert to pandas DataFrame
        df = pd.DataFrame(data, columns=columns)

        print("DataFrame columns:", df.columns)
        print("DataFrame shape:", df.shape)

        # Prepare the features and target
        X = df.drop(['id', 'patient_id', 'output'], axis=1, errors='ignore')
        y = df['output']

        print("Features (X) columns:", X.columns)
        print("Features (X) shape:", X.shape)
        print("Target (y) shape:", y.shape)

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
@doctor_required
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

        return redirect(url_for('doctor_dashboard'))

@app.route('/submit_form')
@doctor_required
def submit_form():
    return render_template('submit_form.html')


@app.route('/data')
@doctor_required
def data():
    cursor.execute("SELECT * FROM heart_data")
    data = cursor.fetchall()
    return render_template('data.html', data=data)


@app.route('/import_csv', methods=['GET', 'POST'])
@doctor_required
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
@doctor_required
def analyze():
    # Fetch data from MySQL
    cursor.execute("SELECT age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall, output FROM heart_data")
    data = cursor.fetchall()

    # Convert data to DataFrame
    columns = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']
    df = pd.DataFrame(data, columns=columns)

    # Basic statistics
    total_patients = len(df)
    heart_disease_count = df['output'].sum()
    healthy_count = total_patients - heart_disease_count

    # Features and target
    X = df.drop('output', axis=1)
    y = df['output']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(model.coef_[0])
    }).sort_values('importance', ascending=False)

    # Create visualizations
    plt.figure(figsize=(10, 6))

    # Pie chart for heart disease distribution
    plt.subplot(2, 2, 1)
    plt.pie([heart_disease_count, healthy_count], labels=['Heart Disease', 'Healthy'], autopct='%1.1f%%')
    plt.title('Distribution of Heart Disease')

    # Bar plot for feature importance
    plt.subplot(2, 2, 2)
    sns.barplot(x='importance', y='feature', data=feature_importance.head(5))
    plt.title('Top 5 Important Features')

    # Age distribution
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='age', hue='output', element='step', stat='density', common_norm=False)
    plt.title('Age Distribution')

    # Cholesterol vs Max Heart Rate
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='chol', y='thalachh', hue='output')
    plt.title('Cholesterol vs Max Heart Rate')

    plt.tight_layout()

    # Save plots to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')

    return render_template('analysis.html',
                           accuracy=accuracy,
                           plot_url=plot_url,
                           total_patients=total_patients,
                           heart_disease_count=heart_disease_count,
                           healthy_count=healthy_count,
                           feature_importance=feature_importance.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)
