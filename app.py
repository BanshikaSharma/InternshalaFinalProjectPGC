from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('app/salary.csv')

# Create a machine learning model
X = df[['role', 'college','city','previous_job']]
y = df['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    role = data['role']
    college = data['college']
    previous_job = data['previous_job']
    city = data['city']
    input_data = pd.DataFrame({'role': [role], 'college': [college]})
    prediction = model.predict(input_data)
    return jsonify({'predicted_salary': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
