from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and preprocessors
model = joblib.load('final_model.joblib')
scaler = joblib.load('scaler.joblib')
imputer = joblib.load('imputer.joblib')


@app.route('/')
def home():
    # Return the HTML page for home
    return render_template("power_outage.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input features from the form
        int_features = [float(x) for x in request.form.values()]

        # Prepare the feature array for prediction
        features_array = np.array(int_features).reshape(1, -1)

        # Impute missing values and scale the features
        features_imputed = imputer.transform(features_array)
        features_scaled = scaler.transform(features_imputed)

        # Predict the probability of a power outage
        prediction = model.predict(features_scaled)

        # Format the probability as a percentage with 2 decimal places
        probability_percent = '{0:.2f}%'.format(probabilityInt * 100)

        # Render the HTML page with the prediction result
        if prediction[0] == 1:
            pred = True
            message = f'Power outage is likely with a probability of {probability_percent}.'
            print("The prediction is Yes")
        else:
            pred = False
            message = f'The probability of a power outage occurring is {probability_percent}.'
            print("The prediction is No")

        return render_template('power_outage.html', prediction=message, probability=probabilityInt)

    except Exception as e:
        # In case of an error, print it to the console and render an error message on the page
        print(e)
        return render_template('power_outage.html', prediction='An error occurred while making the prediction.')


if __name__ == '__main__':
    app.run(debug=True)
