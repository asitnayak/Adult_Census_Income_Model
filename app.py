from flask import Flask, request, render_template
from model import Model
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("Logs/main.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def prediction():
    try:
        print("Hi")

        logger.debug("Starting Prediction process.")

        input_data = [i for i in request.form.values()]
        print("Captured input data.")
        logger.debug("Captured input data.")
        obj = Model(input_data)
        print("Model object created.")
        logger.debug("Model object created.")
        obj.generate_dataframe()
        print("Input DataFrame generated.")
        logger.debug("Input DataFrame generated.")
        obj.preprocessor()
        print("Data preprocessed.")
        logger.debug("Data preprocessed.")
        y_hat = obj.soft_voting_prediction()
        print("Received prediction.")
        logger.debug("Received prediction.")
        #print(y_hat)
        result = "Try Again"

        if y_hat == 1:
            result = "Income is more than $50K"
        elif y_hat == 0:
            result = "Income is less than $50K"

        return render_template('index.html', result=result)

    except:
        print("Error occurred while predicting.")
        logger.debug("Error occurred while predicting.")
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)