# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

from flask import Flask
from financial_api import financial_api
from prediction_api import app as prediction_app

app = Flask(__name__)

# Register blueprints
app.register_blueprint(financial_api, url_prefix='/api/financial')
app.register_blueprint(prediction_app, url_prefix='/api/prediction')

@app.route('/')
def index():
    return {
        'status': 'ok',
        'message': 'Super AI API is running',
        'endpoints': {
            'financial': '/api/financial',
            'prediction': '/api/prediction'
        }
    }

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
