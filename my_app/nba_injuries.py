from flask import Flask, request, render_template
#import build_model
import cPickle as pickle

app = Flask(__name__)

# home page
@app.route('/')
def home():
    return render_template('index.html')

# prediction app
@app.route('/predict', methods=['POST'])
def predict():
    text = str(request.form['user_input'])
    X = vect.transform([text])
    return render_template('predict.html', prediction=model.predict(X))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
