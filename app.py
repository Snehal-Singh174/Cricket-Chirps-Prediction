import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        int_features = int(request.form.get('text'))
        prediction = model.predict([[int_features]])
        prediction1 = model1.predict([[int_features]])
        prediction2 = model2.predict([[int_features]])


        output = round(prediction[0],0)
        output1 = round(prediction1[0], 0)
        output2 = round(prediction2[0], 0)


    return render_template('index.html', p1='OUTPUT:',prediction_text='Number of times cricket chirps predicted by linear regression {}'.format(output),prediction_text1 = 'Number of times cricket chirps predicted by decision tree {}'.format(output1),prediction_text2 = 'Number of times cricket chirps predicted by random forest {}'.format(output2))

if __name__ == "__main__":
    app.run(debug=True)