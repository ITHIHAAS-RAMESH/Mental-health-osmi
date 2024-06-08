from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)





model = pickle.load(open('final_SVM.pkl','rb'))
s = pickle.load(open('scalar.pkl','rb'))


@app.route('/')
def Home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    age = final[0][0].reshape(-1,1)
    print(age)
    test_scaled_set = s.transform(age)
    print(float(test_scaled_set))
    final[0][0] = float(test_scaled_set)
    
    
    output = model.predict(final)
    print(output)
    
    if output == 1:
        return render_template("result1.html")
    else:
        return render_template("result0.html")
    
if __name__ == '__main__':
    app.run(debug=True)

