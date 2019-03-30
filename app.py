from flask import Flask,render_template,url_for,request
from flask_material import Material

# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg
from sklearn.externals import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		age = request.form['age']
		gender = request.form['gender']
		chestpain = request.form['chestpain']
		bloodpressure = request.form['bloodpressure']
		cholestrol=request.form['cholestrol']
		fastingbloodsugar=request.form['fastingbloodsugar']
		restecg=request.form['restecg']
		maxheartrate=request.form['maxheartrate']
		inducedangina=request.form['inducedangina']
		stdepression=request.form['stdepression']
		slope=request.form['slope']
		vessels=request.form['vessels']
		thal=request.form['thal']
		model_choice=request.form['model_choice']
		

		# Clean the data by convert from unicode to float 
		sample_data = [age,gender,chestpain,bloodpressure,cholestrol,fastingbloodsugar,restecg,maxheartrate,inducedangina,stdepression,slope,vessels,thal]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)


		# Reloading the Model
		if model_choice == 'LogisticRegression':
		    LogisticRegression_model = joblib.load('LogisticRegression.pkl')
		    result_prediction = LogisticRegression_model.predict(ex1)
		elif model_choice == 'NaiveBayes':
			NaiveBayes_model = joblib.load('NaiveBayes.pkl')
			result_prediction = NaiveBayes_model.predict(ex1)
		elif model_choice == 'SVM':
			SVM_model = joblib.load('SVM.pkl')
			result_prediction = SVM_model.predict(ex1)
		elif model_choice == 'decisiontree':
			decisiontree_model = joblib.load('decisiontree.pkl')
			result_prediction = decisiontree_model.predict(ex1)
		elif model_choice == 'randomforest':
			randomforest_model = joblib.load('randomforest.pkl')
			result_prediction = randomforest_model.predict(ex1)
		
		
	return render_template('index.html', age=age,
		gender=gender,
		chestpain=chestpain,
		bloodpressure=bloodpressure,
		cholestrol=cholestrol,
		fastingbloodsugar=fastingbloodsugar,
		restecg=restecg,
		maxheartrate=maxheartrate,
		inducedangina=inducedangina,
		stdepression=stdepression,
		slope=slope,
		vessels=vessels,
		thal=thal,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)



if __name__ == '__main__':
	app.run(debug=True)
