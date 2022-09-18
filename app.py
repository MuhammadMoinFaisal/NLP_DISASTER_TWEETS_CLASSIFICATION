from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib


app = Flask(__name__)


with open('model_pkl.pkl' , 'rb') as f:
    model = pickle.load(f)

data= pd.read_csv("train.csv")

data_train = data["text"]

cv = CountVectorizer(stop_words = 'english', max_features = 17171)

X = cv.fit_transform(data_train) 

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = model.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
    #app.run(host = '127.0.0.1', port = 9000)
