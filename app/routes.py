import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

from flask import render_template, flash, redirect, url_for, request
from app.forms import TweetForm
from app import app

@app.route('/')
@app.route('/index')
def index():
	user = {'username': 'Trump Imitator'}
	#form = TweetForm()
	return render_template('index.html', title='TweetLikeTrump', user=user)

@app.route('/tweet', methods=['GET', 'POST'])
def tweet():
    form = TweetForm()
    if form.validate_on_submit():
        flash(form.tweet.data)
        return redirect(url_for('result'))
    return render_template('tweet.html', title = 'TweetLikeTrump - Tweet Now!', form = form)

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        sentence = request.form['tweet']
	
        vect = joblib.load('vect.pkl')
        ocsvm = joblib.load('ocsvm.pkl')
	
	#OneClassSVM
        sentence_tfidf = vect.transform([sentence])
        ocsvm_pred = ocsvm.predict(sentence_tfidf)[0]
	
	#Cosine Similarity
        trump_tfidf = joblib.load('trump_tfidf.pkl')
        cos_dists = cosine_similarity(trump_tfidf, sentence_tfidf)
        max_cos_dist = round(np.max(cos_dists), 3)

        cos_dist_prediction = max_cos_dist * 100

        if ocsvm_pred == -1 and cos_dist_prediction < 30:
            prediction = "You`re not Donald J. Trump!\nCatch the imposter!"
        elif ocsvm_pred == -1 and cos_dist_prediction >= 30 and cos_dist_prediction < 50:
            prediction = "Nice try, imitator!\nBut you`re not Donald! Ha-haaa!"
        elif ocsvm_pred == 1 and cos_dist_prediction >= 50 and cos_dist_prediction < 75:
            prediction = "Hmmm... Your tweet looks like Trump`s one!"
        elif ocsvm_pred == 1 and cos_dist_prediction >= 75 and cos_dist_prediction <= 90:
            prediction = "It`s almost obviously that you`re Mr. Trump!"
        elif (ocsvm_pred == 1 and cos_dist_prediction < 50) or (ocsvm_pred == -1 and cos_dist_prediction >= 50 and cos_dist_prediction <= 90):
            prediction = "Well done, imitator!\n I can`t decide whether you are Trump or not :("
        elif cos_dist_prediction > 90:
            prediction = "Donald, stop tweeting!\nAmerica needs you!"

    return render_template('result.html', title = 'TweetLikeTrump - Score', sentence = sentence, prediction = prediction, cos_dist_prediction = cos_dist_prediction)
