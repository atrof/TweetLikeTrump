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
	
        sentence_tfidf = vect.transform([sentence])
        ocsvm_pred = ocsvm.predict(sentence_tfidf)[0]

        if ocsvm_pred == -1:
            ocsvm_prediction = "No, you tweet not like Donald J. Trump!"
        else:
            ocsvm_prediction = "Hi, Mr. President!"

    return render_template('result.html', title = 'TweetLikeTrump - Score', sentence = sentence, ocsvm_prediction = ocsvm_prediction)
