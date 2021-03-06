import pandas as pd
import numpy as np
import re
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib

from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer

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

		#CLEANING
		sentence_new = pd.Series(sentence)
		#Remove URLs, mentions, numbers
		sentence_cleaned = sentence_new.replace(re.compile(r"http.?://[^\s]+[\s]?"))
		sentence_cleaned = sentence_new.replace(re.compile(r"@[^\s]+[\s]?"))
		sentence_cleaned = sentence_new.replace(re.compile(r"\s?[0-9]+\.?[0-9]*"))
		#Remove punctuation and convert hashtags to normal words
		for remove in map(lambda r: re.compile(re.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$", "@", "%", "^", "*", "(", ")", "{", "}", "[", "]", "|", "/", "\\", ">", "<", "-", "!", "?", ".", "'", "--", "---", "#", "..."]):
			sentence_cleaned.replace(remove, " ", inplace=True)

		sentence_cleaned = sentence_cleaned[0]

		#STEMMING
		stemmer = PorterStemmer()
		sentence_stemmed = ' '.join(stemmer.stem(word) for word in sentence_cleaned.lower().split())

		#Cleaned and stemmed sentences
		#sentence_stemmed = sentence_stemmed[0]
		#sentence_cleaned = sentence_cleaned[0]

		#LOADING TRAINING DATA
		#bow
		trump_bow = joblib.load('models/trump_bow.pkl')
		trump_cleaned_bow = joblib.load('models/trump_cleaned_bow.pkl')
		trump_stemmed_bow = joblib.load('models/trump_stemmed_bow.pkl')
		#tfidf
		trump_tfidf = joblib.load('models/trump_tfidf.pkl')
		trump_cleaned_tfidf = joblib.load('models/trump_cleaned_tfidf.pkl')
		trump_stemmed_tfidf = joblib.load('models/trump_stemmed_tfidf.pkl')

		#LOADING MODELS
		#bow vectorizers
		vect_bow = joblib.load('models/vect_bow.pkl')
		vect_cleaned_bow = joblib.load('models/vect_cleaned_bow.pkl')
		vect_stemmed_bow = joblib.load('models/vect_stemmed_bow.pkl')
		#tfidf vectorizers
		vect_tfidf = joblib.load('models/vect_tfidf.pkl')
		vect_cleaned_tfidf = joblib.load('models/vect_cleaned_tfidf.pkl')
		vect_stemmed_tfidf = joblib.load('models/vect_stemmed_tfidf.pkl')
		#OneClassSVM
		ocsvm_bow = joblib.load('models/ocsvm_bow.pkl')
		ocsvm_cleaned_bow = joblib.load('models/ocsvm_cleaned_bow.pkl')
		ocsvm_stemmed_bow = joblib.load('models/ocsvm_stemmed_bow.pkl')
		ocsvm_tfidf = joblib.load('models/ocsvm_tfidf.pkl')
		ocsvm_cleaned_tfidf = joblib.load('models/ocsvm_cleaned_tfidf.pkl')
		ocsvm_stemmed_tfidf = joblib.load('models/ocsvm_stemmed_tfidf.pkl')

		#SENTENCE VECTORIZING
		#Source sentence BOW and TF-IDF
		sentence_bow = vect_bow.transform([sentence])
		sentence_tfidf = vect_tfidf.transform([sentence])
		#Cleaned sentence BOW and TF-IDF
		sentence_cleaned_bow = vect_cleaned_bow.transform([sentence_cleaned])
		sentence_cleaned_tfidf = vect_cleaned_tfidf.transform([sentence_cleaned])
		#Stemmed sentence BOW and TF-IDF
		sentence_stemmed_bow = vect_stemmed_bow.transform([sentence_stemmed])
		sentence_stemmed_tfidf = vect_stemmed_tfidf.transform([sentence_stemmed])

		#SOURCE SENTENCE (DISPLAYED ON SCORE SCREEN)
		#Cosine Similarity BOW
		cos_dists_bow = cosine_similarity(trump_bow, sentence_bow)
		max_cos_dist_bow = round(np.max(cos_dists_bow), 4)
		max_cos_dist_bow = max_cos_dist_bow * 100
		#Cosine Similarity TF-IDF
		cos_dists_tfidf = cosine_similarity(trump_tfidf, sentence_tfidf)
		max_cos_dist_tfidf = round(np.max(cos_dists_tfidf), 4)
		max_cos_dist_tfidf = max_cos_dist_tfidf * 100
		#Mean Cosine Similarity
		mean_cos_dist = np.mean([np.max(cos_dists_bow), np.max(cos_dists_tfidf)])
		mean_cos_dist_pred = round(mean_cos_dist, 4)
		mean_cos_dist_pred = round(mean_cos_dist_pred * 100, 2)
		#OneClassSVM
		ocsvm_pred_bow = ocsvm_bow.predict(sentence_bow)[0]
		ocsvm_pred_tfidf = ocsvm_tfidf.predict(sentence_tfidf)[0]
		ocsvm_pred = np.random.choice([ocsvm_pred_bow, ocsvm_pred_tfidf])

		#CLEANED SENTENCE (DISPLAYED IN DETAILS)
		#Cosine Similarity BOW
		cos_dists_cleaned_bow = cosine_similarity(trump_cleaned_bow, sentence_cleaned_bow)
		max_cos_dist_cleaned_bow = round(np.max(cos_dists_cleaned_bow), 4)
		max_cos_dist_cleaned_bow = max_cos_dist_cleaned_bow * 100
		#Cosine Similarity TF-IDF
		cos_dists_cleaned_tfidf = cosine_similarity(trump_cleaned_tfidf, sentence_cleaned_tfidf)
		max_cos_dist_cleaned_tfidf = round(np.max(cos_dists_cleaned_tfidf), 4)
		max_cos_dist_cleaned_tfidf = max_cos_dist_cleaned_tfidf * 100
		#Mean Cosine Similarity
		mean_cos_dist_cleaned = np.mean([np.max(cos_dists_cleaned_bow), np.max(cos_dists_cleaned_tfidf)])
		mean_cos_dist_cleaned_pred = round(mean_cos_dist_cleaned, 4)
		mean_cos_dist_cleaned_pred = round(mean_cos_dist_cleaned_pred * 100, 2)
		#OneClassSVM
		ocsvm_pred_cleaned_bow = ocsvm_cleaned_bow.predict(sentence_cleaned_bow)[0]
		ocsvm_pred_cleaned_tfidf = ocsvm_cleaned_tfidf.predict(sentence_cleaned_tfidf)[0]

		#STEMMED SENTENCE (DISPLAYED IN DETAILS)
		#Cosine Similarity BOW
		cos_dists_stemmed_bow = cosine_similarity(trump_stemmed_bow, sentence_stemmed_bow)
		max_cos_dist_stemmed_bow = round(np.max(cos_dists_stemmed_bow), 4)
		max_cos_dist_stemmed_bow = max_cos_dist_stemmed_bow * 100
		#Cosine Similarity TF-IDF
		cos_dists_stemmed_tfidf = cosine_similarity(trump_stemmed_tfidf, sentence_stemmed_tfidf)
		max_cos_dist_stemmed_tfidf = round(np.max(cos_dists_stemmed_tfidf), 4)
		max_cos_dist_stemmed_tfidf = max_cos_dist_stemmed_tfidf * 100
		#Mean Cosine Similarity
		mean_cos_dist_stemmed = np.mean([np.max(cos_dists_stemmed_bow), np.max(cos_dists_stemmed_tfidf)])
		mean_cos_dist_stemmed_pred = round(mean_cos_dist_stemmed, 4)
		mean_cos_dist_stemmed_pred = round(mean_cos_dist_stemmed_pred * 100, 2)
		#OneClassSVM
		ocsvm_pred_stemmed_bow = ocsvm_stemmed_bow.predict(sentence_stemmed_bow)[0]
		ocsvm_pred_stemmed_tfidf = ocsvm_stemmed_tfidf.predict(sentence_stemmed_tfidf)[0]

		#Output messages
		if ocsvm_pred == -1 and mean_cos_dist_pred < 30:
		    prediction = "You`re not Donald J. Trump!\n Catch the imposter!"
		elif ocsvm_pred == -1 and mean_cos_dist_pred >= 30 and mean_cos_dist_pred < 50:
		    prediction = "Nice try, imitator!\n But you`re not Donald! Ha-haaa!"
		elif ocsvm_pred == 1 and mean_cos_dist_pred >= 50 and mean_cos_dist_pred < 75:
		    prediction = "Hmmm... Your tweet looks like Trump`s one!"
		elif ocsvm_pred == 1 and mean_cos_dist_pred >= 75 and mean_cos_dist_pred <= 90:
		    prediction = "It`s almost obviously that you`re Mr. Trump!"
		elif mean_cos_dist_pred == 0:
		    prediction = "No chance. No Trump here."
		elif (ocsvm_pred == 1 and mean_cos_dist_pred < 50) or (ocsvm_pred == -1 and mean_cos_dist_pred >= 50 and mean_cos_dist_pred <= 90):
		    prediction = "Well done, imitator!\n Models show different results and I can`t decide whether you are Trump or not :("
		elif mean_cos_dist_pred > 90:
		    prediction = "Donald, stop tweeting!\n America needs you!"

		#Values to string for correct view
		max_cos_dist_bow = "{}".format(max_cos_dist_bow)
		max_cos_dist_cleaned_bow = "{}".format(max_cos_dist_cleaned_bow)
		max_cos_dist_stemmed_bow = "{}".format(max_cos_dist_stemmed_bow)
		max_cos_dist_tfidf = "{}".format(max_cos_dist_tfidf)
		max_cos_dist_cleaned_tfidf = "{}".format(max_cos_dist_cleaned_tfidf)
		max_cos_dist_stemmed_tfidf = "{}".format(max_cos_dist_stemmed_tfidf)
		mean_cos_dist_pred = "{}".format(mean_cos_dist_pred)
		ocsvm_pred_bow = "{}".format(ocsvm_pred_bow)
		ocsvm_pred_cleaned_bow = "{}".format(ocsvm_pred_cleaned_bow)
		ocsvm_pred_stemmed_bow = "{}".format(ocsvm_pred_stemmed_bow)
		ocsvm_pred_tfidf = "{}".format(ocsvm_pred_tfidf)
		ocsvm_pred_cleaned_tfidf = "{}".format(ocsvm_pred_cleaned_tfidf)
		ocsvm_pred_stemmed_tfidf = "{}".format(ocsvm_pred_stemmed_tfidf)
		ocsvm_pred = "{}".format(ocsvm_pred)

	return render_template('result.html', title = 'TweetLikeTrump - Score', sentence = sentence, prediction = prediction, mean_cos_dist_pred = mean_cos_dist_pred, max_cos_dist_bow = max_cos_dist_bow, max_cos_dist_tfidf = max_cos_dist_tfidf, ocsvm_pred_bow = ocsvm_pred_bow, ocsvm_pred_tfidf = ocsvm_pred_tfidf, max_cos_dist_cleaned_bow = max_cos_dist_cleaned_bow, max_cos_dist_stemmed_bow = max_cos_dist_stemmed_bow, max_cos_dist_cleaned_tfidf = max_cos_dist_cleaned_tfidf,max_cos_dist_stemmed_tfidf = max_cos_dist_stemmed_tfidf, ocsvm_pred_cleaned_bow = ocsvm_pred_cleaned_bow, ocsvm_pred_stemmed_bow = ocsvm_pred_stemmed_bow, ocsvm_pred_cleaned_tfidf = ocsvm_pred_cleaned_tfidf, ocsvm_pred_stemmed_tfidf = ocsvm_pred_stemmed_tfidf)
