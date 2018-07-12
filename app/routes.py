import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib

#word2vec disabled - too slow
#from gensim.models import Word2Vec

from nltk.tokenize import RegexpTokenizer

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

        regexp_tok = RegexpTokenizer(r'\w+')
        sentence_tokenized = regexp_tok.tokenize(sentence)
	
        vect_bow = joblib.load('models/vect_bow.pkl')
        vect_tfidf = joblib.load('models/vect_tfidf.pkl')
        ocsvm = joblib.load('models/ocsvm.pkl')
	
	#Sentence BOW and TF-IDF
        sentence_bow = vect_bow.transform([sentence])
        sentence_tfidf = vect_tfidf.transform([sentence])
	
	#Cosine Similarity BOW
        trump_bow = joblib.load('models/trump_bow.pkl')	
        cos_dists_bow = cosine_similarity(trump_bow, sentence_bow)
        max_cos_dist_bow = round(np.max(cos_dists_bow), 4)
        max_cos_dist_bow = max_cos_dist_bow * 100

	#Cosine Similarity TF-IDF
        trump_tfidf = joblib.load('models/trump_tfidf.pkl')
        cos_dists_tfidf = cosine_similarity(trump_tfidf, sentence_tfidf)
        max_cos_dist_tfidf = round(np.max(cos_dists_tfidf), 4)
        max_cos_dist_tfidf = max_cos_dist_tfidf * 100

	#Mean Cosine Similarity
        mean_cos_dist = np.mean([np.max(cos_dists_bow), np.max(cos_dists_tfidf)])
        cos_dist_prediction = round(mean_cos_dist, 4)
        cos_dist_prediction = round(cos_dist_prediction * 100, 2)

	#OneClassSVM
        ocsvm_pred_bow = ocsvm.predict(sentence_bow)[0]
        ocsvm_pred_tfidf = ocsvm.predict(sentence_tfidf)[0]
        ocsvm_pred = np.random.choice([ocsvm_pred_bow, ocsvm_pred_tfidf])

        if ocsvm_pred_bow == -1:
            ocsvm_pred_bow = "-1 (Negative)"
        else:
            ocsvm_pred_bow = "1 (Positive)"

        if ocsvm_pred_tfidf == -1:
            ocsvm_pred_tfidf = "-1 (Negative)"
        else:
            ocsvm_pred_tfidf = "1 (Positive)"        
        

	#Word2Vec - not active because of the low speed :(
        #df_train_tokenized = joblib.load('models/df_train_tokenized.pkl')
        
        #word2vec_model = joblib.load('models/w2v.pkl')
        
        #regexp_tknzr = RegexpTokenizer(r'\w+')

        #wmd_list = []
        #for tweet in df_train_tokenized:
        #    wmd = word2vec_model.wv.wmdistance(tweet, sentence_tokenized)
        #    wmd_list.append(wmd)

        #wmd_pred = np.max(wmd_list)

        if ocsvm_pred == -1 and cos_dist_prediction < 30:
            prediction = "You`re not Donald J. Trump!\n Catch the imposter!"
        elif ocsvm_pred == -1 and cos_dist_prediction >= 30 and cos_dist_prediction < 50:
            prediction = "Nice try, imitator!\n But you`re not Donald! Ha-haaa!"
        elif ocsvm_pred == 1 and cos_dist_prediction >= 50 and cos_dist_prediction < 75:
            prediction = "Hmmm... Your tweet looks like Trump`s one!"
        elif ocsvm_pred == 1 and cos_dist_prediction >= 75 and cos_dist_prediction <= 90:
            prediction = "It`s almost obviously that you`re Mr. Trump!"
        elif cos_dist_prediction == 0:
            prediction = "No chance. No Trump here."
        elif (ocsvm_pred == 1 and cos_dist_prediction < 50) or (ocsvm_pred == -1 and cos_dist_prediction >= 50 and cos_dist_prediction <= 90):
            prediction = "Well done, imitator!\n Models show different results and I can`t decide whether you are Trump or not :("
        elif cos_dist_prediction > 90:
            prediction = "Donald, stop tweeting!\n America needs you!"

    return render_template('result.html', title = 'TweetLikeTrump - Score', sentence = sentence, prediction = prediction, cos_dist_prediction = cos_dist_prediction, max_cos_dist_bow = max_cos_dist_bow, max_cos_dist_tfidf = max_cos_dist_tfidf, ocsvm_pred_bow = ocsvm_pred_bow, ocsvm_pred_tfidf = ocsvm_pred_tfidf)
