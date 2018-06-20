from flask import render_template, flash, redirect, url_for
from app.forms import TweetForm
from app import app

@app.route('/')
@app.route('/index')
def index():
	user = {'username': 'Trump Imitator'}
	form = TweetForm()
	return render_template('index.html', title='TweetLikeTrump', user=user, form=form)

@app.route('/tweet', methods=['GET', 'POST'])
def tweet():
    form = TweetForm()
    if form.validate_on_submit():
        flash(form.tweet.data)
        return redirect(url_for('result'))
    return render_template('tweet.html', title = 'TweetLikeTrump - Tweet Now!', form = form)

@app.route('/result')
def result():
    form = TweetForm()
    return render_template('result.html', title = 'TweetLikeTrump - Score', form = form)
