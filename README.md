# TweetLikeTrump
is a Flask-based web application which measures the degree of similarity between user's custom text and Donald J. Trump's tweetying style. Funny mix of machine learning and some NLP techniques. 

MVP is available online - [TweetLikeTrump on Heroku](https://tweetliketrump.herokuapp.com).

### Task
Determining the similarity between Trump's and user's style of tweetting.

### Data Collection
Dataset is a base of [Donald J. Trump tweets](https://www.twitter.com/realdonaldtrump). Data was collected via Twitter API  using earlier simple version of the [3200tweets](https://github.com/atrof/3200tweets) script.

### Data Preprocessing
* TF-IDF

### Modeling
Different approaches are used:
1. Cosine similarity
2. One-Class Classification

### Deployment
MVP is online - [TweetLikeTrump on Heroku](https://tweetliketrump.herokuapp.com). Changes are implemented automatically via Github.

## TODO
* **Data Preprocessing**: remove @mentions, #hashtags & URLs from tweets; stemming
* **Modeling**: binary classification, multi-label classification
* **Backend**: scheduler setup for updating models
* **Frontend**: styling, AJAX for single-page app (SPA)
