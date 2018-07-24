## TweetLikeTrump
is a Flask-based web application which measures the degree of similarity between user's custom text and Donald J. Trump's tweeting style. Funny mix of machine learning and some NLP techniques. 

MVP is available online - [TweetLikeTrump on Heroku](https://tweetliketrump.herokuapp.com)

### Task
Determining the similarity between Trump's and user's style of tweetting.

### Data Collection
Dataset is a base of [Donald J. Trump tweets](https://www.twitter.com/realdonaldtrump). Data was collected via Twitter API  using earlier simple version of the [3200tweets](https://github.com/atrof/3200tweets) script.

### Data Preprocessing
* Data Cleansing: remove mentions, numbers, punctuation, URLs from tweets (Regular Expressions) and convert hashtags to usual words
* Stemming (PorterStemmer)
* Bag-Of-Words (CountVectorizer)
* TF-IDF (TfidfVectorizer)

### Modeling
1. Cosine similarity - mean of BOW & TF-IDF
2. One-Class Classification - random of BOW & TF-IDF
3. Word2Vec - Word Mover\`s Distance - DISABLED (too slow for online performance)

### Deployment
[TweetLikeTrump on Heroku](https://tweetliketrump.herokuapp.com). Changes are implemented automatically via Github.

## TODO
* Scheduler setup for updating models
* AJAX for single-page app (SPA)
