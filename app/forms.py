from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from wtforms.widgets import TextArea

class TweetForm(FlaskForm):
    tweet = StringField('Enter your tweet:', widget=TextArea(), validators=[DataRequired()])
    submit = SubmitField('Tweet')
