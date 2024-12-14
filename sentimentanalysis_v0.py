import nltk
import ssl
import matplotlib.pyplot as plt

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('vader_lexicon')

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)

    # Get the sentiment polarity (-1 to 1: negative to positive)
    sentiment_polarity = blob.sentiment.polarity
    print("sentiment_polarity", sentiment_polarity)

    # Classify the sentiment as positive, negative, or neutral
    if sentiment_polarity > 0:
        return "Positive"
    elif sentiment_polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Create a SentimentIntensityAnalyzer object
input_text = input("Enter your text: ")
sia = SentimentIntensityAnalyzer()

scores = sia.polarity_scores(input_text)
posscore = scores.get('pos')
negscore = scores.get('neg')
neuscore = scores.get('neu')
compscore = scores.get('compound')
print(compscore)
# Print the sentiment scores
print(f"scores: {scores}")
if compscore > 0:
    print("Positive")
else:
   print("Negative")

# Create a bar plot using Matplotlib
ax = plt.axes()
ax.set_xticks([1,2,3,4])
if (scores.get('compound')<0):
   compcolor='red'
else:
  compcolor='green'
ax.set_xticklabels(['Negative','Neutral','Positive','Net Score'])
plt.bar(1,negscore,color='orange', edgecolor='black')
plt.bar(2,neuscore,color='yellow',edgecolor='black')
plt.bar(3,posscore, color='beige',edgecolor='black')
plt.bar(4,compscore,color=compcolor,edgecolor='black')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis using VADER Lexicon')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


