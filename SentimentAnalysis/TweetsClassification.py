from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

emoticons = {
    ':*': '<kiss>',
    ':-*': '<kiss>',
    ':x': '<kiss>',
    ':-)': '<happy>',
    ':-))': '<happy>',
    ':-)))': '<happy>',
    ':-))))': '<happy>',
    ':-)))))': '<happy>',
    ':-))))))': '<happy>',
    ':)': '<happy>',
    ':))': '<happy>',
    ':)))': '<happy>',
    ':))))': '<happy>',
    ':)))))': '<happy>',
    ':))))))': '<happy>',
    ':)))))))': '<happy>',
    ':o)': '<happy>',
    ':]': '<happy>',
    ':3': '<happy>',
    ':c)': '<happy>',
    ':>': '<happy>',
    '=]': '<happy>',
    '8)': '<happy>',
    '=)': '<happy>',
    ':}': '<happy>',
    ':^)': '<happy>',
    '|;-)': '<happy>',
    ":'-)": '<happy>',
    ":')": '<happy>',
    '\o/': '<happy>',
    '*\\0/*': '<happy>',
    ':-D': '<laugh>',
    ':D': '<laugh>',
    '8-D': '<laugh>',
    '8D': '<laugh>',
    'x-D': '<laugh>',
    'xD': '<laugh>',
    'X-D': '<laugh>',
    'XD': '<laugh>',
    '=-D': '<laugh>',
    '=D': '<laugh>',
    '=-3': '<laugh>',
    '=3': '<laugh>',
    'B^D': '<laugh>',
    '>:[': '<sad>',
    ':-(': '<sad>',
    ':-((': '<sad>',
    ':-(((': '<sad>',
    ':-((((': '<sad>',
    ':-(((((': '<sad>',
    ':-((((((': '<sad>',
    ':-(((((((': '<sad>',
    ':(': '<sad>',
    ':((': '<sad>',
    ':(((': '<sad>',
    ':((((': '<sad>',
    ':(((((': '<sad>',
    ':((((((': '<sad>',
    ':(((((((': '<sad>',
    ':((((((((': '<sad>',
    ':-c': '<sad>',
    ':c': '<sad>',
    ':-<': '<sad>',
    ':<': '<sad>',
    ':-[': '<sad>',
    ':[': '<sad>',
    ':{': '<sad>',
    ':-||': '<sad>',
    ':@': '<sad>',
    ":'-(": '<sad>',
    ":'(": '<sad>',
    'D:<': '<sad>',
    'D:': '<sad>',
    'D8': '<sad>',
    'D;': '<sad>',
    'D=': '<sad>',
    'DX': '<sad>',
    'v.v': '<sad>',
    "D-':": '<sad>',
    '(>_<)': '<sad>',
    ':|': '<sad>',
    '>:O': '<surprise>',
    ':-O': '<surprise>',
    ':-o': '<surprise>',
    ':O': '<surprise>',
    '°o°': '<surprise>',
    'o_O': '<surprise>',
    'o_0': '<surprise>',
    'o.O': '<surprise>',
    'o-o': '<surprise>',
    '8-0': '<surprise>',
    '|-O': '<surprise>',
    ';-)': '<wink>',
    ';)': '<wink>',
    '*-)': '<wink>',
    '*)': '<wink>',
    ';-]': '<wink>',
    ';]': '<wink>',
    ';D': '<wink>',
    ';^)': '<wink>',
    ':-,': '<wink>',
    '>:P': '<tong>',
    ':-P': '<tong>',
    ':P': '<tong>',
    'X-P': '<tong>',
    'x-p': '<tong>',
    ':-p': '<tong>',
    ':p': '<tong>',
    '=p': '<tong>',
    ':-Þ': '<tong>',
    ':Þ': '<tong>',
    ':-b': '<tong>',
    ':b': '<tong>',
    ':-&': '<tong>',
    '>:\\': '<annoyed>',
    '>:/': '<annoyed>',
    ':-/': '<annoyed>',
    ':-.': '<annoyed>',
    ':/': '<annoyed>',
    ':\\': '<annoyed>',
    '=/': '<annoyed>',
    '=\\': '<annoyed>',
    ':L': '<annoyed>',
    '=L': '<annoyed>',
    ':S': '<annoyed>',
    '>.<': '<annoyed>',
    ':-|': '<annoyed>',
    '<:-|': '<annoyed>',
    ':-X': '<seallips>',
    ':X': '<seallips>',
    ':-#': '<seallips>',
    ':#': '<seallips>',
    'O:-)': '<angel>',
    '0:-3': '<angel>',
    '0:3': '<angel>',
    '0:-)': '<angel>',
    '0:)': '<angel>',
    '0;^)': '<angel>',
    '>:)': '<devil>',
    '>:D': '<devil>',
    '>:-D': '<devil>',
    '>;)': '<devil>',
    '>:-)': '<devil>',
    '}:-)': '<devil>',
    '}:)': '<devil>',
    '3:-)': '<devil>',
    '3:)': '<devil>',
    'o/\o': '<highfive>',
    '^5': '<highfive>',
    '>_>^': '<highfive>',
    '^<_<': '<highfive>',
    '<3': '<heart>',
    '^3^': '<smile>',
    "(':": '<smile>',
    " > < ": '<smile>',
    "UvU": '<smile>',
    "uwu": '<smile>',
    'UwU': '<smile>'
}


def open_file(input_file_name):
    words = set()
    f = open(input_file_name, 'r')
    lines = list(f.read().splitlines())
    for i in lines[31:]:
        words.add(i)
    return words


positive = open_file("positive-words.txt")
negative = open_file("negative-words.txt")


def preprocess_normalize(x):
    lem = WordNetLemmatizer()
    word_list = word_tokenize(x)
    x = ' '.join([lem.lemmatize(w) for w in word_list])
    pattern_url = re.compile(
        r"""(?xi)\b(?:(?:https?|ftp|file)://|www\.|ftp\.|pic\.|twitter\.|facebook\.)(?:\([-A-Z0-9+&@#/%=~_|$?!:;,.]*\)|[-A-Z0-9+&@#/%=~_|$?!:;,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])""")
    pattern_punc = re.compile(r"[\-\"`$%^&*(|)/~\[\]{}:;+,._='?!]+")
    return pattern_punc.sub('', pattern_url.sub('', x))


def preprocess_tweets(tweets):
    company = []
    sentiments = []
    date = []
    text = []
    for t in tweets:
        enc = t.encode('ascii', errors='ignore').decode('utf8', errors='ignore')
        split = list(enc.split(',', 3))
        if split[1][1:-1] == 'irrelevant':
            continue

        company.append(split[0][1:-1])
        sentiments.append(split[1][1:-1])
        date.append(split[2][1:-1])
        text.append(preprocess_normalize(split[3][1:-1]))

    return company, sentiments, date, text


def load_csv(input_file_name):
    f = open(input_file_name, 'r', encoding='utf8')
    lines = list(f.read().splitlines())
    return preprocess_tweets(lines[1:])


class TopicClassifier:
    clf = SVC()
    ppl = Pipeline([('tfidf', TfidfVectorizer()),
                    ('clf', clf)])

    def __init__(self):
        pass

    @staticmethod
    def normalize_tweet(x):
        for em in emoticons:
            x = x.replace(em, '')
        return x

    def normalize(self, tweets):
        return list(map(lambda x: ''.join(list(self.normalize_tweet(x))), tweets))

    def predict(self, tweets):
        return self.ppl.predict(self.normalize(tweets))

    def train(self, tweets, company):
        self.ppl.fit(self.normalize(tweets), company)


class SentimentClassifier:
    clf = LinearSVC()
    ppl = Pipeline([('tfidf', TfidfVectorizer()),
                    ('clf', clf)])

    def __init__(self):
        pass

    @staticmethod
    def normalize_tweet(x):
        for em in emoticons:
            x = x.replace(em, emoticons[em])
        for pos in positive:
            x = x.replace(pos, 'positive')
        for neg in negative:
            x = x.replace(neg, 'negative')
        return x

    def normalize(self, tweets):
        return list(map(lambda x: ''.join(list(self.normalize_tweet(x))), tweets))

    def predict(self, tweets):
        return self.ppl.predict(self.normalize(tweets))

    def train(self, tweets, sentiment):
        self.ppl.fit(self.normalize(tweets), sentiment)


if __name__ == '__main__':
    print("TASK RESULTS")
    company_train, sentiment_train, _, text_train = load_csv("Train.csv")
    company_test, sentiment_test, _, text_test = load_csv("Test.csv")

    tcl = TopicClassifier()
    tcl.train(text_train, company_train)
    company_pred = tcl.predict(text_test)
    print(classification_report(company_test, company_pred))

    scl = SentimentClassifier()
    scl.train(text_train, sentiment_train)
    company_pred = scl.predict(text_test)
    print(classification_report(sentiment_test, company_pred))

    print("NEW TWEETS")
    print("Enter the number of new tweets: ")
    num = int(input())
    new_tweets = []
    for i in range(num):
        new_tweets.append(input())
    company_new_tweets, sentiment_new_tweets, _, text_new_tweets = preprocess_tweets(new_tweets)

    if len(text_new_tweets) != 0:
        company_pred = tcl.predict(text_new_tweets)
        sentiment_pred = scl.predict(text_new_tweets)
        for i in range(len(text_new_tweets)):
            print('\'' + text_new_tweets[i] + '\'' + ': ' + company_pred[i] + ', ' + sentiment_pred[i])