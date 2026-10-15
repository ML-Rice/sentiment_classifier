import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

N = 50000
analyzer = SentimentIntensityAnalyzer()
df = pd.read_csv('data/processed/IMDB_Dataset.csv')
df['binary_sentiment'] = (df['sentiment'] == 'positive').astype(int)

def predict(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0:
        return 1
    else: 
        return 0

y_pred = df['review'].head(N).apply(predict).to_numpy()
y_true = df['binary_sentiment'].head(N).to_numpy()

tp = ((y_true == 1) & (y_pred == 1)).sum()
fp = ((y_true == 0) & (y_pred == 1)).sum()
fn = ((y_true == 1) & (y_pred == 0)).sum()
tn = ((y_true == 0) & (y_pred == 0)).sum()

acc = (tn+tp)/(tp+tn+fp+fn)
pre = (tp)/(tp+fp)
rec = (tp)/(tp+fn)
f1 = (2*pre*rec)/(rec+pre)

print(f"Accuracy: {acc:.3f}, Precision: {pre:.3f}, Recall: {rec:.3f}, f1 score: {f1:.3f}")
print(f"TP: {tp:.3f}, TN: {tn:.3f}, FP: {fp:.3f}, FN: {fn:.3f}")