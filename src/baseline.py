from pathlib import Path

import pandas as pd
import yaml
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

N = 50000
analyzer = SentimentIntensityAnalyzer()
config_path = Path(__file__).with_name("config.yaml")
with config_path.open(encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

raw_dataset_path = Path(config["dataset_path"])
processed_dataset_path = (
    config_path.parent.parent
    / "data"
    / "processed"
    / f"{raw_dataset_path.stem}.csv"
)
df = pd.read_csv(processed_dataset_path)
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