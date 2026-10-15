"""Use the removed_mixed_words function in word_tilt.py to create a trimmed version of each review in a copy of dataset"""
import pandas as pd
import os
import sys

# Pathfinding to import the word_tilt.py file so that we can apply it to the csv file
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(project_root)

import investigation.word_tilt as word_tilt

output_dataset_path = "trimmed_IMDB_Dataset.csv"

current_dir = os.path.dirname(__file__)

csv_path = os.path.abspath(
    os.path.join(current_dir, "..", "raw", "IMDB_Dataset.csv")
)

original_reviews = pd.read_csv(csv_path)
# Only looks at the first 5 reviews, remove head to make it look at all the reviews (takes time)
trimmed_reviews = original_reviews.head(5).copy()

trimmed_reviews["review"] = trimmed_reviews["review"].apply(word_tilt.remove_mixed_words)

trimmed_reviews.to_csv(output_dataset_path, index=False)
# print("File was saved successfully to processed folder")

# Test to see csv copy
# print(trimmed_reviews.head(5).to_string())
