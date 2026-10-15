"""Calculate how strongly words are associated with positive reviews."""

from collections import Counter
from pathlib import Path
import re

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "IMDB_Dataset.csv"
)
MIN_REVIEWS = 5
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def load_reviews(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the IMDb reviews CSV."""
    reviews = pd.read_csv(path)
    required_columns = {"review", "sentiment"}
    missing_columns = required_columns - set(reviews.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required column(s): {missing}")

    reviews = reviews.loc[:, ["review", "sentiment"]].copy()
    reviews["review"] = reviews["review"].fillna("").astype(str)
    reviews["sentiment"] = reviews["sentiment"].str.lower()
    unexpected_sentiments = set(reviews["sentiment"]) - {"positive", "negative"}
    if unexpected_sentiments:
        unexpected = ", ".join(sorted(unexpected_sentiments))
        raise ValueError(f"Unexpected sentiment value(s): {unexpected}")
    return reviews


def _words_in_review(review: str) -> set[str]:
    """Return unique lowercase words from one review."""
    review_without_html = HTML_TAG_PATTERN.sub(" ", review.lower())
    return set(WORD_PATTERN.findall(review_without_html))


def calculate_word_tilts(
    reviews: pd.DataFrame, min_reviews: int = MIN_REVIEWS
) -> pd.DataFrame:
    """Calculate positive-to-negative probability ratios for sufficiently common words."""
    required_columns = {"review", "sentiment"}
    missing_columns = required_columns - set(reviews.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Reviews are missing required column(s): {missing}")
    if min_reviews < 1:
        raise ValueError("min_reviews must be at least 1")

    positive_reviews = reviews[reviews["sentiment"] == "positive"]
    negative_reviews = reviews[reviews["sentiment"] == "negative"]
    if positive_reviews.empty or negative_reviews.empty:
        raise ValueError("The dataset must contain both positive and negative reviews")

    positive_word_counts = Counter(
        word
        for review in positive_reviews["review"]
        for word in _words_in_review(str(review))
    )
    negative_word_counts = Counter(
        word
        for review in negative_reviews["review"]
        for word in _words_in_review(str(review))
    )

    eligible_words = sorted(
        word
        for word in positive_word_counts.keys() & negative_word_counts.keys()
        if positive_word_counts[word] >= min_reviews
        and negative_word_counts[word] >= min_reviews
    )
    positive_review_total = len(positive_reviews)
    negative_review_total = len(negative_reviews)

    result = pd.DataFrame(
        {
            "word": eligible_words,
            "positive_review_count": [
                positive_word_counts[word] for word in eligible_words
            ],
            "negative_review_count": [
                negative_word_counts[word] for word in eligible_words
            ],
        }
    )
    result["positive_probability"] = (
        result["positive_review_count"] / positive_review_total
    )
    result["negative_probability"] = (
        result["negative_review_count"] / negative_review_total
    )
    result["word_tilt"] = (
        result["positive_probability"] / result["negative_probability"]
    )
    return result.sort_values("word_tilt", ascending=False, ignore_index=True)


def main() -> None:
    reviews = load_reviews()
    word_tilts = calculate_word_tilts(reviews).head(20)
    print(word_tilts.to_string(index=False))
    
    # This code prints out the necessary word and the word tilts that have in between (0.9,1.1) word tilts.
    word_tilts = calculate_word_tilts(reviews)
    for i in range(len(word_tilts)):
        if 0.9 <= word_tilts.iloc[i,5] <= 1.1:
            print(word_tilts.iloc[i, 0], word_tilts.iloc[i,5])


if __name__ == "__main__":
    main()

def mixed_words():
    """Return a list words whose word_tilt is within 0.9 and 1.1"""
    mixed_word_list = []
    reviews = load_reviews()
    word_tilts = calculate_word_tilts(reviews)
    for word_index in range(len(word_tilts)):
        if 0.9 <= word_tilts.iloc[word_index,5] <= 1.1:
            mixed_word_list.append(word_tilts.iloc[word_index,0])
    return mixed_word_list

# Testing mixed_words()
# print(mixed_words())


def remove_mixed_words(review):
    mixed_word_list = mixed_words()
    separated_review = review.lower().split()
    separated_review = [word for word in separated_review if word not in mixed_word_list]
    trimmed_review = " ".join(separated_review)
    #cleaning text after combining it to a string
    trimmed_review = re.sub(r'\s+([.,!?])', r'\1', trimmed_review)
    return trimmed_review
# Testing remove_mixed_words()
# review = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."
# print(remove_mixed_words(review))
