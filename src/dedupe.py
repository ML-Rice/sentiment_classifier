DEDUP_COLUMNS = [
    "Text",
    "Summary",
    "Score",
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
    "UserId",
]


def remove_duplicates(df):
    before = len(df)

    df = df.sort_values("Time", kind="stable")
    df = df.drop_duplicates(subset=DEDUP_COLUMNS, keep="last")

    df = df.sort_index()

    removed = before - len(df)
    print(f"Duplicates removed: {removed} ({removed / before * 100:.1f}%)")
    print(f"Rows: {before} -> {len(df)}")
    return df
