"""
Script which randomly samples one caption from each of the five captions given for each sample in AudioCaps test set.
Outputs as .json file which is used for generating our test set.
"""

import json

import pandas as pd


def generate_caption_csv(input_csv, output_path):
    # for every youtube_ID, randomly return one caption, save in df, spit out into separate file
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Create an empty list to store sampled rows
    sampled_rows = []

    # Get unique youtube_ids
    unique_youtube_ids = df["youtube_id"].unique()

    # Randomly sample one row for each unique youtube_id
    for youtube_id in unique_youtube_ids:
        rows = df[df["youtube_id"] == youtube_id]
        sampled_row = rows.sample(n=1)
        sampled_rows.append(sampled_row.iloc[0].to_dict())

    # Convert the list of sampled rows to a DataFrame
    sampled_df = pd.DataFrame(sampled_rows)

    # Convert sampled DataFrame to JSON format
    caption_json = sampled_df.to_json(orient="records")

    # Save JSON to file
    with open(output_path, "w") as f:
        json.dump(json.loads(caption_json), f, indent=4)


generate_caption_csv("../../data/audiocaps/dataset/test.csv", "test_captions.json")
