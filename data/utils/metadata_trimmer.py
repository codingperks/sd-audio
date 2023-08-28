import pandas as pd


# Remove 'a spectrogram of' from all metadata
def trim_metadata_captions(file_path):
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Remove the substring from the 'text' column
    df["text"] = df["text"].str.replace("a spectrogram of", "", regex=False)
    df["text"] = df["text"].str.replace('"', "", regex=False)

    # Save the modified DataFrame back to the CSV
    df.to_csv(file_path, index=False)
