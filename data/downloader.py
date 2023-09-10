import csv
import os
import subprocess

import pandas as pd


def download_audiocaps(index_csv, success_csv, failures_csv, output_dir, limit):
    """
    Function which takes in an audiocaps index csv for train/test/val and attempts to download AC/AS clips from YouTube.
    Generates a success file and a failure file denoting (un)successfully scraped audio clips.

    Args:
        index_csv: audiocaps dataset index_csv (train/test/val)
        success_csv: location to output csv detailing successfully scraped audio
        failures_csv: location to output csv detailing unsuccessfully scraped audio
        output_dir: location to output scraped .wav audio files
        limit (integer between 0 and 1): percentage of files to download (useful for downloading subsets of data)
    """
    os.makedirs(output_dir, exist_ok=True)

    processed_ytids = set()
    n_downloaded = 0

    assert limit <= 1 & limit > 0, "limit should be between 0 and 1"

    total_rows = len(pd.read_csv(index_csv))
    target = total_rows * limit

    # Set up csv to output successfully downloaded samples to
    if os.path.exists(success_csv):
        with open(success_csv, "r") as successfile:
            reader = csv.reader(successfile)
            next(reader, None)
            for row in reader:
                processed_ytids.add(row[1])

    # Set up csv to output unsuccessfully downloaded samples to
    if os.path.exists(failures_csv):
        with open(failures_csv, "r") as failurefile:
            reader = csv.reader(failurefile)
            next(reader, None)
            for row in reader:
                processed_ytids.add(row[1])

    # Open these two files to write to during download
    with open(index_csv, "r") as csvfile, open(
        failures_csv, "a", newline=""
    ) as failurefile, open(success_csv, "a", newline="") as successfile:
        reader = csv.reader(csvfile)
        failure_writer = csv.writer(failurefile)
        success_writer = csv.writer(successfile)

        if not processed_ytids:
            failure_writer.writerow(["audiocap_id", "youtube_id", "error"])
            success_writer.writerow(["audiocap_id", "youtube_id", "caption"])

        next(reader, None)

        for row in reader:
            audiocap_id = row[0]
            ytid = row[1]

            print(f"Processing {ytid}")

            if ytid in processed_ytids:
                print(f"Skipping already processed video {ytid}")
                continue

            start_seconds = row[2]
            duration = "00:00:10"
            caption = row[3]

            video_url = f"https://www.youtube.com/watch?v={ytid}"
            output_audio_file = os.path.join(output_dir, f"{ytid}.wav")

            if os.path.exists(output_audio_file):
                print(f"Skipping already downloaded file {output_audio_file}")
                continue

            print(f"using url {video_url}")

            get_url = f"youtube-dl -4 -g -f bestaudio {video_url}"
            print(f"audio_url {get_url}")

            audio_url = subprocess.getoutput(get_url).strip()

            if not audio_url:
                failure_writer.writerow([audiocap_id, ytid, "URL retrieval failure"])
                print(f"Failed to retrieve URL for video {ytid}")
                continue

            download_and_convert = f'ffmpeg -ss {start_seconds} -i "{audio_url}" -t {duration} -acodec pcm_s16le -ar 44100 {output_audio_file}'

            download_result = os.system(download_and_convert)

            if download_result != 0:
                failure_writer.writerow(
                    [audiocap_id, ytid, "Download or conversion failure"]
                )
                print(f"Failed to process video {ytid}")
                continue

            print(f"Successfully processed video {ytid}")
            success_writer.writerow([audiocap_id, ytid, caption])

            n_downloaded += 1
            if n_downloaded >= target:
                break


download_audiocaps(
    "dataset/train.csv",
    "dataset_full/train_download_success.csv",
    "dataset_full/train_download_fail.csv",
    "dataset_full/train",
    1,
)
