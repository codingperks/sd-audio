from downloader import download_audiocaps

if __name__ == "__main__":
    print("running")
    download_audiocaps(
        "dataset/train.csv",
        "dataset_full/train_download_success.csv",
        "dataset_full/train_download_fail.csv",
        "dataset_full/train",
        1,
    )
