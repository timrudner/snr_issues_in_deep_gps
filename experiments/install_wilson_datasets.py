"""Installs 'wilson' datasets to the bayesian_benchmarks package."""
import os
import shutil
import tarfile
import tempfile
import uuid
from argparse import ArgumentParser
from typing import List

from bayesian_benchmarks.data import DATA_PATH, get_regression_data


def main(file_path: str, datasets: List[str]):
    if file_path.endswith("tar.gz"):
        mode = "r:gz"
    elif file_path.endswith("tar"):
        mode = "r:"
    else:
        raise ValueError

    extract_path = os.path.join(tempfile.gettempdir(), f"uci-{uuid.uuid4()}")
    uci_path = os.path.join(extract_path, "uci")
    try:
        with tarfile.open(file_path, mode) as tar:
            assert len(tar.members) == 1
            tar.extractall(extract_path)

        for dataset in datasets:
            dataset_path = os.path.join(uci_path, dataset)
            if not os.path.isdir(dataset_path):
                raise ValueError(f"Unknown dataset {dataset} (available {tar.members}")

            dest_path = os.path.join(DATA_PATH, "uci", dataset)

            if os.path.isdir(dest_path):
                print(f"Skipping existing dataset {dataset}")
            else:
                print(f"Moving dataset {dataset} to {dest_path}")
                shutil.move(dataset_path, dest_path)

            assert get_regression_data(f"wilson_{dataset}") is not None
    finally:
        shutil.rmtree(extract_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "file",
        type=str,
        help="Download file from https://drive.google.com/open?id=0BxWe_IuTnMFcYXhxdUNwRHBKTlU",
    )
    parser.add_argument("--datasets", type=str, help="Comma delimited list of datasets to install")
    args = parser.parse_args()

    datasets = [dataset.strip() for dataset in args.datasets.split(",")]

    main(args.file, datasets)
