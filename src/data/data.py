import json
import os
import pickle
from typing import Literal, List

import numpy as np
import polars as pl
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import (
    RandomBrightness,
    RandomRotate,
    RandomErodeDilate,
    RandomSharpen,
)
from mltu.annotations.images import CVImage

from mltu.tensorflow.dataProvider import DataProvider

from tqdm import tqdm

from .constants import VOCABULARY
from ..config import ModelConfigs
from ..transformers import ImageThresholding


def get_data_splits(
    root_data_dir: str,
    split: List[float] = [0.8, 0.1, 0.1],
    force_recompute: bool = False,
) -> dict:
    """
    Create consistent data splits for both datasets that can be reused
    Args:
        root_data_dir (str): The root data directory
        split (List[float]): Train/val/test split ratios
        force_recompute (bool): If True, recompute splits even if cached version exists
    Returns:
        dict: Dictionary containing splits for each dataset
    """
    # Create splits directory if it doesn't exist
    splits_dir = os.path.join(root_data_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    splits_file = os.path.join(splits_dir, f"splits_{split[0]}_{split[1]}_{split[2]}.pkl")

    # Try to load existing splits
    if not force_recompute and os.path.exists(splits_file):
        print(f"Loading existing splits from {splits_file}")
        with open(splits_file, "rb") as f:
            return pickle.load(f)

    print("Computing new splits...")
    splits = {}

    def load_dataset(data_dir: str) -> pl.DataFrame:
        """Helper function to load data from a directory"""
        data_keys = ["id", "img_path", "label"]
        data_dict = {k: [] for k in data_keys}

        # i = 0
        for f in tqdm(os.listdir(data_dir), desc="Loading dataset"):
            file_path = os.path.join(data_dir, f)
            if os.path.isfile(file_path):
                id, ext = os.path.splitext(f)
                if ext != ".jpg":
                    continue
                if os.path.exists(os.path.join(data_dir, id + ".json")):
                    with open(os.path.join(data_dir, id + ".json"), "r") as file:
                        label_info = json.load(file)
                        label = label_info["description"]
                        # If label has any symbols other than from VOCABULARY - skip
                        if not all(c in VOCABULARY for c in label):
                            continue
                        data_dict["id"].append(id)
                        data_dict["img_path"].append(file_path)
                        data_dict["label"].append(label_info["description"])
                        # i += 1
                        # if i == 1000:
                        #     break

        return pl.DataFrame(data_dict)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create splits for HKR dataset
    print("Processing HKR dataset...")
    hkr_dir = os.path.join(root_data_dir, "hkr", "words")
    hkr_data = load_dataset(hkr_dir)
    n_hkr = len(hkr_data)
    indices = np.random.permutation(n_hkr)
    train_end = int(n_hkr * split[0])
    val_end = int(n_hkr * (split[0] + split[1]))
    splits["hkr"] = {
        "train": hkr_data[indices[:train_end]],
        "val": hkr_data[indices[train_end:val_end]],
        "test": hkr_data[indices[val_end:]],
    }

    # Create splits for KOHTD dataset
    print("Processing KOHTD dataset...")
    kohtd_dir = os.path.join(root_data_dir, "kohtd")
    kohtd_data = load_dataset(kohtd_dir)
    n_kohtd = len(kohtd_data)
    indices = np.random.permutation(n_kohtd)
    train_end = int(n_kohtd * split[0])
    val_end = int(n_kohtd * (split[0] + split[1]))
    splits["kohtd"] = {
        "train": kohtd_data[indices[:train_end]],
        "val": kohtd_data[indices[train_end:val_end]],
        "test": kohtd_data[indices[val_end:]],
    }

    # Create combined dataset splits
    print("Creating combined splits...")
    splits["both"] = {
        "train": pl.concat([splits["hkr"]["train"], splits["kohtd"]["train"]]),
        "val": pl.concat([splits["hkr"]["val"], splits["kohtd"]["val"]]),
        "test": pl.concat([splits["hkr"]["test"], splits["kohtd"]["test"]]),
    }

    # Save splits to disk
    print(f"Saving splits to {splits_file}")
    with open(splits_file, "wb") as f:
        pickle.dump(splits, f)

    # Print split sizes for verification
    for dataset in splits:
        print(f"\n{dataset.upper()} split sizes:")
        for subset in splits[dataset]:
            print(f"{subset}: {len(splits[dataset][subset])} samples")

    return splits


def get_data_provider(
    dataset: Literal["hkr", "kohtd", "both"],
    splits: dict,
    subset: Literal["train", "val", "test"],
) -> (DataProvider, ModelConfigs):
    """
    Get data provider for a specific dataset and subset
    Args:
        dataset (Literal["hkr", "kohtd", "both"]): The dataset to use
        splits (dict): Dictionary containing the pre-computed splits
        subset (Literal["train", "val", "test"]): Which subset to use
    Returns:
        DataProvider: The data provider for the specified subset
        ModelConfigs: The model configurations
    """
    data = splits[dataset][subset]

    # Get dataset statistics (use all data to ensure consistent vocab and max length)
    all_data = pl.concat([splits[dataset]["train"], splits[dataset]["val"], splits[dataset]["test"]])

    dataset_array = data.select(["img_path", "label"]).to_numpy()
    vocab, max_text_len = (
        VOCABULARY,
        all_data["label"].str.len_chars().max(),
    )

    # Configure model
    configs = ModelConfigs()
    configs.vocab = "".join(vocab)
    configs.max_text_length = max_text_len

    # Create data provider with transformations
    data_provider = DataProvider(
        dataset=dataset_array,
        skip_validation=True,
        batch_size=configs.batch_size,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(
                configs.width,
                configs.height,
                keep_aspect_ratio=False,
                # keep_aspect_ratio=True,
                # padding_color=(255, 255, 255),
            ),
            LabelIndexer(configs.vocab),
            LabelPadding(
                max_word_length=configs.max_text_length,
                padding_value=len(configs.vocab),
            ),
            ImageThresholding(),
        ],
    )

    # Add augmentations only for training data
    if subset == "train":
        data_provider.augmentors = [
            RandomBrightness(),
            RandomErodeDilate(),
            RandomSharpen(),
            RandomRotate(angle=10),
        ]

    return data_provider, configs
