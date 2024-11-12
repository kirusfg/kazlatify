import argparse
from datetime import datetime

import tensorflow as tf

try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass

from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CWERMetric

from .data.data import get_data_provider, get_data_splits
from .model import train_model


def seed_everything(seed=42):
    import os
    import random

    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(dataset="hkr", train_epochs=1000, train_workers=1):
    splits = get_data_splits("/raid/kirill_kirillov/kazlatify/data")

    train_data_provider, configs = get_data_provider(dataset, splits, "train")
    val_data_provider, _ = get_data_provider(dataset, splits, "val")

    configs.dataset = dataset
    configs.model_path = f"checkpoints/{dataset}_{datetime.strftime(datetime.now(), '%Y%m%d%H%M')}"

    configs.train_epochs = train_epochs
    configs.train_workers = train_workers
    configs.save()

    model = train_model(
        input_dim=(configs.height, configs.width, 1),
        output_dim=len(configs.vocab),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
        loss=CTCloss(),
        metrics=[CWERMetric(padding_token=len(configs.vocab))],
    )
    model.summary(line_length=110)
    return

    earlystopper = EarlyStopping(monitor="val_CER", mode="min", patience=50, verbose=1)
    checkpoint = ModelCheckpoint(
        f"{configs.model_path}/model.h5",
        monitor="val_CER",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    train_logger = TrainLogger(configs.model_path)
    tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_CER",
        factor=0.9,
        min_delta=1e-10,
        patience=8,
        verbose=1,
        mode="auto",
    )
    model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

    model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=configs.train_epochs,
        callbacks=[
            earlystopper,
            checkpoint,
            train_logger,
            lr_scheduler,
            tb_callback,
            model2onnx,
        ],
        workers=configs.train_workers,
    )


if __name__ == "__main__":
    seed_everything()

    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("--dataset", type=str, default="hkr")
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--train_workers", type=int, default=8)

    args = parser.parse_args()

    train(**vars(args))
