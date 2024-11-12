import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.dataset = "hkr"
        self.model_path = os.path.join(
            "checkpoints",
            self.dataset + "_" + datetime.strftime(datetime.now(), "%Y%m%d%H%M"),
        )
        self.vocab = ""
        self.height = 32
        self.width = 128
        self.max_text_length = 0
        self.batch_size = 128
        self.learning_rate = 0.0005
        self.train_epochs = 1000
        self.train_workers = 16
