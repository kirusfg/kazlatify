import codecs
import os

import cv2
import typing
import numpy as np
from tqdm import tqdm

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.annotations.images import CVImage
from mltu.configs import BaseModelConfigs
from word_beam_search import WordBeamSearch


from .data.data import get_data_provider, get_data_splits
from .data.constants import VOCABULARY, LETTERS
from .model import train_model
from .transformers import ImageThresholding


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], lang: str, use_wbs: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

        if lang == "ru":
            self.corpus = codecs.open(os.path.join("corpora", "russian.txt")).read()
        elif lang == "kz":
            self.corpus = codecs.open(os.path.join("corpora", "kazakh.txt")).read()
        else:
            # Open both corpora and concatenate them
            self.corpus = (
                codecs.open(os.path.join("corpora", "kazakh.txt")).read() + codecs.open(os.path.join("corpora", "russian.txt")).read()
            )

        self.wbs = WordBeamSearch(
            25,
            "NGrams",
            0.01,
            self.corpus.encode("utf8"),
            "".join(VOCABULARY).encode("utf8"),
            "".join(LETTERS).encode("utf8"),
        )

        self.use_wbs = use_wbs

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        image_pred = np.expand_dims(image_pred, axis=-1).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        text_wbs = text
        if self.use_wbs:
            preds_to_wbs = np.transpose(preds, (1, 0, 2))
            text_wbs = self.wbs_decode(preds_to_wbs)[0]

        return text, text_wbs

    def wbs_decode(self, preds: np.array) -> list[str]:
        label_str = self.wbs.compute(preds)
        char_str = []
        for curr_label_str in label_str:
            s = ""
            for label in curr_label_str:
                s += self.char_list[label]
            char_str.append(s)
        # print(label_str[0], char_str[0])

        return char_str


def main(models: list[str], datasets: list[str] = None, images: list[str] = None, wbs_langs: list[str] = []):
    """
    Test OCR models on datasets or individual images

    Args:
        models: List of model names to test ('HKR', 'KOHTD', 'HKR+KOHTD')
        datasets: List of dataset names to test on ('hkr', 'kohtd', 'both')
        images: List of paths to individual images to test
        use_wbs: Whether to use Word Beam Search for decoding
    """
    models_checkpoints = {
        "HKR": "hkr_202411051410",
        "KOHTD": "kohtd_202411051413",
        "HKR+KOHTD": "both_202411051413",
    }

    # Validate models
    for model in models:
        if model not in models_checkpoints:
            raise ValueError(f"Invalid model name: {model}. Choose from {list(models_checkpoints.keys())}")

    if datasets:
        splits = get_data_splits("/raid/kirill_kirillov/kazlatify/data")
        # Validate datasets
        for dataset in datasets:
            if dataset not in splits:
                raise ValueError(f"Invalid dataset name: {dataset}. Choose from {list(splits.keys())}")

    for model_name in models:
        print(f"\nTesting model: {model_name}")
        model_checkpoint = models_checkpoints[model_name]
        configs = BaseModelConfigs.load(f"checkpoints/{model_checkpoint}/configs.yaml")

        if wbs_langs:
            for lang in wbs_langs:
                model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab, lang=lang, use_wbs=True)

                if datasets:
                    # Test on datasets
                    for dataset in datasets:
                        print(f"\nTesting on dataset: {dataset}")
                        df = splits[dataset]["test"].select(["img_path", "label"]).rows()

                        accum_cer, accum_wer = [], []
                        accum_cer_wbs, accum_wer_wbs = [], []

                        for image_path, label in tqdm(df, desc="Processing images"):
                            image = CVImage(image_path)
                            image, _ = ImageThresholding()(image, None)

                            prediction_text, prediction_text_wbs = model.predict(image)

                            cer = get_cer(prediction_text, label)
                            wer = get_wer(prediction_text, label)
                            accum_cer.append(cer)
                            accum_wer.append(wer)

                            cer_wbs = get_cer(prediction_text_wbs, label)
                            wer_wbs = get_wer(prediction_text_wbs, label)
                            accum_cer_wbs.append(cer_wbs)
                            accum_wer_wbs.append(wer_wbs)

                        print(f"Average CER on {dataset.upper()}: {np.average(accum_cer)}")
                        print(f"Average WER on {dataset.upper()}: {np.average(accum_wer)}")
                        print(f"Average CER on {dataset.upper()} with WBS: {np.average(accum_cer_wbs)}")
                        print(f"Average WER on {dataset.upper()} with WBS: {np.average(accum_wer_wbs)}")

                if images:
                    # Test on individual images
                    print("\nTesting on individual images:")
                    for image_path in images:
                        if not os.path.exists(image_path):
                            print(f"Warning: Image not found - {image_path}")
                            continue

                        image = CVImage(image_path)
                        image, _ = ImageThresholding()(image, None)

                        prediction_text, prediction_text_wbs = model.predict(image)

                        print(f"\nImage: {image_path}")
                        print(f"Prediction: {prediction_text}")
                        print(f"Prediction (WBS): {prediction_text_wbs}")

        if not wbs_langs:
            model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab, lang="", use_wbs=False)

            if datasets:
                # Test on datasets
                for dataset in datasets:
                    print(f"\nTesting on dataset: {dataset}")
                    df = splits[dataset]["test"].select(["img_path", "label"]).rows()

                    accum_cer, accum_wer = [], []
                    accum_cer_wbs, accum_wer_wbs = [], []

                    for image_path, label in tqdm(df, desc="Processing images"):
                        image = CVImage(image_path)
                        image, _ = ImageThresholding()(image, None)

                        prediction_text, prediction_text_wbs = model.predict(image)

                        cer = get_cer(prediction_text, label)
                        wer = get_wer(prediction_text, label)
                        accum_cer.append(cer)
                        accum_wer.append(wer)

                    print(f"Average CER on {dataset.upper()}: {np.average(accum_cer)}")
                    print(f"Average WER on {dataset.upper()}: {np.average(accum_wer)}")

                if images:
                    # Test on individual images
                    print("\nTesting on individual images:")
                    for image_path in images:
                        if not os.path.exists(image_path):
                            print(f"Warning: Image not found - {image_path}")
                            continue

                        image = CVImage(image_path)
                        image, _ = ImageThresholding()(image, None)

                        prediction_text, prediction_text_wbs = model.predict(image)

                        print(f"\nImage: {image_path}")
                        print(f"Prediction: {prediction_text}")
                        print(f"Prediction (WBS): {prediction_text_wbs}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test OCR models on datasets or individual images")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of models to test (HKR,KOHTD,HKR+KOHTD)")
    parser.add_argument("--datasets", type=str, help="Comma-separated list of datasets to test on (hkr,kohtd,both)")
    parser.add_argument("--images", type=str, help="Comma-separated list of image paths to test")
    parser.add_argument("--wbs", type=str, help="Comma-separated list of languages to use WBS for (ru,kz)")

    args = parser.parse_args()

    if not args.datasets and not args.images:
        parser.error("Either --datasets or --images must be specified")

    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    images = [i.strip() for i in args.images.split(",")] if args.images else None
    wbs_langs = [w.strip() for w in args.wbs.split(",")] if args.wbs else None

    main(models, datasets, images, wbs_langs)
