# KazLatify

OCR models for Kazakh handwritten text recognition

## Datasets

Datasets used for training are available at

- HKR: https://github.com/abdoelsayed2016/HKR_Dataset
- KOHTD: https://github.com/abdoelsayed2016/KOHTD

and must be downloaded, extracted, and placed in the `data` folder.

## Training

To train the models, run `train_hkr.sh` to train on HKR, `train_kohtd.sh` to train on KOHTD, and `train_all.sh` to train on both.

### Pre-trained models

The pre-trained models are available in the `checkpoints` folder. The hyper-parameters used for training are also available in the respective `configs.yaml` files.

Below are the training and validation CER graphs on images for the 3 pre-trained models.

Model trained on HKR:
![HKR training](https://github.com/kirusfg/kazlatify/blob/master/static/hkr_training.png)

Model trained on KOHTD:
![KOHTD training](https://github.com/kirusfg/kazlatify/blob/master/static/kohtd_training.png)

Model trained on both:
![Both training](https://github.com/kirusfg/kazlatify/blob/master/static/both_training.png)

## Testing

### OCR

To test the OCR models, there is a script `test.py` that can be used to test on the test subsets of the datasets or individual images.

For example, this

```bash
python -m src.test --models KOHTD --datasets kohtd
```

will test the model trained on KOHTD on the KOHTD test subset.

Similarly, this

```bash
python -m src.test --models HKR+KOHTD --images ./images/hkr/0001.jpg --wbs kz,ru
```

will test the model trained on both datasets on the image `0001.jpg` and use the WBS decoder with Kazakh and Russian corpora. To use WBS (Word Beam Search) decoding, you need to install the [CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch) package and put space-separated lists of words in the `corpora` folder.

### End-to-end

To test end-to-end, there are two scripts `page.py` and `page_paddle.py` using Tesseract and PaddleOCR for page segmentation, respectively. PaddleOCR generally performs better.

Example usage:

```bash
python -m src.page --model HKR --images ./images/hkr/ancient_scripture.jpg --save_segments
```

will segment the image `ancient_scripture.jpg` using the model trained on HKR and save the segmented regions in the `segments_ancient_scripture` folder.