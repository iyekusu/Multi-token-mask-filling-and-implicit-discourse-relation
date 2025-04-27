## Multi-token mask-filling and implicit discourse relation

This package provides code for the paper *Multi-token mask-filling and implicit discourse relation*.

Our code is adapted from the code related this paper:[Simple and Effective Multi-Token Completion from Masked Language Models](https://aclanthology.org/2023.findings-eacl.179/)
Bibtex entry:
```
@inproceedings{DBLP:conf/eacl/KalinskyKLG23,
  author       = {Oren Kalinsky and
                  Guy Kushilevitz and
                  Alexander Libov and
                  Yoav Goldberg},
  title        = {Simple and Effective Multi-Token Completion from Masked Language Models},
  booktitle    = {Findings of the Association for Computational Linguistics: {EACL}
                  2023, Dubrovnik, Croatia, May 2-6, 2023},
  pages        = {2311--2324},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://aclanthology.org/2023.findings-eacl.179},
  timestamp    = {Mon, 08 May 2023 14:38:37 +0200},
  biburl       = {https://dblp.org/rec/conf/eacl/KalinskyKLG23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Steps to run the code

### Configuration
* pip install -r requirements.txt
* pip install transformers==4.5.1

### Data preprocessing
Training/dev dataset extracted from Wikipedia English dataset (20220301.en)
Test dataset: eg: test_data/test_preposed.csv

To create the dataset you will first need to parse our released data (training/dev/test) by running mtc_model_2args. For example, to create the experiment 1 dataset run: `mtc_model_2args.py --dataset_name dataset_name`. This will create the preprocessed dataset under `data/input_data_bert-base-cased_dataset_name/` using bert-base-cased by default.

### Training
* __EMAT decoder__ - run `matrix_plugin.py --input_path data/input_data_bert-base-cased_dataset_name/`

### Testing
* __EMAT decoder__ - run `matrix_plugin.py --input_path data/input_data_bert-base-cased_dataset_name/  --ckpt <CHECKPOINT_PATH> --test`
