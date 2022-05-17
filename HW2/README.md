# README

To train the model, make sure all the shell scripts under the directory are executable, and then use the file `train_all.sh` with specified data directory

```bash
bash train_all.sh [data_dir]
```

The argument `data_dir` should include the following files `context.json`,  `train.json`, `valid.json`,  and `test.json`. The script first executes `preprocess_json.py` to generate data with appropriate format for later use, and store the output preprocessed data in the directory `data/` under current directory. Next it executes the script `multiple-choice/run_select.sh` and `qa/run_qa.sh`. Both scripts include proper parameters for training the model. As default, all the models and tokenizers are saved in `ckpt/select` and `ckpt/qa` , and cache for data are stored in `cache/`. Note that the scripts will download pre-trained model from huggingface if no cached files are found. The following models are used in this homework.

* [bert-base-chinese](https://huggingface.co/bert-base-chinese) (used in `multiple-choice/run_select.sh`)
* [hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) (used in `qa/run_qa.sh`)

One can change the choice of pretrained model by modifying both scripts.

After training, one can use `run.sh` with specified arguments to predict a given test file.

```bash
bash ./run.sh [context_file_path] [test_file_path] [output_prediction_path]
```
