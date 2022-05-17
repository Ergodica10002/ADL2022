# README.md

## Intent Classification

To train the model, execute the command

```bash
python train_intent.py [-h] [--data_dir DATA_DIR] [--cache_dir CACHE_DIR]
                       [--ckpt_dir CKPT_DIR] [--max_len MAX_LEN]
                       [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS]
                       [--dropout DROPOUT] [--bidirectional BIDIRECTIONAL]
                       [--lr LR] [--batch_size BATCH_SIZE] [--device DEVICE]
                       [--num_epoch NUM_EPOCH] [--optimizer OPTIMIZER]
                       [--add_eval]
```

The description of the parameters are given:

* -h, --help: Show the above help message.
* --data_dir DATA_DIR: Desginate the directory used to store training and validation data. The given `DATA_DIR` should contain two files, `train.json` and `eval.json`, respectively. The default value is `./data/intent/`.
* --cache_dir CACHE_DIR: Designate the directory used to store preprocessed data, e.g. embeddings. The default value is `./cache/intent`, which should have been given.
* --ckpt_dir CKPT_DIR: Designate the directory to save the model, which will be named `model.pt`.
* --max_len MAX_LEN : This parameter is not used in this homework.
* --hidden_size HIDDEN_SIZE: Set the hidden layer size of the GRU model. The default value is 512.
* --num_layer NUM_LAYERS: Set the number of layers of the GRU model. The default value is 2.
* --dropout DROPOUT: Set the dropout value of the GRU model. The default value is 0.1.
* --bidirectional BIDIRECTIONAL: Set whether to use bidirectional model. The default value is True.
* --lr LR: Set the learning rate of the optimizer. The default value is 0.001.
* --batch_size BATCH_SIZE: Set the size of data samples to use in one batch. The default value is 128.
* --device DEVICE: Designate the device to train the model. Accepted values include `cpu`, `cuda`, `cuda:0`, `cuda:1`.The default value is `cpu`.
* --num_epoch NUM_EPOCH: Set the number of epochs in training. The default value is 100.
* --optimizer OPTIMIZER: Choose which optimizer to use in training. Accpeted values include `SGD` and `Adam`. The default value is `SGD`.
* --add_eval: Set this flag to use validation data set as part of training data set.

The given best model(downloaded by `download.sh`) is can be reproduced by

```bash
python train_intent.py --optimizer Adam
```

where the `DATA_DIR` should include files `train.json` and `eval.json`, and the model is saved at the default path `./ckpt/intent/model.pt`.



## Slot Tagging

To train the model, execute the command

```bash
python train_slot.py [-h] [--data_dir DATA_DIR] [--cache_dir CACHE_DIR]
                     [--ckpt_dir CKPT_DIR] [--max_len MAX_LEN]
                     [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS]
                     [--dropout DROPOUT] [--bidirectional BIDIRECTIONAL]
                     [--lr LR] [--batch_size BATCH_SIZE] [--device DEVICE]
                     [--num_epoch NUM_EPOCH] [--optimizer OPTIMIZER]
                     [--add_eval]
```

The description of the parameters are given:

* -h, --help: Show the above help message.
* --data_dir DATA_DIR: Desginate the directory used to store training and validation data. The given `DATA_DIR` should contain two files, `train.json` and `eval.json`, respectively. The default value is `./data/slot/`.
* --cache_dir CACHE_DIR: Designate the directory used to store preprocessed data, e.g. embeddings. The default value is `./cache/slot`, which should have been given.
* --ckpt_dir CKPT_DIR: Designate the directory to save the model, which will be named `model.pt`
* --max_len MAX_LEN : This parameter is not used in this homework.
* --hidden_size HIDDEN_SIZE: Set the hidden layer size of the LSTM model. The default value is 512.
* --num_layer NUM_LAYERS: Set the number of layers of the LSTM model. The default value is 2.
* --dropout DROPOUT: Set the dropout value of the LSTM model. The default value is 0.1.
* --bidirectional BIDIRECTIONAL: Set whether to use bidirectional model. The default value is True.
* --lr LR: Set the learning rate of the optimizer. The default value is 0.001.
* --batch_size BATCH_SIZE: Set the size of data samples to use in one batch. The default value is 128.
* --device DEVICE: Designate the device to train the model. Accepted values include `cpu`, `cuda`, `cuda:0`, `cuda:1`.The default value is `cpu`.
* --num_epoch NUM_EPOCH: Set the number of epochs in training. The default value is 100.
* --optimizer OPTIMIZER: Choose which optimizer to use in training. Accpeted values include `SGD` and `Adam`. The default value is `SGD`.
* --add_eval: Set this flag to use validation data set as part of training data set.

The given best model(downloaded by `download.sh`) is can be reproduced by

```bash
python train_slot.py --num_layer 3 --optimizer Adam --add_eval True 
```

where the `DATA_DIR` should include files `train.json` and `eval.json`, and the model is saved at the default path `./ckpt/slot/model.pt`.