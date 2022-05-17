from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoConfig
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

import torch
import numpy as np
import json
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model.",
        default="out/pytorch_model.bin"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the config file.",
        default="out/"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to the tokenizer.",
        default = "./out/"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Path to the test data.",
        default="data/sample_test.jsonl"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to save the cache.",
        default = "./cache/"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for predicting",
        default=16,
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        help="gradient accumulation step",
        default=1,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the output.",
        default="./submission.jsonl",
    )
    parser.add_argument(
        "--max_text_length",
        type=int,
        help="max length to truncate maintext",
        default=256
    )
    parser.add_argument(
        "--max_title_length",
        type=int,
        help="max length to generate title",
        default=64
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        help="beam number for decoder",
        default=1
    )
    args = parser.parse_args()
    return args


def main(args):

    # Load data
    data_files = {}
    data_files["test"] = args.test_file
    dataset = load_dataset(
        'json',
        data_files = data_files,
        cache_dir=args.cache_dir
    )
    
    # Preprocess (tokenize + truncation)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        cache_dir = args.cache_dir
    )

    prefix = "summarize: "
    textname = 'maintext'
    def preprocess(examples):
        texts = [prefix + text for text in examples[textname]]
        model_inputs = tokenizer(
            texts,
            max_length = args.max_text_length,
            padding="max_length",
            truncation = True)
        
        # model_inputs["decoder_input_ids"] = model_inputs["input_ids"]
        # print(len(model_inputs["input_ids"][0]))
        # print(tokenizer("<pad>").input_ids)
        decoder_input_ids = tokenizer(
            "<pad>",
            max_length = args.max_text_length,
            padding="max_length",
            truncation = True
        )
        model_inputs["decoder_input_ids"] = [decoder_input_ids["input_ids"]] * len(model_inputs["input_ids"])
        
        return model_inputs

    column_names = dataset["test"].column_names
    column_names.remove("id")
    tokenized_data = dataset.map(
        preprocess,
        batched = True,
        remove_columns=column_names,
        desc="Running tokenizer"
    )

    # Load model and predict
    config = AutoConfig.from_pretrained(
        args.config_path,
        cache_dir = args.cache_dir
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path,
        config=config,
        cache_dir=args.cache_dir
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model)

    training_args = Seq2SeqTrainingArguments(
        output_dir = './pred_dir/',
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        gradient_accumulation_steps = args.accumulation_steps,
        predict_with_generate = True
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        tokenizer = tokenizer,
        data_collator = data_collator
    )

    results = trainer.predict(
        tokenized_data['test'],
        max_length = args.max_title_length,
        num_beams=args.num_beams)

    trainer.save_metrics("predict", results.metrics)
    
    predictions = tokenizer.batch_decode(
        results.predictions,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    predictions = [pred.strip() for pred in predictions]
    ids = tokenized_data['test']['id']
    with open(args.output_file, 'w') as jsonFile:
        jsonFile.truncate()
        for i in range(len(predictions)):
            jsonDir = {'id': ids[i], 'title': predictions[i]}
            jsonStr = json.dumps(jsonDir, ensure_ascii = False)
            jsonFile.write(jsonStr + '\n')

if __name__ == "__main__":
    args = parse_args()
    main(args)
