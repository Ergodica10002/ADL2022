python3 qa/run_pred.py \
  --model_name_or_path ckpt/qa/pytorch_model.bin \
  --config_name ckpt/qa/config.json \
  --tokenizer_path ckpt/qa/ \
  --test_file data/pred_for_qa.json \
  --context_file ${1} \
  --do_predict \
  --output_dir out/qa \
  --output_file ${2} \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output