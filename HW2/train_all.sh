python3 preprocess_json.py --data_dir ${1}
bash ./multiple-choice/run_select.sh
bash ./qa/run_qa.sh