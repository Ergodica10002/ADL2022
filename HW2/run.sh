python3 preprocess_json.py --data_file ${2}
chmod u+x ./multiple-choice/run_pred.sh
chmod u+x ./qa/run_pred.sh
bash ./multiple-choice/run_pred.sh ${1} ${2}
bash ./qa/run_pred.sh ${1} ${3}
