import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="ntu-adl-hw2-spring-2021/"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./data/",
    )
    parser.add_argument(
    	"--data_file",
    	type=Path,
    	help="Data to be prerprocessed if data_dir not specified",
    	default=None
    )
    args = parser.parse_args()
    return args

def main(args):
	if args.data_file == None:
		split_list = ["train", "valid", "test"]
		context_path = args.data_dir / 'context.json'
		context = json.loads(context_path.read_text())
		context_str = json.dumps(context, indent = 2, ensure_ascii = False)
		with open(args.output_dir / 'context.json', "w") as jsonFile:
			jsonFile.truncate()
			jsonFile.write(context_str)
	else:
		split_list = [args.data_file]

	for split in split_list:
		if args.data_file == None:
			dataset_path = args.data_dir / f'{split}.json'
		else:
			dataset_path = split
		data = json.loads(dataset_path.read_text())
		data_size = len(data)
		logging.info(f"Loading {split}: data size = {data_size}")
		keys = data[0].keys()
		select_list, qa_list = [], []
		for i in range(data_size):
			select_data, qa_data = {}, {}
			select_data['id'], qa_data['id'] = str(data[i]['id']), str(data[i]['id'])
			select_data['question'], qa_data['question'] = str(data[i]['question']), str(data[i]['question'])
			for j in range(4):
				select_data[f'paragraph{j}'] = int(data[i]['paragraphs'][j])
			if 'relevant' in keys:
				select_data['relevant'], qa_data['relevant'] = int(data[i]['relevant']), int(data[i]['relevant'])
				select_data['label'] = int(data[i]['paragraphs'].index(data[i]['relevant']))
			if 'answer' in keys:
				qa_data['answer'] = {}
				qa_data['answer']['text'] = [data[i]['answer']['text']]
				qa_data['answer']['start'] = [data[i]['answer']['start']]
			select_list.append(select_data)
			qa_list.append(qa_data)
		select_dir, qa_dir = {}, {}
		select_dir["data"], qa_dir["data"] = select_list, qa_list
		select_jsonstr = json.dumps(select_dir, indent = 2, ensure_ascii = False)
		qa_jsonstr = json.dumps(qa_dir, indent = 2, ensure_ascii = False)

		if args.data_file == None:
			select_filepath = args.output_dir / f'select_{split}.json'
			qa_filepath = args.output_dir / f'qa_{split}.json'
		else:
			select_filepath = args.output_dir / f'select_test.json'
			qa_filepath = args.output_dir / f'qa_test.json'

		with open(select_filepath, "w") as jsonFile:
			jsonFile.truncate()
			jsonFile.write(select_jsonstr)
		with open(qa_filepath, "w") as jsonFile:
			jsonFile.truncate()
			jsonFile.write(qa_jsonstr)

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)