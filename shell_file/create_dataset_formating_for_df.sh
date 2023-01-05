input_path="/work/masaki/data/wikilinks/dataset/wikilinks_with_figer_types.jsonl"


python3 wikilinks_preprocess.py --preprocessors create_dataset_formating_for_df --input $input_path 
wait