input_path="/work/masaki/data/wikilinks/ne/Person_Location/aggregated/wikilinks_more10.jsonl"


python3 wikilinks_preprocess.py --preprocessors extract_ne_df --input $input_path  --is_add_title
wait