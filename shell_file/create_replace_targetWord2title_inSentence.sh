input_path="/work/masaki/data/wikilinks/ne/Person_Location/aggregated/wikilinks_more10_title.jsonl"

python3 wikilinks_preprocess.py --preprocessors create_replace_targetWord2title_inSentence --input $input_path 
wait