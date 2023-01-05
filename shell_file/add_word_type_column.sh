input_path="/work/masaki/data/wikilinks/ne/Person_Location/aggregated/wikilinks_more10.jsonl"
word_type=ne

python3 wikilinks_preprocess.py --preprocessors add_word_type_column --input $input_path --word_type $word_type
wait


