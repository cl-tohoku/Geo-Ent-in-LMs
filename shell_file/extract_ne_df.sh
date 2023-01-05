input_path="/work/masaki/data/wikilinks/ne/512token/aggregated/preprocessed_512token_wikilinks_more_10_vocab.jsonl"


python3 wikilinks_preprocess.py --preprocessors create_ne_df --input $input_path --is_vocab
wait