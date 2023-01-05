input_path="/work/masaki/data/wikilinks/ne/Person_Location/aggregated/wikilinks.jsonl"

sentence_count_lower=10
echo sentence_count_lower=$sentence_count_lower
python3 wikilinks_preprocess.py --preprocessors extract_ne_df --input $input_path  --sentence_count_lower $sentence_count_lower
wait