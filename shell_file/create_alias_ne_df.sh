input_path="/work/masaki/data/wikilinks/ne/Person_Location/aggregated/wikilinks_more10.jsonl"
output_path="/work/masaki/data/wikilinks/ne/Person_Location/aggregated/ne_alias/wikilinks_more10.jsonl"
alias_count_lower=2
alias_count_upper=9
sentence_count_upper=16056

python3 wikilinks_preprocess.py --preprocessors create_alias_ne_df --input $input_path --output $output_path --alias_count_lower $alias_count_lower --alias_count_upper $alias_count_upper --sentence_count_upper $sentence_count_upper
wait