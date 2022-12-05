input_path="/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000.jsonl"
split=4
python3 wikilinks_preprocess.py --preprocessors save_split_jsonl --input $input_path --split $split
wait