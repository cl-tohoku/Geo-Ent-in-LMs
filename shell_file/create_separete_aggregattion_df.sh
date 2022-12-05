#input_path="/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10.jsonl"
#input_path="/data/wikilinks/common_noun/512token/reduced_targetword/aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10.jsonl"

#split=50
#python3 wikilinks_preprocess.py --preprocessors create_separete_aggregattion_df --input $input_path --split $split
#wait

#neの以前の形式のデータを target_word, sentenceの形にする
input_path="/data/wikilinks/ne/512token/aggregated/preprocessed_512token_wikilinks_more_10_vocab.jsonl"
#split=4
#python3 wikilinks_preprocess.py --preprocessors create_separete_aggregattion_df --input $input_path --split $split

output_path="/data/wikilinks/ne/512token/separete/preprocessed_512token_wikilinks_more_10_vocab.jsonl"
python3 wikilinks_preprocess.py --preprocessors create_separete_aggregattion_df --input $input_path --output $output_path

wait