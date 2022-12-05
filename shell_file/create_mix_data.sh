
target_word_path_list=(\
"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/target_word_list_delete_symbol_preprocessed_512tokens_non_ne_vocabwikilinks_more_10.bin" \
"/data/wikilinks/ne/512token/aggregated/separete/target_word_list_preprocessed_512token_wikilinks_more_10_vocab.bin" \
)

sentence_path_list=(\
"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/sentence_list_delete_symbol_preprocessed_512tokens_non_ne_vocabwikilinks_more_10.bin" \
"/data/wikilinks/ne/512token/aggregated/separete/sentence_list_preprocessed_512token_wikilinks_more_10_vocab.bin" \
)

word_type_path_list=(\
"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/word_type_list_delete_symbol_preprocessed_512tokens_non_ne_vocabwikilinks_more_10.bin" \
"/data/wikilinks/ne/512token/aggregated/separete/word_type_list_preprocessed_512token_wikilinks_more_10_vocab.bin" \
)

SEED=42
MAX_SENTENCE_NUM=10100000
output_path="/data/wikilinks/mix/ne_and_non_ne_10100000.jsonl"

python3 wikilinks_preprocess.py --preprocessors create_mix_data --target_word_path_list ${target_word_path_list[@]} --sentence_path_list ${sentence_path_list[@]} --word_type_path_list ${word_type_path_list[@]} --SEED $SEED --MAX_SENTENCE_NUM $MAX_SENTENCE_NUM --output $output_path
wait

