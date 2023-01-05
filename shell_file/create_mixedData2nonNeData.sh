
input_path_list=(\
"/work/masaki/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split1.jsonl" \
"/work/masaki/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split2.jsonl" \
"/work/masaki/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split3.jsonl" \
"/work/masaki/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split4.jsonl" \
)

emb_path_list=(\
"/work/masaki/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split1_tensor.pt" \
"/work/masaki/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split2_tensor.pt" \
"/work/masaki/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split3_tensor.pt" \
"/work/masaki/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split4_tensor.pt" \
)



output_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/sampled/1000000/non_ne_998125.jsonl"

python3 wikilinks_preprocess.py --preprocessors create_mixedData2nonNeData --input_list ${input_path_list[@]} --emb_path_list ${emb_path_list[@]}  --output $output_path
wait

