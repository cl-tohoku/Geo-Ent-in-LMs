# NE（多様な周辺文脈）
#input_path="/work/masaki/data/wikilinks/ne/Person_Location/separete/subword1_5/wikilinks_ne.jsonl"
#emb_path="/work/masaki/data/wikilinks/ne/Person_Location/separete/subword1_5/wikilinks_ne_tensor.pt"
#sentence_count_upper=15962
#output_jsonl_path="/work/masaki/data/wikilinks/ne/Person_Location/aggregated/subword1_5/sampled_15962/sampled_wikilinks_ne.jsonl"
#output_emb_path="/work/masaki/data/wikilinks/ne/Person_Location/separete/subword1_5/sampled_15962/sampled_wikilinks_ne_tensor.pt"
#python3 wikilinks_preprocess.py --preprocessors samplingData --input $input_path --emb_path $emb_path  --output $output_jsonl_path --output_emb $output_emb_path --sentence_count_upper $sentence_count_upper --is_ne

# non_ne
#input_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/separete/sampled/998125/non_ne_998125.jsonl"
#emb_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/separete/sampled/998125/non_ne_998125_tensor.pt"
#sentence_count_upper=984038
#output_jsonl_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/aggregated/sampled/sampled_non_ne.jsonl"
#output_emb_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/separete/sampled/984038/filtered/sampled_non_ne_tensor.pt"
#python3 wikilinks_preprocess.py --preprocessors samplingData --input $input_path --emb_path $emb_path  --output $output_jsonl_path --output_emb $output_emb_path --sentence_count_upper $sentence_count_upper


# NE（多様な周辺文脈 and 多様な表層） （Embeddingは未取得）
###input_path="/work/masaki/data/wikilinks/ne/Person_Location/unique_sentence/separete/wikilinks_more10.jsonl"
#input_path="/work/masaki/data/wikilinks/ne/Person_Location/unique_sentence/aggregated/wikilinks_more10.jsonl"
###emb_path="/work/masaki/data/wikilinks/ne/Person_Location/separete/subword1_5/wikilinks_ne_tensor.pt"
#sentence_count_upper=15962
#output_jsonl_path="/work/masaki/data/wikilinks/ne/Person_Location/unique_sentence/aggregated/sampled_15962/wikilinks_more10.jsonl"
###output_emb_path="/work/masaki/data/wikilinks/ne/Person_Location/separete/subword1_5/sampled_15962/sampled_wikilinks_ne_tensor.pt"
#python3 wikilinks_preprocess.py --preprocessors samplingData --input $input_path   --output $output_jsonl_path  --sentence_count_upper $sentence_count_upper --is_ne --is_group_by_Wiki_id


# NE（多様な周辺文脈）センテンスの重複なし
input_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separete/subword1_5/wikilinks_ne.jsonl"
emb_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separete/subword1_5/wikilinks_ne_tensor.pt"
sentence_count_upper=15962
output_jsonl_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/unique_sentence/aggregated/subword1_5/sampled_15962/wikilinks_ne.jsonl"
output_emb_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separete/subword1_5/sampled_15962/wikilinks_ne_tensor.pt"
python3 wikilinks_preprocess.py --preprocessors samplingData --input $input_path --emb_path $emb_path  --output $output_jsonl_path --output_emb $output_emb_path --sentence_count_upper $sentence_count_upper --is_ne --is_separated_df

wait

