
# NE (Person Location)  (点群数のフィルター済み) センテンスの重複なし

input_path_list=(\
"/work/masaki/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/aggregated/wikilinks_more10.jsonl" \
"/work/masaki/data/wikilinks/ne/Person_Location/various_context/unique_sentence/aggregated/subword1_5/sampled_15962/wikilinks_ne.jsonl" )

output_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/aggregated/sampled_15962/wikilinks_ne.jsonl"
python3 wikilinks_preprocess.py --preprocessors create_various_context_surface_df --input_list ${input_path_list[@]}   --output $output_path 