# NE (all data)
#input_path="/work/masaki/data/wikilinks/ne/Person_Location/aggregated/wikilinks_more10.jsonl"
#output_path="/work/masaki/data/wikilinks/ne/Person_Location/unique_sentence/aggregated/wikilinks_more10.jsonl"
#python3 wikilinks_preprocess.py --preprocessors extractUniqueSentences --input $input_path --output $output_path 


# Non NE (sampled data) input:separate
#input_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/separete/sampled/954067/filtered/sampled_non_ne.jsonl"
#output_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/aggregated/sampled/954067/filtered/sampled_non_ne.jsonl"
#emb_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/separete/sampled/954067/filtered/sampled_non_ne_tensor.pt"
#output_emb_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne_tensor.pt"
#python3 wikilinks_preprocess.py --preprocessors extractUniqueSentences --input $input_path --emb_path $emb_path  --output $output_path --output_emb $output_emb_path --is_separated_df 

# NE (Person Location)  (点群数のフィルター済み) センテンスの重複なし
input_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/separete/subword1_5/wikilinks_ne.jsonl"
output_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/unique_sentence/aggregated/subword1_5/wikilinks_ne.jsonl"
emb_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/separete/subword1_5/wikilinks_ne_tensor.pt"
output_emb_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separete/subword1_5/wikilinks_ne_tensor.pt"
python3 wikilinks_preprocess.py --preprocessors extractUniqueSentences --input $input_path --emb_path $emb_path  --output $output_path --output_emb $output_emb_path --is_separated_df --is_ne