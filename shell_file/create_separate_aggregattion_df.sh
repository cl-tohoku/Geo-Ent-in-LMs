#input_path="/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10.jsonl"
#input_path="/data/wikilinks/common_noun/512token/reduced_targetword/aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10.jsonl"

#split=50
#python3 wikilinks_preprocess.py --preprocessors create_separete_aggregattion_df --input $input_path --split $split
#wait

#neの以前の形式のデータを target_word, sentenceの形にする
#input_path="/data/wikilinks/ne/512token/aggregated/preprocessed_512token_wikilinks_more_10_vocab.jsonl"
#input_path="/work/masaki/data/wikilinks/ne/Person_Location/aggregated/replaced_wikilinks_more10_title.jsonl"

#split=4
#python3 wikilinks_preprocess.py --preprocessors create_separete_aggregattion_df --input $input_path --split $split

## NE(Person + Location サンプリング済み) (周辺文脈が多様)
#input_path="/work/masaki/data/wikilinks/ne/Person_Location/aggregated/subword1_5/sampled/16056/sampled_wikilinks_ne.jsonl"
#output_path="/work/masaki/data/wikilinks/ne/Person_Location/separete/subword1_5/sampled/16056/sampled_wikilinks_ne.jsonl"
#python3 wikilinks_preprocess.py --preprocessors create_separate_df --input $input_path --output $output_path --is_ne


## Non NE (サンプリング済み) (センテンスの重複あり)
#input_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/aggregated/sampled/954067/filtered/sampled_non_ne.jsonl"
#output_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/separete/sampled/954067/filtered/sampled_non_ne.jsonl"
#python3 wikilinks_preprocess.py --preprocessors create_separate_df --input $input_path --output $output_path 

## NE (サンプリング済み) (NEの表層が多様 (Alias)) サンプル済み
#input_path="/work/masaki/data/wikilinks/ne/Person_Location/aggregated/ne_alias/wikilinks_more10.jsonl"
#output_path="/work/masaki/data/wikilinks/ne/Person_Location/separete/ne_alias/wikilinks_more10.jsonl"
#python3 wikilinks_preprocess.py --preprocessors create_separate_df --input $input_path --output $output_path --is_ne


## NE（多様な周辺文脈 and 多様な表層） 未サンプリング
#input_path="/work/masaki/data/wikilinks/ne/Person_Location/unique_sentence/aggregated/wikilinks_more10.jsonl"
#output_path="/work/masaki/data/wikilinks/ne/Person_Location/unique_sentence/separete/wikilinks_more10.jsonl"
#python3 wikilinks_preprocess.py --preprocessors create_separate_df --input $input_path --output $output_path --is_ne


## NE（多様な周辺文脈 and 多様な表層） サンプリング済み
#input_path="/work/masaki/data/wikilinks/ne/Person_Location/unique_sentence/aggregated/sampled_15962/wikilinks_more10.jsonl"
#output_path="/work/masaki/data/wikilinks/ne/Person_Location/unique_sentence/separete/sampled_15962/wikilinks_more10.jsonl"
#python3 wikilinks_preprocess.py --preprocessors create_separate_df --input $input_path --output $output_path --is_ne


## Non NE (サンプリング済み) (センテンスの重複なし)
#input_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/aggregated/sampled/954067/filtered/sampled_non_ne.jsonl"
#output_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne.jsonl"
#python3 wikilinks_preprocess.py --preprocessors create_separate_df --input $input_path --output $output_path 

## NE(Person + Location ) (周辺文脈が多様)(センテンスの重複なし)
#input_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/unique_sentence/aggregated/subword1_5/wikilinks_ne.jsonl"
#output_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separete/subword1_5/wikilinks_ne.jsonl"
#python3 wikilinks_preprocess.py --preprocessors create_separate_df --input $input_path --output $output_path --is_ne

## NE(Person + Location ) サンプリング済み (周辺文脈が多様)(センテンスの重複なし)
#input_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/unique_sentence/aggregated/subword1_5/sampled_15962/wikilinks_ne.jsonl"
#output_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separete/subword1_5/sampled_15962/wikilinks_ne.jsonl"
#python3 wikilinks_preprocess.py --preprocessors create_separate_df --input $input_path --output $output_path --is_ne

## NE(Person + Location ) サンプリング済み (周辺文脈と表層が多様)(センテンスの重複なし)　(周辺文脈多様とのNEが同じ)
input_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/aggregated/sampled_15962/wikilinks_ne.jsonl"
output_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_ne.jsonl"
python3 wikilinks_preprocess.py --preprocessors create_separate_df --input $input_path --output $output_path --is_ne


wait