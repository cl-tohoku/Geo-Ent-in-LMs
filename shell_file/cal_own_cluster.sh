model="albert-xxlarge-v2"
#"bert-base-cased"
#  model_list=(\
#  "bert-base-uncased"
#  "bert-large-uncased" \
#  "bert-base-cased"
#  "bert-large-cased" \
#  "roberta-base" \ 
#  "roberta-large" \ 
#  "albert-base-v2" \
#  "albert-large-v2" \
#  "albert-xxlarge-v2" \
#  "distilbert-base-uncased" \
#  "studio-ousia/luke-base" \
#  "studio-ousia/luke-large" \
#  )

echo $model

cuda_number=0



## 【NEの周辺文脈が多様】 
jsonl_path=(\
"/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1_5/sampled_15962/wikilinks_ne.jsonl" \
"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne.jsonl" \
)
emb_path=(\
#"/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1_5/sampled_15962/$model/wikilinks_ne_tensor_$layer.pt" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/$model/sampled_non_ne_tensor_$layer.pt" \
"/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1_5/sampled_15962/$model/wikilinks_ne_tensor_lastLayer.pt" \
"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/$model/sampled_non_ne_tensor_lastLayer.pt" \
)
#output_path="./result/$model/VariousContext/subword1_5/cluster_ratio_$layer.jsonl"
output_path="./result/$model/VariousContext/subword1_5/cluster_ratio_lastLayer.jsonl"
python3 cal_own_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path --do_series --cuda_number $cuda_number




#### 【NEの周辺文脈が多様】 subword=1
#jsonl_path=(\
#"/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1/balanced_ratio_sampling/wikilinks_ne.jsonl" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne.jsonl" \
#)
#emb_path=(\
##"/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1/balanced_ratio_sampling/$model/wikilinks_ne_tensor_$layer.pt" \
##"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/$model/sampled_non_ne_tensor_$layer.pt"
#"/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1/balanced_ratio_sampling/$model/wikilinks_ne_tensor_lastLayer.pt" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/$model/sampled_non_ne_tensor_lastLayer.pt" \
#)
#output_path="./result/$model/VariousContext/subword1/balanced_ratio_sampling/cluster_ratio_lastLayer.jsonl"
#python3 cal_own_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path --do_series --cuda_number $cuda_number




### 【NEの周辺文脈とメンションが多様】 
#jsonl_path=(\
#"/data/wikilinks/ne/Person_Location/various_context_surface/delete_symbol/unique_sentence/separate/sampled_15962/wikilinks_ne.jsonl" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne.jsonl" \
#)
#emb_path=(\
##"/data/wikilinks/ne/Person_Location/various_context_surface/delete_symbol/unique_sentence/separate/sampled_15962/$model/wikilinks_ne_tensor_$layer.pt" \
##"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/$model/sampled_non_ne_tensor_$layer.pt"
#"/data/wikilinks/ne/Person_Location/various_context_surface/delete_symbol/unique_sentence/separate/sampled_15962/$model/wikilinks_ne_tensor_lastLayer.pt" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/$model/sampled_non_ne_tensor_lastLayer.pt" \
#)
##output_path="./result/$model/VariousContextMention/cluster_ratio_$layer.jsonl"
#output_path="./result/$model/VariousContextMention/cluster_ratio_lastLayer.jsonl"
#python3 cal_own_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path --do_series --is_group_by_Wiki_id --cuda_number $cuda_number


