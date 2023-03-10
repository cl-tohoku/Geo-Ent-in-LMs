model="bert-base-uncased"

echo $model

data_dir_path="your_data_dir_path"

# e.g.
data_dir_path="data"


## 【NEの周辺文脈が多様】 
jsonl_path=(\
"/$data_dir_path/ne/various_context/separate/wikilinks_ne.jsonl" \
"/$data_dir_path/non_ne/separate/sampled_non_ne.jsonl" \
)
emb_path=(\
"/model/ne/various_context/$model/wikilinks_ne_tensor_lastLayer.pt" \
"/model/non_ne/$model/sampled_non_ne_tensor_lastLayer.pt" \
)
output_path="./result/$model/VariousContext/cluster_ratio_lastLayer.jsonl"
python3 cal_own_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path --do_series



## 【NEの周辺文脈とメンションが多様】 
jsonl_path=(\
"/$data_dir_path/ne/various_context_mention/separate/wikilinks_ne.jsonl" \
"/$data_dir_path/non_ne/separate/sampled_non_ne.jsonl" \
)
emb_path=(\
"/model/ne/various_context_mention/$model/wikilinks_ne_tensor_lastLayer.pt" \
"/model/non_ne/$model/sampled_non_ne_tensor_lastLayer.pt" \
)
output_path="./result/$model/VariousContextMention/cluster_ratio_lastLayer.jsonl"
python3 cal_own_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path --do_series --is_group_by_Wiki_id

