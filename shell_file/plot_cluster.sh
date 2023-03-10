
layer=12
model="bert-base-uncased"

# 【NEの周辺文脈とメンションが多様】 delete_symbol　なし
#jsonl_path=(\
#"/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separate/sampled_15962/wikilinks_ne.jsonl" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne.jsonl" \
#)
#emb_path=(\
#"/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separate/sampled_15962/${model}/wikilinks_ne_tensor_layer$layer.pt" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne_tensor_layer$layer.pt"
#)
#output_path="./result/plot_cluster_layer$layer.png"
#python3 ./tool/plot_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path  --is_group_by_Wiki_id




#delete_symbol あり



# 【周辺文脈とNE表層が多様】 delete_symbol 
#jsonl_path=(\
#"/data/wikilinks/ne/Person_Location/various_context_surface/delete_symbol/unique_sentence/separate/sampled_15962/wikilinks_ne.jsonl" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne.jsonl" \
#)
#
#emb_path=(\
#"/data/wikilinks/ne/Person_Location/various_context_surface/delete_symbol/unique_sentence/separate/sampled_15962/$model/wikilinks_ne_tensor_layer12.pt" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/$model/sampled_non_ne_tensor_layer12.pt"
#)



## 【NEの周辺文脈とメンションが多様】 
jsonl_path=(\
"/data/wikilinks/ne/Person_Location/various_context_surface/delete_symbol/unique_sentence/separate/sampled_15962/wikilinks_ne.jsonl" \
"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne.jsonl" \
)
emb_path=(\
#"/data/wikilinks/ne/Person_Location/various_context_surface/delete_symbol/unique_sentence/separate/sampled_15962/$model/wikilinks_ne_tensor_$layer.pt" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/$model/sampled_non_ne_tensor_$layer.pt"
"/data/wikilinks/ne/Person_Location/various_context_surface/delete_symbol/unique_sentence/separate/sampled_15962/$model/wikilinks_ne_tensor_layer$layer.pt" \
"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/$model/sampled_non_ne_tensor_layer$layer.pt" \
)


# plot_nearest_cluster
#plot_target_word="Erykah Badu"
#numberOfClusters=10
#marker_size=32
#output_path="./result/${model}/VariousContextMention/plot_nearest_cluster_Erykah_Badu_${numberOfClusters}_layer$layer.png"
#echo $plot_target_word
#python3 ./tool/plot_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path  --is_group_by_Wiki_id --is_plot_nearest_cluster --plot_target_word "$plot_target_word" --numberOfClusters $numberOfClusters --marker_size $marker_size



# plot_nearest_cluster
plot_target_word="Alberto R. Gonzales"
numberOfClusters=50
marker_size=150
output_path="./result/${model}/VariousContextMention/plot_nearest_cluster_Alberto_${numberOfClusters}_layer$layer.png"
echo $plot_target_word
python3 ./tool/plot_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path  --is_group_by_Wiki_id --is_plot_nearest_cluster --plot_target_word "$plot_target_word" --numberOfClusters $numberOfClusters --marker_size $marker_size
