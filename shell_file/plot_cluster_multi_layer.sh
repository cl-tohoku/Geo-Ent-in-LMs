
layer_list=(\
1 2 3 4 5 6 7 8 9 10 11 12
)

model="bert-base-uncased"
#  model_list=(\
#  "bert-base-uncased"
#  "bert-large-uncased" \
#  "roberta-base" \ 
#  "roberta-large" \ 
#  "albert-base-v2" \
#  "albert-large-v2" \
#  "albert-xxlarge-v2" \
#  "distilbert-base-uncased" \
#  "studio-ousia/luke-base" \
#  "studio-ousia/luke-large" \
#  )

for layer in ${layer_list[@]}
do  
    # 【NEの周辺文脈が多様】 
    jsonl_path=(\
    "/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separate/subword1_5/sampled_15962/wikilinks_ne.jsonl" \
    "/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne.jsonl" \
    )
    emb_path=(\
    "/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separate/subword1_5/sampled_15962/$model/wikilinks_ne_tensor_layer$layer.pt" \
    "/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne_tensor_layer$layer.pt"
    )
    output_path="./result/$model/VariousContext/subword1_5/plot_cluster_BERTlayer$layer.png"
    python3 ./tool/plot_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path 

    ## 【NEの周辺文脈が多様】 subword長=1
    #jsonl_path=(\
    #"/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separate/subword1/sampled_15962/wikilinks_ne.jsonl" \
    #"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne.jsonl" \
    #)
    #emb_path=(\
    #"/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separate/subword1/sampled_15962/$model/wikilinks_ne_tensor_layer$layer.pt" \
    #"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne_tensor_layer$layer.pt"
    #)
    #output_path="./result/$model/VariousContext/subword1/plot_cluster_BERTlayer$layer.png"
    #python3 ./tool/plot_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path 

    

    ## 【NEの周辺文脈とメンションが多様】 
    #jsonl_path=(\
    #"/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_ne.jsonl" \
    #"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne.jsonl" \
    #)

    #emb_path=(\
    #"/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_ne_tensor_layer$layer.pt" \
    #"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne_tensor_layer$layer.pt"
    #)
    #output_path="./result/plot_cluster_BERTlayer$layer.png"
    #python3 ./tool/plot_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path  --is_group_by_Wiki_id

done

wait