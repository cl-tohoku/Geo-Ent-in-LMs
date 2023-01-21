
layer_list=(\
1 2 3 4 5 6 7 8 9 10 11 12
)

for layer in ${layer_list[@]}
do  

    # 【周辺文脈とNE表層が多様】 ne(Person Location) と Non_ne の Mixデータ (点群数のフィルター済み)  センテンスの重複なし (周辺文脈のやつにNEをあわせた)
    jsonl_path=(\
    "/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_ne.jsonl" \
    "/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne.jsonl" \
    )

    emb_path=(\
    "/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_ne_tensor_layer$layer.pt" \
    "/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne_tensor_layer$layer.pt"
    )
    output_path="./result/plot_cluster_BERTlayer$layer.png"
    python3 plot_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path  --is_group_by_Wiki_id

done

wait