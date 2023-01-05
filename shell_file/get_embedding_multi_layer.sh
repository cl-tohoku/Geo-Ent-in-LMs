
layer_list=(\
1 2 3 4 4 5 6 7 8 9 10 11
)


for layer in ${layer_list[@]}
do  
    ## NE  (Person + Location) (多様なNE表層 & 多様なNE Alias) (多様なNE表層とNEをあわせた)
    input_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_ne.jsonl"
    output_path="/work/masaki/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_ne_tensor_layer$layer.pt"
    batch_size=2048
    python3 bert-embedding.py --input $input_path  --output $output_path --batch_size $batch_size --layer $layer
    ### Non_NE
    input_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne.jsonl"
    output_path="/work/masaki/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne_tensor_layer$layer.pt"
    batch_size=2048
    python3 bert-embedding.py --input $input_path  --output $output_path --batch_size $batch_size --layer $layer
done

wait
