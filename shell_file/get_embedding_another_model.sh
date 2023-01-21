
model_list=(\
"bert-large-uncased" \
"roberta-base" \ ## not uncased
"roberta-large" \ ## not uncased
"albert-base-v2" \
"albert-large-v2" \
"albert-xxlarge-v2" \
"distilbert-base-uncased" \
"studio-ousia/luke-base" \
"studio-ousia/luke-large" \
)


for model in ${model_list[@]}
do  
    ## NE  (Person + Location) (多様なNE表層 & 多様なNE Alias) (多様なNE表層とNEをあわせた)
    input_path="/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_ne.jsonl"
    output_path="/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/$model/wikilinks_ne_tensor_$model.pt"
    batch_size=256
    python3 bert-embedding.py --input $input_path  --output $output_path --batch_size $batch_size --model $model
    ### Non_NE
    input_path="/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne.jsonl"
    output_path="/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/$model/sampled_non_ne_tensor_$model.pt"
    batch_size=256
    python3 bert-embedding.py --input $input_path  --output $output_path --batch_size $batch_size --model $model
done

wait
