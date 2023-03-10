
layer_list=(\
1 2 3 4 5 6 7 8 9 10 11 12 #13 14 15 16 17 18 19 20 21 22 23 24
)

#model="bert-base-uncased"
#model="roberta-base"
#model="bert-large-uncased"
#model="distilbert-base-uncased"
model_list=(\
#"roberta-base" \
#"roberta-large" \
  "bert-base-uncased" \
#  "bert-large-uncased" \
#  "roberta-base" \ 
#  "roberta-large" \ 
#  "albert-base-v2" \
#  "albert-large-v2" \
#  "albert-xxlarge-v2" \
#  "distilbert-base-uncased" \
#  "studio-ousia/luke-base" \
#  "studio-ousia/luke-large" \
)

cuda_number=3


for model in ${model_list[@]}
do
    for layer in ${layer_list[@]}
    do  
    

        ### NE  (Person + Location) (多様な周辺文脈) & Subword=1
        #input_path="/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1/balanced_ratio_sampling/add_span/remove_NE_ambiguous/wikilinks_ne.jsonl"
        #output_path="/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1/balanced_ratio_sampling/add_span/remove_NE_ambiguous/$model/wikilinks_ne_tensor_layer$layer.pt"
        #batch_size=64
        #python3 get-embedding.py --input $input_path  --output $output_path --batch_size $batch_size --layer $layer  --model $model  --cuda_number $cuda_number
#
#
        ### NE  (Person + Location) (多様な周辺文脈) 
        #input_path="/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1_5/sampled_15962/add_span/remove_NE_ambiguous/wikilinks_ne.jsonl"
        #output_path="/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1_5/sampled_15962/add_span/remove_NE_ambiguous/w$model/wikilinks_ne_tensor_layer$layer.pt"
        #batch_size=64
        #python3 get-embedding.py --input $input_path  --output $output_path --batch_size $batch_size --layer $layer  --model $model  --cuda_number $cuda_number


        ## NE  (Person + Location) (多様な周辺文脈 & 多様なメンション)
        #input_path="/data/wikilinks/ne/Person_Location/various_context_mention/delete_symbol/unique_sentence/separate/sampled_15962/wikilinks_ne.jsonl"
        input_path="/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1_5/sampled_15962/add_span/remove_NE_ambiguous/wikilinks_ne.jsonl"
        output_path="/model/wikilinks/ne/Person_Location/various_context_mention/delete_symbol/unique_sentence/separate/sampled_15962/add_span/remove_NE_ambiguous/$model/wikilinks_ne_tensor_layer$layer.pt"
        batch_size=64
        python3 get-embedding.py --input $input_path  --output $output_path --batch_size $batch_size --layer $layer  --model $model  --cuda_number $cuda_number
      

        ### Non_NE
        #input_path="/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne.jsonl"
        #output_path="/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/$model/sampled_non_ne_tensor_layer$layer.pt"
        #batch_size=32
        #python3 get-embedding.py --input $input_path  --output $output_path --batch_size $batch_size --layer $layer  --model $model  --cuda_number $cuda_number
    
    done
done

wait
