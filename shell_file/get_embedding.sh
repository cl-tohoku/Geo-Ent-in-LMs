

model="studio-ousia/luke-base"

#model_list=(\
#  "bert-base-uncased" \
#  "bert-large-uncased" \
#  "bert-base-cased" \
#  "bert-large-cased" \
#  "roberta-base" \ 
#  "roberta-large" \ 
#  "albert-base-v2" \
#  "albert-large-v2" \
#  "albert-xxlarge-v2" \
#  "distilbert-base-uncased" \
#  "studio-ousia/luke-base" \
#  "studio-ousia/luke-large" \
#)

cuda_number=3


## NE  (Person + Location) (多様な周辺文脈) 
input_path="/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1_5/sampled_15962/wikilinks_ne.addSpan.jsonl"
output_path="/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1_5/sampled_15962/$model/wikilinks_ne_tensor_lastLayer.pt"
batch_size=64
python3 get-embedding.py --input $input_path  --output $output_path --batch_size $batch_size   --model $model  --cuda_number $cuda_number --is_ne

#
#
### NE  (Person + Location) (多様な周辺文脈) & Subword=1
#input_path="/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1/balanced_ratio_sampling/wikilinks_ne.addSpan.jsonl"
#output_path="/data/wikilinks/ne/Person_Location/various_context/delete_symbol/unique_sentence/separate/subword1/balanced_ratio_sampling/$model/wikilinks_ne_tensor_lastLayer.pt"
#batch_size=64
#python3 get-embedding.py --input $input_path  --output $output_path --batch_size $batch_size   --model $model  --cuda_number $cuda_number --is_ne


## NE  (Person + Location) (多様な周辺文脈 & 多様なメンション)
#input_path="/data/wikilinks/ne/Person_Location/various_context_mention/delete_symbol/unique_sentence/separate/sampled_15962/wikilinks_ne.addSpan.jsonl"
#output_path="/data/wikilinks/ne/Person_Location/various_context_mention/delete_symbol/unique_sentence/separate/sampled_15962/$model/wikilinks_ne_tensor_lastLayer.pt"
#batch_size=64
#python3 get-embedding.py --input $input_path  --output $output_path --batch_size $batch_size   --model $model  --cuda_number $cuda_number --is_ne



## Non_NE
#input_path="/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/sampled_non_ne.jsonl"
#output_path="/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unicordNormalizeSentence/unique_sentence/separate/sampled/954067/filtered/$model/sampled_non_ne_tensor_lastLayer.pt"
#batch_size=32
#python3 get-embedding.py --input $input_path  --output $output_path --batch_size $batch_size   --model $model  --cuda_number $cuda_number

