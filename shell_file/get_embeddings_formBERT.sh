

model="bert-base-uncased"



## NE  (Person + Location) (多様な周辺文脈) 
input_path="/data/ne/various_context/separate/wikilinks_ne.jsonl"
output_path="/model/ne/various_context/$model/wikilinks_ne_tensor_lastLayer.pt"
batch_size=64
python3 get-embedding.py --input $input_path  --output $output_path --batch_size $batch_size   --model $model   --is_ne

# NE  (Person + Location) (多様な周辺文脈 & 多様なメンション)
input_path="/data/ne/various_context_mention/separate/wikilinks_ne.jsonl"
output_path="/model/ne/various_context_mention/$model/wikilinks_ne_tensor_lastLayer.pt"
batch_size=64
python3 get-embedding.py --input $input_path  --output $output_path --batch_size $batch_size   --model $model   --is_ne



## Non_NE
input_path="/data/non_ne/separate/sampled_non_ne.jsonl"
output_path="/model/non_ne/$model/sampled_non_ne_tensor_lastLayer.pt"
batch_size=32
python3 get-embedding.py --input $input_path  --output $output_path --batch_size $batch_size   --model $model  

