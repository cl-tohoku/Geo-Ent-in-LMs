
## NE  (Person + Location) (多様な周辺文脈)
#input_path_list=(\
#"/work/masaki/data/wikilinks/ne/Person_Location/separete/target_word_in_sentence_replaced_wikilinks_more10_title.jsonl" \
#)

## NE  (Person + Location) (多様なNE表層 & 多様なNE Alias)
#input_path_list=(\
#"/work/masaki/data/wikilinks/ne/Person_Location/unique_sentence/separete/sampled_15962/wikilinks_more10.jsonl"
#)

## NE  (Person + Location) (多様なNE表層 & 多様なNE Alias) (多様なNE表層とNEをあわせた)
input_path_list=(\
"/work/masaki/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_ne.jsonl"
)

batch_size=2048

for path in ${input_path_list[@]}
do  
    (python3 bert-embedding.py --input $path  --batch_size $batch_size) &
done
wait



