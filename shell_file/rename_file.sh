
before_name_path=(\
"/data/wikilinks/ne/512token/preprocessed_wikilinks_more_10_vocab_split2_ave_tensor.pt" \
"/data/wikilinks/ne/512token/preprocessed_wikilinks_more_10_vocab_split2.jsonl" \
"/data/wikilinks/ne/512token/preprocessed_wikilinks_more_10_vocab_split2_tensor.pt" \
"/data/wikilinks/ne/512token/preprocessed_wikilinks_more_10_vocab_split3_ave_tensor.pt" \
"/data/wikilinks/ne/512token/preprocessed_wikilinks_more_10_vocab_split3.jsonl" \
"/data/wikilinks/ne/512token/preprocessed_wikilinks_more_10_vocab_split3_tensor.pt" \
"/data/wikilinks/ne/512token/preprocessed_wikilinks_more_10_vocab_split4_ave_tensor.pt" \
"/data/wikilinks/ne/512token/preprocessed_wikilinks_more_10_vocab_split4.jsonl" \
"/data/wikilinks/ne/512token/preprocessed_wikilinks_more_10_vocab_split4_tensor.pt" \
)



after_name_path=(\
"/data/wikilinks/ne/512token/preprocessed_512token_wikilinks_more_10_vocab_split2_ave_tensor.pt" \
"/data/wikilinks/ne/512token/preprocessed_512token_wikilinks_more_10_vocab_split2.jsonl" \
"/data/wikilinks/ne/512token/preprocessed_512token_wikilinks_more_10_vocab_split2_tensor.pt" \
"/data/wikilinks/ne/512token/preprocessed_512token_wikilinks_more_10_vocab_split3_ave_tensor.pt" \
"/data/wikilinks/ne/512token/preprocessed_512token_wikilinks_more_10_vocab_split3.jsonl" \
"/data/wikilinks/ne/512token/preprocessed_512token_wikilinks_more_10_vocab_split3_tensor.pt" \
"/data/wikilinks/ne/512token/preprocessed_512token_wikilinks_more_10_vocab_split4_ave_tensor.pt" \
"/data/wikilinks/ne/512token/preprocessed_512token_wikilinks_more_10_vocab_split4.jsonl" \
"/data/wikilinks/ne/512token/preprocessed_512token_wikilinks_more_10_vocab_split4_tensor.pt" \
)


for ix in ${!before_name_path[@]}
do  
    mv ${before_name_path[ix]}  ${after_name_path[ix]}
done
wait

