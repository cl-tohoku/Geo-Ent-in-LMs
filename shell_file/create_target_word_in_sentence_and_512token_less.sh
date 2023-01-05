#input_path_list=(\
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split1.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split2.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split3.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split4.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split5.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split6.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split7.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split8.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split9.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split10.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split11.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split12.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split13.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split14.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split15.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split16.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split17.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split18.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split19.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split20.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split21.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split22.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split23.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split24.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split25.jsonl" \
#"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split26.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split27.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split28.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split29.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split30.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split31.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split32.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split33.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split34.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split35.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split36.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split37.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split38.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split39.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split40.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split41.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split42.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split43.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split44.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split45.jsonl" \
#"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split46.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split47.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split48.jsonl" \
#"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split49.jsonl" \
##"/data/wikilinks/preprocessed_non_ne_vocabwikilinks_more_10_split50.jsonl" \
#)

# ga02
#input_path_list=(\
#"/data/wikilinks/mix/ne_and_non_ne_10000000_split1.jsonl" \
#"/data/wikilinks/mix/ne_and_non_ne_10000000_split2.jsonl" \
#"/data/wikilinks/mix/ne_and_non_ne_10000000_split3.jsonl" \
#"/data/wikilinks/mix/ne_and_non_ne_10000000_split4.jsonl" \
#)

# ga01
#input_path_list=(\
#"/data/wikilinks/mix/ne_and_non_ne_10000000.jsonl" \
#)

# NE(Person+Location) (多様な周辺文脈）
#input_path_list=(\
#"/work/masaki/data/wikilinks/ne/Person_Location/separete/replaced_wikilinks_more10_title.jsonl" \
#)

## NE（多様な周辺文脈 and 多様な表層） サンプリング済み
input_path_list=(\
"/work/masaki/data/wikilinks/ne/Person_Location/unique_sentence/separete/sampled_15962/wikilinks_more10.jsonl"
)

batch_size=2048

for input_path in ${input_path_list[@]}
do  
    (python3 wikilinks_preprocess.py --preprocessors create_target_word_in_sentence_and_512token_less --input $input_path --batch_size $batch_size --is_ne) &
done
wait