input_path_list=(\
"/data/wikilinks/wikilinks_more_10_split1.jsonl" \
"/data/wikilinks/wikilinks_more_10_split2.jsonl" \
"/data/wikilinks/wikilinks_more_10_split3.jsonl" \
"/data/wikilinks/wikilinks_more_10_split4.jsonl" \
"/data/wikilinks/wikilinks_more_10_split5.jsonl" \
"/data/wikilinks/wikilinks_more_10_split6.jsonl" \
"/data/wikilinks/wikilinks_more_10_split7.jsonl" \
"/data/wikilinks/wikilinks_more_10_split8.jsonl" \
"/data/wikilinks/wikilinks_more_10_split9.jsonl" \
"/data/wikilinks/wikilinks_more_10_split10.jsonl" \
"/data/wikilinks/wikilinks_more_10_split11.jsonl" \
"/data/wikilinks/wikilinks_more_10_split12.jsonl" \
"/data/wikilinks/wikilinks_more_10_split13.jsonl" \
"/data/wikilinks/wikilinks_more_10_split14.jsonl" \
"/data/wikilinks/wikilinks_more_10_split15.jsonl" \
"/data/wikilinks/wikilinks_more_10_split16.jsonl" \
"/data/wikilinks/wikilinks_more_10_split17.jsonl" \
"/data/wikilinks/wikilinks_more_10_split18.jsonl" \
"/data/wikilinks/wikilinks_more_10_split19.jsonl" \
"/data/wikilinks/wikilinks_more_10_split20.jsonl" \
"/data/wikilinks/wikilinks_more_10_split21.jsonl" \
"/data/wikilinks/wikilinks_more_10_split22.jsonl" \
"/data/wikilinks/wikilinks_more_10_split23.jsonl" \
"/data/wikilinks/wikilinks_more_10_split24.jsonl" \
"/data/wikilinks/wikilinks_more_10_split25.jsonl" \
"/data/wikilinks/wikilinks_more_10_split26.jsonl" \
"/data/wikilinks/wikilinks_more_10_split27.jsonl" \
"/data/wikilinks/wikilinks_more_10_split28.jsonl" \
"/data/wikilinks/wikilinks_more_10_split29.jsonl" \
"/data/wikilinks/wikilinks_more_10_split30.jsonl" \
"/data/wikilinks/wikilinks_more_10_split31.jsonl" \
"/data/wikilinks/wikilinks_more_10_split32.jsonl" \
"/data/wikilinks/wikilinks_more_10_split33.jsonl" \
"/data/wikilinks/wikilinks_more_10_split34.jsonl" \
"/data/wikilinks/wikilinks_more_10_split35.jsonl" \
"/data/wikilinks/wikilinks_more_10_split36.jsonl" \
"/data/wikilinks/wikilinks_more_10_split37.jsonl" \
"/data/wikilinks/wikilinks_more_10_split38.jsonl" \
"/data/wikilinks/wikilinks_more_10_split39.jsonl" \
"/data/wikilinks/wikilinks_more_10_split40.jsonl" \
"/data/wikilinks/wikilinks_more_10_split41.jsonl" \
"/data/wikilinks/wikilinks_more_10_split42.jsonl" \
"/data/wikilinks/wikilinks_more_10_split43.jsonl" \
"/data/wikilinks/wikilinks_more_10_split44.jsonl" \
"/data/wikilinks/wikilinks_more_10_split45.jsonl" \
"/data/wikilinks/wikilinks_more_10_split46.jsonl" \
"/data/wikilinks/wikilinks_more_10_split47.jsonl" \
"/data/wikilinks/wikilinks_more_10_split48.jsonl" \
"/data/wikilinks/wikilinks_more_10_split49.jsonl" \
"/data/wikilinks/wikilinks_more_10_split50.jsonl" \
)

# df colum rename
for input_path in ${input_path_list[@]}
do  
    (python3 wikilinks_preprocess.py --input $input_path ) &
done
wait