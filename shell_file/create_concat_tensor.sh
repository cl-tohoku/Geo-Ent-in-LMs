
input_path_list=(\
    "/data/wikilinks/ne/512token/aggregated/aggregated_split/preprocessed_512token_wikilinks_more_10_vocab_split1_split1_tensor.pt" \
    "/data/wikilinks/ne/512token/aggregated/aggregated_split/preprocessed_512token_wikilinks_more_10_vocab_split1_split2_tensor.pt" \
    "/data/wikilinks/ne/512token/aggregated/aggregated_split/preprocessed_512token_wikilinks_more_10_vocab_split1_split3_tensor.pt" \
    "/data/wikilinks/ne/512token/aggregated/aggregated_split/preprocessed_512token_wikilinks_more_10_vocab_split1_split4_tensor.pt" \
    "/data/wikilinks/ne/512token/aggregated/aggregated_split/preprocessed_512token_wikilinks_more_10_vocab_split2_tensor.pt" \
    "/data/wikilinks/ne/512token/aggregated/aggregated_split/preprocessed_512token_wikilinks_more_10_vocab_split3_tensor.pt" \
    "/data/wikilinks/ne/512token/aggregated/aggregated_split/preprocessed_512token_wikilinks_more_10_vocab_split4_tensor.pt" \

)


python3 wikilinks_preprocess.py --preprocessors create_concat_tensor --input_list ${input_path_list[@]} 
wait

