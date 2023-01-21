
# NE
#jsonl_path=(\
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split1.jsonl" \
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split2.jsonl" \
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split3.jsonl" \
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split4.jsonl" \
#    "/data/wikilinks/preprocessed_wikilinks_more_10_vocab_split2.jsonl" \
#    "/data/wikilinks/preprocessed_wikilinks_more_10_vocab_split3.jsonl" \
#    "/data/wikilinks/preprocessed_wikilinks_more_10_vocab_split4.jsonl")
#
#emb_path=(\
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split1_tensor.pt" \
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split2_tensor.pt" \
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split3_tensor.pt" \
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split4_tensor.pt" \
#    "/data/wikilinks/preprocessed_wikilinks_more_10_vocab_split2_tensor.pt" \
#    "/data/wikilinks/preprocessed_wikilinks_more_10_vocab_split3_tensor.pt" \
#    "/data/wikilinks/preprocessed_wikilinks_more_10_vocab_split4_tensor.pt")
#
#ave_emb_path=(\
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split1_ave_tensor.pt" \
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split2_ave_tensor.pt" \
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split3_ave_tensor.pt" \
#    "/data/wikilinks/preprocessed_512token_wikilinks_more_10_vocab_split1_split4_ave_tensor.pt" \
#    "/data/wikilinks/preprocessed_wikilinks_more_10_vocab_split2_ave_tensor.pt" \
#    "/data/wikilinks/preprocessed_wikilinks_more_10_vocab_split3_ave_tensor.pt" \
#    "/data/wikilinks/preprocessed_wikilinks_more_10_vocab_split4_ave_tensor.pt")
#output_path="./result/preprocessed_512token_wikilinks_more_10_vocab.jsonl"
## XXX: なぜかsplit3のデータを使用した場合だけ Bus errorとなる→よってpreprocessed_wikilinks_more_10_vocab_split3はとりあえず使用しない.と思ったらできたが．．． →いやセンテンス長か?→ メモリっぽい



# Common noun
#jsonl_path=(\
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split1.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split2.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split3.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split4.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split5.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split6.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split7.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split8.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split9.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split10.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split11.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split12.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split13.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split14.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split15.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split16.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split17.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split18.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split19.jsonl" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split20.jsonl" \
#)
#emb_path=(\
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split1_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split2_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split3_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split4_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split5_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split6_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split7_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split8_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split9_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split10_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split11_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split12_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split13_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split14_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split15_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split16_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split17_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split18_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split19_tensor.pt" \
#"/data/wikilinks/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split20_tensor.pt" \
#)
#output_path="./result/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10.jsonl"


## 「NE  (Embedding取得済み)」と「普通名詞(センテンス数をNEデータに揃えた場合)　(Embedding取得済み)」の2つのデータを混ぜて，手早く簡易的な実験結果を出してみる
#jsonl_path=(\
#"/data/wikilinks/ne/512token/separate/preprocessed_512token_wikilinks_more_10_vocab.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split1.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split2.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split3.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split4.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split5.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split6.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split7.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split8.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split9.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split10.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split11.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split12.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split13.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split14.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split15.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split16.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split17.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split18.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split19.jsonl" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split20.jsonl" \
#)
#emb_path=(\
#"/data/wikilinks/ne/512token/separate/concat_preprocessed_512token_wikilinks_more_10_vocab_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split1_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split2_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split3_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split4_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split5_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split6_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split7_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split8_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split9_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split10_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split11_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split12_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split13_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split14_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split15_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split16_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split17_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split18_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split19_tensor.pt" \
#"/data/wikilinks/common_noun/512token/reduced_sentence/reduced_aggregated_preprocessed_common_noun_512tokens_non_ne_vocabwikilinks_more_10_split20_tensor.pt" \
#)
#output_path="./result/ne_and_common_noun_redused_sentence.jsonl"

#USERNAME=$(whoami)

# ne と Non_ne の Mixデータ
#jsonl_path=(\
#"/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split1.jsonl" \
#"/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split2.jsonl" \
#"/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split3.jsonl" \
#"/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split4.jsonl" \
#)

#emb_path=(\
#"/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split1_tensor.pt" \
#"/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split2_tensor.pt" \
#"/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split3_tensor.pt" \
#"/data/wikilinks/mix/target_word_in_sentence/ne_and_non_ne_1000000_split4_tensor.pt" \
#)

# ne(Person Location) と Non_ne の Mixデータ
#jsonl_path=(\
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/sampled/1000000/non_ne_984038.jsonl" \
#"/data/wikilinks/ne/Person_Location/separete/subword1_5/sampled_15962/wikilinks_ne.jsonl" \
#)
#
#emb_path=(\
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/sampled/1000000/non_ne_984038_tensor.pt " \
#"/data/wikilinks/ne/Person_Location/separete/subword1_5/sampled_15962/wikilinks_ne_tensor.pt" \
#)



###python3 cal_own_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --ave_emb_path ${ave_emb_path[@]} --output_path $output_path --do_series
#並列では現状動かない
###python3 cal_own_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path --do_parallel

# ne(Person Location) と Non_ne の Mixデータ (点群数のフィルター済み) (表層が多様)
#jsonl_path=(\
#"/data/wikilinks/ne/Person_Location/separete/ne_alias/wikilinks_more10.jsonl" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/separete/sampled/954067/filtered/sampled_non_ne.jsonl" \
#)
#
#emb_path=(\
#"/data/wikilinks/ne/Person_Location/separete/ne_alias/wikilinks_more10_tensor.pt" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/separete/sampled/954067/filtered/sampled_non_ne_tensor.pt" \
#)
#output_path="./result/mix_Alias_ne_Location_Person_and_non_ne.jsonl"


# ne(Person Location) と Non_ne の Mixデータ (点群数のフィルター済み&センテンスの重複なし) (周辺文脈が多様 & NEの表層が多様)
#jsonl_path=(\
#"/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_more10.jsonl" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne.jsonl" \
#)
#
#emb_path=(\
#"/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_more10_tensor.pt" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne_tensor.pt" \
#)
#output_path="./result/various_context_surface_ne_Location_Person_and_non_ne.jsonl"
#
#python3 cal_own_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path --do_series --is_group_by_Wiki_id


# 【周辺文脈が多様】 ne(Person Location) と Non_ne の Mixデータ (点群数のフィルター済み)  センテンスの重複なし
#jsonl_path=(\
#"/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separete/subword1_5/sampled_15962/wikilinks_ne.jsonl" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne.jsonl" \
#)
#
#emb_path=(\
#"/data/wikilinks/ne/Person_Location/various_context/unique_sentence/separete/subword1_5/sampled_15962/wikilinks_ne_tensor.pt" \
#"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/last_layer/sampled_non_ne_tensor.pt" \
#)
#output_path="./result/UniqueSentence_VariousContext_NE_NonNE_addWrongList.jsonl"
#python3 cal_own_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path --do_series


# 【周辺文脈とNE表層が多様】 ne(Person Location) と Non_ne の Mixデータ (点群数のフィルター済み)  センテンスの重複なし (周辺文脈のやつにNEをあわせた)
jsonl_path=(\
"/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/wikilinks_ne.jsonl" \
"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/sampled_non_ne.jsonl" \
)

emb_path=(\
"/data/wikilinks/ne/Person_Location/various_context_surface/unique_sentence/separete/sampled_15962/last_layer/wikilinks_ne_tensor.pt" \
"/data/wikilinks/non_ne/512tokens/vocab/delete_symbol/unique_sentence/separete/sampled/954067/filtered/last_layer/sampled_non_ne_tensor.pt"
)
output_path="./result/NE_AWASETA_UniqueSentence_VariousContextSurface_NE_NonNE_addWrongList.jsonl"
python3 cal_own_cluster.py --jsonl_path ${jsonl_path[@]} --emb_path ${emb_path[@]} --output_path $output_path --do_series --is_group_by_Wiki_id



wait

