# Geometry of Entity Representations in LMs
code for [事前学習済み言語モデルによるエンティティの概念化(NLP2023)](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/B6-4.pdf)


# Setup
[This Docker repository]() is utilized for constructing the environment.



# Reproducing our experiments
## Data downlad
The dataset for our experiments is available at [url](https://drive.google.com/file/d/1vWoT956XcQB9ApLGMgysiJZbr8O8oyzD/view?usp=sharing).  
Please download the file "reproduction_data_NLP2023.zip" from Google Drive into your designated data directory, and then proceed to unzip the file.

## Visualization of our results
The repository's ``result`` directory already includes the results of the experiment, which makes reproducing the visualization easy.

Run experiments according to [visualization_reproduction.ipynb]().

## Get Contextual word Embeddings from the model


Please ensure that the 'data_dir_path' in ``get_embeddings_formBERT.sh`` corresponds to the directory where you have extracted the 'data.zip' file.

You can get the embeddings from BERT by executing the following code:
```
bash ./shell_file/get_embeddings_formBERT.sh
```


## Calculation of condensation rate
The condensation rate can be calculated to measure the degree of separation for each cluster.
You can run the program below to calculate the condensation rate and get the result.
```
bash ./shell_file/cal_condensation_ratio.sh
```
