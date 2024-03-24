## Quickstart

### Step 1: Prepare the data
```bash
cd data
sh  download_data.sh
download data.zip from "https://drive.google.com/drive/folders/17nerxW9hP5sIFGDjtZUNiDdtsYMZIc4a"
unzip "data.zip"
mv zipdata/* retrieved_evidence/
rm -r zipdata/
sbatch 01_prepare.sh
cd ..
```

### Step 2: Train a veracity prediction (claim verification) model
```bash
cd claim-verification
#for Deberta+Tapas
sbatch --output=train-deberta-tapas.txt 01_train.sh --text 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli' --table 'google/tapas-large-finetuned-tabfact' --model deberta-tapas
#for Roberta+Tapas
sbatch --output=train-roberta-tapas.txt 01_train.sh --text 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli' --table 'google/tapas-large-finetuned-tabfact' --model roberta-tapas
#for Bart+Tapas
sbatch --output=train-bart-tapas.txt 01_train.sh --text 'ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli' --table 'google/tapas-large-finetuned-tabfact' --model bart-tapas

#for Deberta+Tapex
sbatch --output=train-deberta-tapex.txt 01_train.sh --text 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli' --table 'microsoft/tapex-large-finetuned-tabfact' --model deberta-tapex
#for Roberta+Tapex
sbatch --output=train-roberta-tapex.txt 01_train.sh --text 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli' --table 'microsoft/tapex-large-finetuned-tabfact' --model roberta-tapex
#for Bart+Tapex
sbatch --output=train-bart-tapex.txt 01_train.sh --text 'ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli' --table 'microsoft/tapex-large-finetuned-tabfact' --model bart-tapex


#For Model Pasta:
mkdir bert_weights
cd bert_weights
download Pasta model from "https://drive.google.com/drive/folders/1sqZt8Wu7PQ3ha4260E7Gcq4WipKj6LD8?usp=sharing" to bert_weights
mv spam.model spm.model
cd ..


#for Deberta+Pasta
sbatch --output=train-deberta-pasta.txt 01_train.sh --text 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli' --table './bert_weights/pasta' --model deberta-pasta
#for Roberta+Pasta
sbatch --output=train-roberta-pasta.txt 01_train.sh --text 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli' --table './bert_weights/pasta' --model roberta-pasta
#for Bart+Pasta
sbatch --output=train-bart-pasta.txt 01_train.sh --text 'ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli' --table './bert_weights/pasta' --model bart-pasta

```

### Step 3: Predict veracity relation labels
Replace the ${model_name} with the trained model in the last step, such as "deberta-tapas", "roberta-tapas", etc
```bash
sbatch --output=eval-${model_name}.txt 02_predict.sh --dataset dev --model ${model_name}
```

### Results For Deberta+Tapas

```bash
Feverous scores...
feverous score: 0.34942965779467683
label accuracy: 0.7186311787072244
```


### Results For Roberta+Tapas

```bash
Feverous scores...
feverous score: 0.34131812420785806
label accuracy: 0.703168567807351
```



### Results For Bart+Tapas

```bash
Feverous scores...
feverous score: 0.3385297845373891
label accuracy: 0.6975918884664132
```



### Results For Deberta+Tapex

```bash
Feverous scores...
feverous score: 0.3300380228136882
label accuracy: 0.6939163498098859
```


### Results For Roberta+Tapex

```bash
Feverous scores...
feverous score: 0.3223067173637516
label accuracy: 0.6727503168567808
```

### Results For Bart+Tapex

```bash
Feverous scores...
feverous score: 0.31280101394169835
label accuracy: 0.6538656527249683
```

### Results For Deberta+Pasta

```bash
Feverous scores...
feverous score: 0.34220532319391633
label accuracy: 0.7070975918884664
```


### Results For Roberta+Pasta

```bash
Feverous scores...
feverous score: 0.31609632446134345
label accuracy: 0.6698352344740177

```


### Results For Bart+Pasta

```bash
Feverous scores...
feverous score: 0.3453738910012674
label accuracy: 0.7022813688212928
```
