#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH --job-name=eval-feverous
#SBATCH -p grete
#SBATCH -G A100:1
#SBATCH --mail-user=sdabbag@gwdg.de
#SBATCH --mail-type=all


# shellcheck disable=SC1090
module load anaconda3
module load cuda
source activate mla-2

#conda_setup="/home/smg/$(whoami)/anaconda3/etc/profile.d/conda.sh"
#if [[ -f "${conda_setup}" ]]; then
#  # shellcheck disable=SC1090
#  . "${conda_setup}"
#  conda activate mla-2
#fi

pretrained='claim-verification'
data_dir='../data'
pred_sent_dir="${data_dir}/retrieved_evidence"
max_len=256
max_val_acc=0
latest=""
model_name='deberta-tapas'
dataset_type='dev'



function display_usage() {
    echo "Usage: $0 [--dataset dataset_type] [--model model_name]"
    echo "Options:"
    echo "  --dataset : test or dev data_dir (default: $dataset_type)"
    echo "  --model : Path to save the model (default: $model_name)"
    exit 1
}


while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dataset) dataset_type="$2"; shift;;
        --model) model_name="$2"; shift;;
        --help) display_usage;;
        *) echo "Unknown option: $1"; display_usage;;
    esac
    shift
done




model_dir="${pretrained}-${max_len}-${model_name}-mod"
out_dir="${pretrained}-${max_len}-${model_name}-out"
split_pred="${dataset_type}.combined.not_precomputed.p5.s5.t3.cells"




for file in "${model_dir}/checkpoints"/*.ckpt; do
    echo "file: $file"
    val_acc=$(echo "$file" | grep -oP '(?<=val_acc=)[0-9.]+')

    # Use awk for floating-point comparison
    if (( $(awk -v val_acc="$val_acc" -v max_val_acc="$max_val_acc" 'BEGIN {print (val_acc > max_val_acc)}') )); then
        max_val_acc=$val_acc
        latest=$file
    fi
done


echo "Highest val_acc file: $latest with accuracy: $max_val_acc"


if [[ -z "${latest}" ]]; then
  echo "Cannot find any checkpoint in ${model_dir}"
  exit
fi
mkdir -p "${out_dir}"


if [[ -f "${out_dir}/${split_pred}.jsonl" ]]; then
  echo "Result '${out_dir}/${split_pred}.jsonl' exists!"
  exit
fi

python '../../preprocess_claim_verification.py' \
  --corpus "${data_dir}/corpus.jsonl" \
  --in_file "${pred_sent_dir}/${split_pred}.jsonl" \
  --out_file "${out_dir}/${split_pred}.tsv"

python '../../predict.py' \
  --checkpoint_file "${latest}" \
  --in_file "${out_dir}/${split_pred}.tsv" \
  --out_file "${out_dir}/${split_pred}.prob" \
  --batch_size 4 \
  --gpus 1


if [ "$dataset_type" = "test" ]; then
  python '../../postprocess_claim_verification.py' \
    --pred_sent_file "${pred_sent_dir}/${split_pred}.jsonl" \
    --prob_file "${out_dir}/${split_pred}.prob" \
    --out_file "${out_dir}/${split_pred}.jsonl"\
    --testing

  echo "Result is saved in '${out_dir}/${split_pred}.jsonl'"
  exit

fi


python '../../postprocess_claim_verification.py' \
  --pred_sent_file "${pred_sent_dir}/${split_pred}.jsonl" \
  --prob_file "${out_dir}/${split_pred}.prob" \
  --out_file "${out_dir}/${split_pred}.jsonl"


python '../../eval_feverous.py' \
  --in_file "${out_dir}/${split_pred}.jsonl" \
  --out_file "${out_dir}/eval.${split_pred}.txt"
