#!/bin/bash
#SBATCH --job-name=training-feverous
#SBATCH -t 1-00:00:00
#SBATCH -p grete
#SBATCH -G A100:2
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


echo "${HOSTNAME}"
task='claim-verification'
max_len=256
data_dir='../data'
attn_bias_type='none'


# Default values in case no arguments are provided
pretrained_text='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
pretrained_tab='google/tapas-large-finetuned-tabfact'
model_name='deberta-tapas'

function display_usage() {
    echo "Usage: $0 [--text pretrained_text] [--table pretrained_tab] [--model model_name]"
    echo "Options:"
    echo "  --text : Pretrained text model (default: $pretrained_text)"
    echo "  --table : Pretrained table model (default: $pretrained_tab)"
    echo "  --model : Path to save the model (default: $model_name)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --text) pretrained_text="$2"; shift;;
        --table) pretrained_tab="$2"; shift;;
        --model) model_name="$2"; shift;;
        --help) display_usage;;
        *) echo "Unknown option: $1"; display_usage;;
    esac
    shift
done

model_dir="${task}-${max_len}-${model_name}-mod"
inp_dir="${task}-${max_len}-${model_name}-inp"


if [[ -d "${model_dir}" ]]; then
  echo "${model_dir} exists! Skip training."
  exit
fi

mkdir -p "${inp_dir}"

python '../../preprocess_claim_verification.py' \
  --corpus "${data_dir}/corpus.jsonl" \
  --in_file "${data_dir}/retrieved_evidence/train.combined.not_precomputed.p5.s5.t3.cells.jsonl" \
  --out_file "${inp_dir}/train.tsv" \
  --training


python '../../preprocess_claim_verification.py' \
  --corpus "${data_dir}/corpus.jsonl" \
  --in_file "${data_dir}/retrieved_evidence/dev.combined.not_precomputed.p5.s5.t3.cells.jsonl" \
  --out_file "${inp_dir}/dev.tsv" \
  --training


python '../../train.py' \
  --task "${task}" \
  --data_dir "${inp_dir}" \
  --default_root_dir "${model_dir}" \
  --pretrained_model_name_text "${pretrained_text}" \
  --pretrained_model_name_table "${pretrained_tab}" \
  --max_seq_length "${max_len}" \
  --attn_bias_type "${attn_bias_type}" \
  --max_epochs 10 \
  --cache_dir "${inp_dir}" \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --accumulate_grad_batches 2 \
  --learning_rate 1e-5\
  --precision 16 \
  --gradient_clip_val 1.0 \
  --warmup_ratio 0.02 \
  --num_workers 8 \
  --use_wandb \
  --save_all_checkpoints \
  --deterministic false\
  --no_init_text bert_text\
  --no_init_table bert_tab\
  --do_eval \
  --gpus 2
