#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --out="preprocessing-logs.txt"
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -t 1-00:00:00
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

set -ex

evid_dir='retrieved_evidence'

if [[ ! -f 'corpus.jsonl' ]]; then
  python '../../preprocess_corpus.py' \
    --db_file 'feverous_wikiv1.db' \
    --in_file_gold "${evid_dir}/train.jsonl" \
    --out_file "tmp_train.jsonl"; \
  python '../../preprocess_corpus.py' \
  --db_file 'feverous_wikiv1.db' \
  --in_file_gold "${evid_dir}/dev.jsonl" \
  --out_file "tmp_dev.jsonl"; \
  for split in 'train.pages.p5' 'test.pages.p5' 'dev.pages.p5'; do
    python '../../preprocess_corpus.py' \
      --db_file 'feverous_wikiv1.db' \
      --in_file "${evid_dir}/${split}.jsonl" \
      --in_file_gold "${evid_dir}/train.jsonl" \
      --out_file "tmp_${split}.jsonl"
  done
  cat tmp_*.jsonl | sort | uniq > 'corpus.jsonl'
  rm -f tmp_*.jsonl
  wc -l 'corpus.jsonl'
fi
