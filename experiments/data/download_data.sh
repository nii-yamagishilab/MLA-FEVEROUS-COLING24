#!/bin/bash

evid_dir='retrieved_evidence'
if [[ ! -d "${evid_dir}" ]]; then
  mkdir -p "${evid_dir}"
fi

base_url='https://fever.ai/download/feverous'
wget -O "${evid_dir}/train.jsonl" "${base_url}/feverous_train_challenges.jsonl"
wget -O "${evid_dir}/dev.jsonl" "${base_url}/feverous_dev_challenges.jsonl"
wget -O "${evid_dir}/test.jsonl" "${base_url}/feverous_test_unlabeled.jsonl"
wget -O "${evid_dir}/feverous-wiki-pages-db.zip" "${base_url}/feverous-wiki-pages-db.zip"
unzip "${evid_dir}/feverous-wiki-pages-db.zip"
