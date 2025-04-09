# separate test, from cotest.sh

python test.py \
    --device_num 1 \
    --mode checkgpt_origin \
    --test_dataset_path MGHCLD/detect_data/checkgpt_origin \
    --test_dataset_name test \
    --only_classifier --model_dir MGHCLD/runs/checkgptoriginfirstdeberta_v0 --detail_result_path MGHCLD/results/detail_result.csv --model_name models/deberta-v3-base



