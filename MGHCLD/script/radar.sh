# evaluation on our second layers' pathcing performance on existing detectors.
# beter performance: use humanized(attacked) dataset.
# a cotest for fine-tuned deberta on checkgpt_origin dataset,switch mode, data_path, model_dir, detail_result_path for other tests.

python patchradar.py \
    --device_num 1 \
    --mode checkgpt_origin \
    --test_dataset_path MGHCLD/detect_data/checkgpt_origin \
    --test_dataset_name test \
    --detail_result_path MGHCLD/results/detail_result.csv 

echo "Step 2: intermediate data processing..."
python tool/building_id.py --label_file MGHCLD/results/detail_result.csv --text_file MGHCLD/detect_data/checkgpt_origin/test.csv --output_dir MGHCLD/detect_data/middle/


echo "Step 3: subdivision layer processing..."
python test.py \
    --device_num 1 \
    --mode middle \
    --test_dataset_path MGHCLD/detect_data/middle/ \
    --test_dataset_name valid \
    --model_dir MGHCLD/runs/checkgptoriginradar_v0 \
    --detail_result_path MGHCLD/results/second_detail_result.csv 

echo "Step 4: merge final result..."
python tool/merge_id.py --file1 MGHCLD/results/detail_result.csv --file2 MGHCLD/results/second_detail_result.csv --output_file MGHCLD/results/final_result.csv

