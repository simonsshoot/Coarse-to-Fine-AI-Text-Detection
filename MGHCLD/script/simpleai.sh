set -e  

# evaluation on our second layers' pathcing performance on existing detectors.
# beter performance: use humanized(attacked) dataset.
# a cotest for fine-tuned deberta on checkgpt_origin dataset,switch mode, data_path, model_dir, detail_result_path for other tests.

echo "Step 1: 第一阶段检测..."
python newtest.py \
    --device_num 1 \
    --mode checkgpt \
    --test_dataset_path MGHCLD/detect_data/checkgpt \
    --test_dataset_name test \
    --detail_result_path MGHCLD/results/detail_result.csv 


echo "Step 2: 执行中间数据处理..."
python tool/building_id.py --label_file MGHCLD/results/detail_result.csv --text_file MGHCLD/detect_data/checkgpt/test.csv --output_dir MGHCLD/detect_data/middle/


echo "Step 3: 第二阶段检测..."
python test.py \
    --device_num 1 \
    --mode middle \
    --test_dataset_path MGHCLD/detect_data/middle/ \
    --test_dataset_name valid \
    --model_dir MGHCLD/runs/checkgptattacksimpleai_v0 \
    --detail_result_path MGHCLD/results/second_detail_result.csv 

echo "Step 4: 合并测试结果..."
python tool/merge_id.py --file1 MGHCLD/results/detail_result.csv --file2 MGHCLD/results/second_detail_result.csv --output_file MGHCLD/results/final_result.csv
