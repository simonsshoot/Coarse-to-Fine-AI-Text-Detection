import csv
import json
import logging

def csv_to_jsonl(csv_file_path, jsonl_file_path):
    logging.basicConfig(filename='nips/conversion_errors.log', level=logging.ERROR)
    
    with open(csv_file_path, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file,delimiter='|')

        
        with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
            for line_num, row in enumerate(csv_reader, start=1):
                try:
                    if row['label']=='human':
                        continue
                    json_line = {
                        "article": (row["sequence"]),  
                        "label": 'human' if row['label'].lower() == 'human' else 'gpt'
                    }
                    jsonl_file.write(json.dumps(json_line, ensure_ascii=False) + '\n')
                except KeyError as e:
                    logging.error(f" {line_num} lines lack: {e}，original data: {row}")
                except Exception as e:
                    logging.error(f"{line_num} lines deal with error: {e}，original data: {row}")


csv_to_jsonl("MGHCLD/detect_data/watermark/gpt2_hc3origintest.csv", 'MGHCLD/detect_data/watermark/hc3gpt.jsonl')