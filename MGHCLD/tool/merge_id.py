import pandas as pd
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import compute_metrics


def merge_csv(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Extract the unique records in the first file (id not in the second file)
    unique_in_df1 = df1[~df1['id'].isin(df2['id'])]
    
    # Merge data: all records in the second file + unique records in the first file
    merged_df = pd.concat([df2, unique_in_df1], ignore_index=True)
    
    merged_df.sort_values(by='id', inplace=True)
    
    merged_df.to_csv(output_file, index=False)
    
    eval_df = pd.read_csv(output_file)
    all_ids=eval_df['id'].tolist()
    all_labels=eval_df['true_label'].tolist()
    all_preds=eval_df['pred_label'].tolist()
    all_labels = [str(label) for label in all_labels]
    all_preds = [str(pred) for pred in all_preds]
    all_ids = [str(id) for id in all_ids]
    human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_metrics(
            all_labels, all_preds, all_ids
        )

    print("\nFinal Test Metrics:")
    print(f"HumanRec: {human_rec:.4f}")
    print(f"MachineRec: {machine_rec:.4f}")
    print(f"AvgRec: {avg_rec:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str,default='MGHCLD/results/detail_result.csv',help='The first CSV file.')
    parser.add_argument('--file2', type=str,default='MGHCLD/results/second_detail_result.csv', help='The second CSV file.')
    parser.add_argument('--output_file', type=str, default='MGHCLD/results/final_result.csv', help='The final file.')
    args = parser.parse_args()
    
    merge_csv(args.file1, args.file2, args.output_file)