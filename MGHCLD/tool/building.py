import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
def building(opt):
    label_df = pd.read_csv(opt.label_file)
    selected_ids = label_df[label_df['pred_label'] == 1]['id'].tolist()

    text_df = pd.read_csv(opt.text_file)
    result_df = text_df.iloc[selected_ids].copy()
    

    result_df.insert(0, 'id', selected_ids)  

    result_df.to_csv(os.path.join(opt.output_dir, 'valid.csv'), index=False)

    train_df, test_df = train_test_split(
        result_df,
        test_size=0.2,
        random_state=42
    )


    train_df.to_csv(os.path.join(opt.output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(opt.output_dir, 'test.csv'), index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--label_file', type=str, default='MGHCLD/results/detail_result.csv', help='path to label file')
  parser.add_argument('--text_file', type=str, default='MGHCLD/detect_data/checkgpt/test.csv', help='path to text file')
  parser.add_argument('--output_dir', type=str, default='MGHCLD/detect_data/middle/', help='path to output directory')

  opt = parser.parse_args() 
  building(opt)
