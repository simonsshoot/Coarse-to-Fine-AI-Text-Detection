import pandas as pd
import random
import tqdm
import re
import numpy as np
import os
import json

method_name_dct = {
    'char':['spaceremoval','insertspace','wordcase_pct50','wordmerge','punctuationremoval','punctuationappend'],
    'word':['typos','spellingerror','insertadv','mlmword'],
    'sent':['appendirr','backtransent','repeatsent','mlmsent'],
    'para':['reverse','backtran','paraphrase'],
    'human':['human'],
    'machine_origin':['machine_origin']
}
method_set={'char':0,'word':1,'sent':2,'para':3,'human':4,'machine_origin':5}
    
def trim_quotes(s):
    return s.strip("\"'")

'''Used to process spaces and punctuation in text to make it more standardized'''
def process_spaces(text):
    text=text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()
    return trim_quotes(text)

def load_MyData(file_folder=None):
    data={
        'train':[],
        'test':[],
        'valid':[]
    }
    folder=os.listdir(file_folder)
    for now in folder:
        if now[-3:]!='csv':
            continue
        full_path=os.path.join(file_folder,now)
        keyname=now.split('.')[0]
        assert keyname in data.keys(), f'{keyname} is not in data.keys()'
        now_data=pd.read_csv(full_path,on_bad_lines='skip')
        for i in range(len(now_data)):
            id=now_data.iloc[i]['id']
            text,src=now_data.iloc[i]['Generation'],now_data.iloc[i]['label']
            label= '1' if src=='human' else '0'
            data[keyname].append((process_spaces(str(text)),label,src,id))
    
    return data
