import torch
import argparse
from torch.utils.data import DataLoader
from utils.dataset_utils import load_MyData
from src.dataset  import PassagesDataset
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score, f1_score


def collate_fn(batch):
    text, label, attack_method, attack_method_set, id = default_collate(batch)
    encoded_batch = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True,
    )
    return encoded_batch, label, attack_method, attack_method_set, id

def test(opt):
    model=AutoModelForSequenceClassification.from_pretrained("/data/Content Moderation/model/roberta-base",num_labels=2)
    checkpoint_path = "saved_model/simpleai_checkgptori.pt"
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {k: v for k, v in state_dict.items() if "roberta.embeddings.position_ids" not in k}
    model.load_state_dict(state_dict,strict=False)
    device="cuda:2"
    print(f"using device:{device}")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("/data/Content Moderation/model/roberta-base")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("loaded model....")

    model.eval()
    if opt.mode in ('checkgpt_origin', 'checkgpt','middle','HC3','HC3origin','seqxgpt_origin','seqxgpt'):
        test_data = load_MyData(opt.test_dataset_path)[opt.test_dataset_name]

    test_dataset = PassagesDataset(test_data, mode=opt.mode, need_ids=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.per_gpu_eval_batch_size,
        num_workers=opt.num_workers,
        collate_fn=collate_fn,
        shuffle=False,  
        drop_last=False
    )

    all_preds = []
    all_labels = []
    all_ids = []
    true_human=0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            encoded_batch, batch_labels, attack_method, attack_method_set, ids = batch
            encoded_batch=encoded_batch.to(device)
            batch_labels=batch_labels.to(device)
            attack_method=attack_method.to(device)
            attack_method_set=attack_method_set.to(device)
            ids=ids.to(device)
            outputs=model(encoded_batch["input_ids"],attention_mask=encoded_batch["attention_mask"])
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[-1] if type(outputs) == list else outputs
            preds = torch.argmax(logits, dim=1)
            preds=1-preds
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_ids.extend(ids.cpu().numpy())
            true_human+=np.sum(batch_labels.cpu().numpy()==1) 
            predict_true_human=np.sum(preds.cpu().numpy()==1)&np.sum(batch_labels.cpu().numpy()==1)

            acc=accuracy_score(all_labels,all_preds)
            human_acc=predict_true_human/true_human if true_human!=0 else 0
            f1=f1_score(all_labels,all_preds,average='macro')
            human_recall=recall_score(all_labels,all_preds,pos_label=1)
            machine_recall=recall_score(all_labels,all_preds,pos_label=0)
            avg_recall=(human_recall+machine_recall)/2.0
        
        print("\nFinal Test Metrics:")
        print(f"HumanRec: {human_recall:.4f}")
        print(f"MachineRec: {machine_recall:.4f}")
        print(f"AvgRec: {avg_recall:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1: {f1:.4f}")
            
        result_df = pd.DataFrame({
            'id': all_ids,
            'true_label': all_labels,
            'pred_label': all_preds
        })
        result_df.to_csv(opt.detail_result_path, index=False)
        print(f"Saved detailed results to {opt.detail_result_path}")
    



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default="/data/Content Moderation/unsup-simcse-roberta-base/", help="Model name")
  parser.add_argument('--test_dataset_path', type=str, default="MGHCLD/detect_data/checkgpt_origin/", help="Test dataset path")
  parser.add_argument('--test_dataset_name', type=str, default="test")
  parser.add_argument('--num_workers', type=int, default=4, help="Number of workers")
  parser.add_argument('--device_num', type=int, default=1, help="Number of devices")
  parser.add_argument('--mode', type=str, default="checkgpt", help="Mode")
  parser.add_argument("--one_loss",action='store_true',help="only use single contrastive loss")
  parser.add_argument("--only_classifier", action='store_true',help="only use classifier, no contrastive loss")
  parser.add_argument("--per_gpu_eval_batch_size", type=int, default=16, help="Batch size per GPU/CPU for evaluation.")
  parser.add_argument("--model_dir",type=str,default="/MGHCLD/runs/simpleai")
  parser.add_argument("--temperature", type=float, default=0.07, help="contrastive loss temperature")
  parser.add_argument('--a', type=float, default=1)
  parser.add_argument('--b', type=float, default=1) 
  parser.add_argument('--c', type=float, default=1)
  parser.add_argument('--d', type=float, default=1,help="classifier loss weight")
  parser.add_argument('--classifier_dim', type=int, default=2,help="classifier out dim")
  parser.add_argument('--projection_size', type=int, default=768, help="Pretrained model output dim")
  parser.add_argument("--resum", type=bool, default=False)
  parser.add_argument('--detail_result_path',type=str,default="MGHCLD/results/detail_result.csv",help="detail result path")
  opt = parser.parse_args()
  opt.model_name="/data/Content Moderation/roberta-base"
  tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
  test(opt)