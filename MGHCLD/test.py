import torch
import argparse
from torch.utils.data import DataLoader
from src.text_embedding import TextEmbeddingModel
from utils.dataset_utils import load_MyData
from src.dataset  import PassagesDataset
from lightning import Fabric
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer
from utils.utils import compute_metrics
from lightning.fabric.strategies import DDPStrategy
from src.models import First_Classifier,Second_Classifier
import pandas as pd
import os



def collate_fn(batch):
    text, label, attack_method, attack_method_set, id = default_collate(batch)
    encoded_batch = tokenizer.batch_encode_plus(
        text,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True,
    )
    return encoded_batch, label, attack_method, attack_method_set, id

def test(opt):
    torch.set_float32_matmul_precision("medium")
    if opt.device_num > 1:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)
        fabric = Fabric(accelerator="cuda", 
                       precision="bf16-mixed",
                       devices=opt.device_num,
                       strategy=ddp_strategy)
    else:
        fabric = Fabric(accelerator="cuda",
                       precision="bf16-mixed",
                       devices=opt.device_num)
    fabric.launch()

    if opt.only_classifier:
        opt.a=opt.b=opt.c=0
        opt.d=1
        opt.one_loss=True

    if opt.one_loss:
        model = First_Classifier(opt, fabric)
    else:
        model = Second_Classifier(opt, fabric)

    checkpoint_path = os.path.join(opt.model_dir, 'model_classifier_best.pth')
    state_dict = torch.load(checkpoint_path, map_location=fabric.device)
    print(f"loading model from {checkpoint_path}")
    
    # handle weight prefixes saved for multi-GPU training
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"Missing keys: {missing}")  # should be an empty list
    print(f"Unexpected keys: {unexpected}")  # should be an empty list

    
    model.eval()
    model = fabric.setup(model)
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
    test_loader = fabric.setup_dataloaders(test_loader)

    all_preds = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            encoded_batch, labels, attack_method, attack_method_set, ids = batch
            
            _, outputs, _, _ = model(encoded_batch, attack_method, attack_method_set, labels, ids)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(fabric.all_gather(preds).cpu().numpy())
            all_labels.extend(fabric.all_gather(labels).cpu().numpy())
            all_ids.extend(fabric.all_gather(ids).cpu().numpy())

    # the main process saves the results
    if fabric.global_rank == 0:
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

        # save detailed results
        result_df = pd.DataFrame({
            'id': all_ids,
            'true_label': all_labels,
            'pred_label': all_preds
        })
        result_df.to_csv(opt.detail_result_path, index=False)
        print(f"Saved detailed results to {opt.detail_result_path}")
    



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default="unsup-simcse-roberta-base/", help="Model name")
  parser.add_argument('--test_dataset_path', type=str, default="/detect_data/checkgpt_origin/", help="Test dataset path")
  parser.add_argument('--test_dataset_name', type=str, default="test")
  parser.add_argument('--num_workers', type=int, default=4, help="Number of workers")
  parser.add_argument('--device_num', type=int, default=1, help="Number of devices")
  parser.add_argument('--mode', type=str, default="checkgpt_origin", help="Mode")
  parser.add_argument("--one_loss",action='store_true',help="only use single contrastive loss")
  parser.add_argument("--only_classifier", action='store_true',help="only use classifier, no contrastive loss")
  parser.add_argument("--per_gpu_eval_batch_size", type=int, default=16, help="Batch size per GPU/CPU for evaluation.")
  parser.add_argument("--model_dir",type=str,default="runs/your_test_model")
  parser.add_argument("--temperature", type=float, default=0.07, help="contrastive loss temperature")
  parser.add_argument('--a', type=float, default=1)
  parser.add_argument('--b', type=float, default=1) 
  parser.add_argument('--c', type=float, default=1)
  parser.add_argument('--d', type=float, default=1,help="classifier loss weight")
  parser.add_argument('--classifier_dim', type=int, default=2,help="classifier out dim")
  parser.add_argument('--projection_size', type=int, default=768, help="Pretrained model output dim")
  parser.add_argument("--resum", type=bool, default=False)
  parser.add_argument('--detail_result_path',type=str,default="results/detail_result.csv",help="detail result path")
  opt = parser.parse_args()
  tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
  test(opt)