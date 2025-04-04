import os
import random
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from dataset import load_siqa, load_csqa, load_cmqa, load_piqa, prepare_batch
from dataset import SocialiqaDataset, CommonsenseqaDataset, CosmosqaDataset, PhysicaliqaDataset, MultiTaskDataset
from model import MultitaskModel
from utils import get_best_model, test_mtl

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cur_dir', type=str, default='/home/intern/seungjun/commonsense/MTL')
    parser.add_argument('--root_dir', type=str, default='/home/intern/nas')

    parser.add_argument('--lm', type=str, default='roberta-large', choices=['roberta-large', 'roberta-cskg'], help='Pre-trained LM or KG fine-tuned LM.')
    parser.add_argument('--training_size', type=int, required=True, help='Number of samples to use for training LM.')

    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    return args

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    dev_siqa, dev_siqa_labels = load_siqa(args.root_dir, 'dev')
    dev_csqa, dev_csqa_label =load_csqa(args.root_dir, 'dev')
    dev_cmqa, dev_cmqa_label =load_cmqa(args.root_dir, 'dev')
    dev_piqa, dev_piqa_label =load_piqa(args.root_dir, 'dev')


    dev_siqa_dataset = SocialiqaDataset(tokenizer, dev_siqa, dev_siqa_labels)
    dev_csqa_dataset = CommonsenseqaDataset(tokenizer, dev_csqa, dev_csqa_label)
    dev_cmqa_dataset = CosmosqaDataset(tokenizer, dev_cmqa, dev_cmqa_label)
    dev_piqa_dataset = PhysicaliqaDataset(tokenizer, dev_piqa, dev_piqa_label)

    dev_siqa_loader = DataLoader(dev_siqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))
    dev_csqa_loader = DataLoader(dev_csqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))
    dev_cmqa_loader = DataLoader(dev_cmqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))
    dev_piqa_loader = DataLoader(dev_piqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))

    test_dataloader_list = [dev_siqa_loader, dev_csqa_loader, dev_cmqa_loader, dev_piqa_loader]
    # train_dataloader_list = [iter(dev_siqa_loader), iter(dev_csqa_loader), iter(dev_cmqa_loader), iter(dev_piqa_loader)]
    # valid_dataloader_list = [iter(valid_siqa_loader), iter(valid_csqa_loader), iter(valid_cmqa_loader), iter(valid_piqa_loader)]

    task_name_list=["siqa","csqa","cmqa","piqa"]
    select_seed=[0,1,2,3]
    roberta_model = RobertaModel.from_pretrained('roberta-large')
    model = MultitaskModel.create(roberta_model, task_name_list)

    path = os.path.join(args.root_dir, 'models', 'MT')
    condition = f'{args.lm}-ts{args.training_size}-bs{args.batch_size}'
    best_name, best_acc = get_best_model(path, condition)
    best_path = os.path.join(path, best_name)
    restore_dict = torch.load(best_path, map_location=device)
    model.load_state_dict(restore_dict)
    print('Best model:', best_name)

    _, labels, preds = test_mtl(model, select_seed, test_dataloader_list, device)

    print(labels[:10])
    print("======")
    print(preds[:10])
    print(accuracy_score(labels, preds))
    with open(os.path.join(args.root_dir, 'results','seungjun', 'mtl', f'{best_name[:-2]}txt'), 'w') as f:
        f.write(f'{accuracy_score(labels, preds)}\n')

    
if __name__ == '__main__':
	args = parser_args()
	main(args)
