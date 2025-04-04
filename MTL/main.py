import os
import random
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, RobertaModel, AdamW

from sklearn.model_selection import train_test_split

from dataset import load_siqa, load_csqa, load_cmqa, load_piqa, prepare_batch
from dataset import SocialiqaDataset, CommonsenseqaDataset, CosmosqaDataset, PhysicaliqaDataset, MultiTaskDataset
from model import MultitaskModel
from utils import TrainerForMT

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
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--T', type=float, default=1)
    args = parser.parse_args()
    return args

def main(args):
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

	train_siqa, train_siqa_labels = load_siqa('/home/intern/nas', 'train')
	train_csqa, train_csqa_label =load_csqa('/home/intern/nas', 'train')
	train_cmqa, train_cmqa_label =load_cmqa('/home/intern/nas', 'train')
	train_piqa, train_piqa_label =load_piqa('/home/intern/nas', 'train')

	train_siqa, valid_siqa, train_siqa_labels, valid_siqa_labels = train_test_split(train_siqa, train_siqa_labels, train_size=args.training_size, random_state=42)
	train_csqa, valid_csqa, train_csqa_label, valid_csqa_labels = train_test_split(train_csqa, train_csqa_label, train_size=args.training_size, random_state=42)
	train_cmqa, valid_cmqa, train_cmqa_label, valid_cmqa_labels = train_test_split(train_cmqa, train_cmqa_label, train_size=args.training_size, random_state=42)
	train_piqa, valid_piqa, train_piqa_label, valid_piqa_labels = train_test_split(train_piqa, train_piqa_label, train_size=args.training_size, random_state=42)

	train_siqa_dataset = SocialiqaDataset(tokenizer, train_siqa, train_siqa_labels)
	valid_siqa_dataset = SocialiqaDataset(tokenizer, valid_siqa, valid_siqa_labels)

	train_csqa_dataset = CommonsenseqaDataset(tokenizer, train_csqa, train_csqa_label)
	valid_csqa_dataset = CommonsenseqaDataset(tokenizer, valid_csqa, valid_csqa_labels)

	train_cmqa_dataset = CosmosqaDataset(tokenizer, train_cmqa, train_cmqa_label)
	valid_cmqa_dataset = CosmosqaDataset(tokenizer, valid_cmqa, valid_cmqa_labels)

	train_piqa_dataset = PhysicaliqaDataset(tokenizer, train_piqa, train_piqa_label)
	valid_piqa_dataset = PhysicaliqaDataset(tokenizer, valid_piqa, valid_piqa_labels)

	train_siqa_loader = DataLoader(train_siqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))
	valid_siqa_loader = DataLoader(valid_siqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))

	train_csqa_loader = DataLoader(train_csqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))
	valid_csqa_loader = DataLoader(valid_csqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))

	train_cmqa_loader = DataLoader(train_cmqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))
	valid_cmqa_loader = DataLoader(valid_cmqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))

	train_piqa_loader = DataLoader(train_piqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))
	valid_piqa_loader = DataLoader(valid_piqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))

	train_dataloader_list = [train_siqa_loader, train_csqa_loader, train_cmqa_loader, train_piqa_loader]
	valid_dataloader_list = [valid_siqa_loader, valid_csqa_loader, valid_cmqa_loader, valid_piqa_loader]
	# train_dataloader_list = [iter(train_siqa_loader), iter(train_csqa_loader), iter(train_cmqa_loader), iter(train_piqa_loader)]
	# valid_dataloader_list = [iter(valid_siqa_loader), iter(valid_csqa_loader), iter(valid_cmqa_loader), iter(valid_piqa_loader)]

	task_name_list=["siqa","csqa","cmqa","piqa"]
	select_seed=[0,1,2,3]
	roberta_model = RobertaModel.from_pretrained('roberta-large')
	model = MultitaskModel.create(roberta_model, task_name_list)
	optimizer = AdamW(model.parameters(), lr=args.lr)
	trainer = TrainerForMT(model=model, optimizer=optimizer, train_dataloader=train_dataloader_list, valid_dataloader=valid_dataloader_list, task_name_list=task_name_list, select_seed=select_seed,device=device)
	save_dir = os.path.join(args.root_dir, 'models', 'MT')
	trainer.run_training(args, save_dir)


if __name__ == '__main__':
	args = parser_args()
	main(args)
