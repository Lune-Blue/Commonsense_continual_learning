import torch
from torch import nn
from transformers import RobertaModel


class MultitaskModel(nn.Module):
	def __init__(self, roberta_model: RobertaModel, classifier, task_list):
		super(MultitaskModel, self).__init__()
		self.roberta = roberta_model
		self.dropout = nn.Dropout(self.roberta.config.hidden_dropout_prob)
		self.classifier = classifier
		self.task_list = task_list
		self.siqa_classifier = nn.Linear(roberta_model.config.hidden_size, 1)
		self.csqa_classifier = nn.Linear(roberta_model.config.hidden_size, 1)
		self.cmqa_classifier = nn.Linear(roberta_model.config.hidden_size, 1)
		self.piqa_classifier = nn.Linear(roberta_model.config.hidden_size, 1)


	@classmethod
	def create(cls, roberta_model: RobertaModel, task_list):
		classifier={}
		for task in task_list:
			create_classifier = nn.Linear(roberta_model.config.hidden_size, 1)
			classifier[task] = create_classifier
		return cls(roberta_model=roberta_model, classifier=classifier, task_list = task_list)

	def return_classifier(self, task_index):
		if task_index==0:
			return self.siqa_classifier
		elif task_index==1:
			return self.csqa_classifier
		elif task_index==2:
			return self.cmqa_classifier
		elif task_index==3:
			return self.piqa_classifier

	def forward(self, task_index, input_ids: torch.tensor, attention_mask: torch.tensor, labels=None):	
		num_choices = input_ids.shape[1]
		flat_input_ids = input_ids.view(-1, input_ids.size(-1))
		flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

		outputs = self.roberta(
			input_ids = flat_input_ids,
			attention_mask = flat_attention_mask,
		)
		pooled_output = outputs[1]
		pooled_output = self.dropout(pooled_output)

		get_classifier = self.return_classifier(task_index)
		logits = get_classifier(pooled_output)

		# logits = self.classifier[get_task_name](pooled_output)
		reshaped_logits = logits.view(-1, num_choices)

		loss_fct = nn.CrossEntropyLoss()
		loss = loss_fct(reshaped_logits, labels)
		return loss, reshaped_logits

