import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from pytorch_transformers.modeling_utils import add_start_docstrings

from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from pytorch_transformers.modeling_roberta import RobertaModel, RobertaClassificationHead, RobertaConfig, \
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

class SpanClassifier(nn.Module):
    """given the span embeddings, classify whether their relations"""
    def __init__(self, d_inp):
        super(SpanClassifier, self).__init__()
        self.d_inp = d_inp
        self.bilinear_layer = nn.Bilinear(d_inp, d_inp, 1)
        self.output = nn.Sigmoid()
        self.loss = BCELoss()


    def forward(self, span_emb_1, span_emb_2, label=None):
        """Calculate the similarity as bilinear product of span embeddings.

        Args:
            span_emb_1: [batch_size, hidden] (Tensor) hidden states for span_1
            span_emb_2: [batch_size, hidden] (Tensor) hidden states for span_2
            label: [batch_size] 0/1 Tensor, if none is supplied do prediction.
        """
        similarity = self.bilinear_layer(span_emb_1, span_emb_2)
        probs = self.output(similarity)
        outputs = (similarity,)
        if label is not None:
            cur_loss = self.loss(probs, label)
            outputs = (cur_loss,) + outputs
        return outputs


class BertForSpanClassification(BertPreTrainedModel):
    """For span classification tasks such as WiC or WSC."""
    def __init__(self, config, task_num_labels, tasks):
        super(BertForSpanClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = SpanClassifier(d_inp=config.hidden_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, span_1, span_2, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, position_ids=None, head_mask=None)
        span1_emb = self._extract_span_emb(outputs[0], span_1)
        span2_emb = self._extract_span_emb(outputs[0], span_2)
        outputs = self.classifier(span1_emb, span2_emb, labels)
        return outputs

    def _extract_span_emb(self, sequence_outputs, span):
        """Extract embeddings for spans, sum up when span is multiple bpe units.

        Args:
            sequence_outputs: [batch_size x max_seq_length x hidden_size] (Tensor) The last layer hidden states for
                all tokens.
            span: list(str). The list of token ids corresponding to the span

        """
        prod = sequence_outputs * span.unsqueeze(-1).float()
        emb_sum = prod.sum(dim=-2)
        return emb_sum



class BertForSequenceClassificationMultiTask(BertPreTrainedModel):
    """
    BERT model for classification with mutiple linear layers for multi-task setup

     Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False
        `task_num_labels`: the number of classes for each classifier
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config, task_num_labels, tasks):
        """
        Initiate BertForSequenceClassificationMultiTask with task informations.

         Params:
            `config`: a BertConfig class instance with the configuration to build a new model
            `task_num_labels`: a dictionary mapping task name to the number of labels for that task
            `tasks`: a list of task names. It has to be consistent with `task_num_labels`
        """
        super(BertForSequenceClassificationMultiTask, self).__init__(config)
        self.task_num_labels = task_num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, task_num_labels[task]) for task in tasks])
        self.id2task = tasks
        self.task2id = {task: i for i, task in enumerate(tasks)}
        self.apply(self.init_weights)

    def forward(self, task_id, input_ids, token_type_ids, attention_mask, labels=None):
        """ one batch can be only one task """
        outputs = self.bert(input_ids, token_type_ids, attention_mask, position_ids=None, head_mask=None)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        classifier = self.classifiers[task_id]
        logits = classifier(pooled_output)

        outputs = (logits,) + outputs[2:]
        num_labels = self.task_num_labels[self.id2task[task_id]]
        if labels is not None:
            if num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaForSequenceClassificationMultiTask(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RoertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, tasks, task_num_labels):
        super(RobertaForSequenceClassificationMultiTask, self).__init__(config)
        self.task_num_labels = task_num_labels

        self.roberta = RobertaModel(config)
        self.classifiers = nn.ModuleList([RobertaClassificationHeadForMTL(config, task_num_labels[task])
                                         for task in tasks])
        self.id2task = tasks
        self.task2id = {task: i for i, task in enumerate(tasks)}

    def forward(self, task_id, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        logits = self.classifiers[task_id](sequence_output)


        outputs = (logits,) + outputs[2:]
        if labels is not None:
            num_labels = self.task_num_labels[self.id2task[task_id]]
            if num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaClassificationHeadForMTL(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)