import numpy as np
import json
import collections
import torch
from typing import List, Optional, Union, Tuple

# we use the fmeasure class from allennlp, they are copied here so that no
# requirement is needed to install allennlp package
class FBetaMeasure(object):
    """Compute precision, recall, F-measure and support for each class.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.
    If we have precision and recall, the F-beta score is simply:
    ``F-beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)``
    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.
    The support is the number of occurrences of each class in ``y_true``.
    Parameters
    ----------
    beta : ``float``, optional (default = 1.0)
        The strength of recall versus precision in the F-score.
    average : string, [None (default), 'micro', 'macro']
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted mean.
            This does not take label imbalance into account.
    labels: list, optional
        The set of labels to include and their order if ``average is None``.
        Labels present in the data can be excluded, for example to calculate a
        multi-class average ignoring a majority negative class. Labels not present
        in the data will result in 0 components in a macro average.
    """
    def __init__(self,
                 beta: float = 1.0,
                 average: str = None,
                 labels: List[int] = None) -> None:
        assert beta > 0, "`beta` should be >0 in the F-beta score. beta=" + str(beta)

        self._beta = beta
        self._average = average
        self._labels = labels

        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self._true_positive_sum: Union[None, torch.Tensor] = None
        # the total number of instances
        # Shape: (num_classes, )
        self._total_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self._pred_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self._true_sum: Union[None, torch.Tensor] = None


    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_classes = predictions.size(-1)
        assert not (gold_labels >= num_classes).any(), "A gold label passed to FBetaMeasure contains an id >= {num_classes}"
        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes)
            self._true_sum = torch.zeros(num_classes)
            self._pred_sum = torch.zeros(num_classes)
            self._total_sum = torch.zeros(num_classes)

        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.to(dtype=torch.uint8)
        gold_labels = gold_labels.float()

        argmax_predictions = predictions.max(dim=-1)[1].float()

        true_positives = (gold_labels == argmax_predictions) * mask
        true_positives_bins = gold_labels[true_positives]

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes)
        else:
            true_positive_sum = torch.bincount(true_positives_bins.long(), minlength=num_classes).float()

        pred_bins = argmax_predictions[mask].long()
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes)

        gold_labels_bins = gold_labels[mask].long()
        if gold_labels.shape[0] != 0:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum
        self._total_sum += mask.sum().to(torch.float)


    def get_metric(self,
                   reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precisions : List[float]
        recalls : List[float]
        f1-measures : List[float]
        If ``self.average`` is not ``None``, you will get ``float`` instead of ``List[float]``.
        """
        if self._true_positive_sum is None:
            raise RuntimeError("You never call this metric before.")

        tp_sum = self._true_positive_sum
        pred_sum = self._pred_sum
        true_sum = self._true_sum

        if self._average == 'micro':
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()
            true_sum = true_sum.sum()

        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        precision = _prf_divide(tp_sum, pred_sum)
        recall = _prf_divide(tp_sum, true_sum)
        fscore = ((1 + beta2) * precision * recall /
                  (beta2 * precision + recall))
        fscore[tp_sum == 0] = 0.0

        if self._average == 'macro':
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()

        if reset:
            self.reset()

        if self._labels is not None:
            # Retain only selected labels and order them
            precision = precision[self._labels]
            recall = recall[self._labels]
            fscore = fscore[self._labels]

        if self._average is None:
            return {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "fscore": fscore.tolist()
            }
        else:
            return {
                    "precision": precision.item(),
                    "recall": recall.item(),
                    "fscore": fscore.item()
            }


    def reset(self) -> None:
        self._true_positive_sum = None
        self._pred_sum = None
        self._true_sum = None
        self._total_sum = None

    @property
    def _true_negative_sum(self):
        if self._total_sum is None:
            return None
        else:
            true_negative_sum = self._total_sum - self._pred_sum - self._true_sum + self._true_positive_sum
            return true_negative_sum

    @staticmethod
    def unwrap_to_tensors(*tensors: torch.Tensor):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly and that they are on
        the CPU.
        """
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)

class F1Measure(FBetaMeasure):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, positive_label: int) -> None:
        super().__init__(beta=1,
                         labels=[positive_label])

    def get_metric(self,
                   reset: bool = False) -> Tuple[float, float, float]:
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        metric = super().get_metric(reset=reset)
        # Because we just care about the class `positive_label`
        # there is just one item in `precision`, `recall`, `fscore`
        precision = metric['precision'][0]
        recall = metric['recall'][0]
        fscore = metric['fscore'][0]
        return precision, recall, fscore

    @property
    def _true_positives(self):
        # When this metric is never called, `self._true_positive_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_positive_sum is None:
            return 0.0
        else:
            # Because we just care about the class `positive_label`,
            # there is just one item in `self._true_positive_sum`.
            return self._true_positive_sum[0]

    @property
    def _true_negatives(self):
        # When this metric is never called, `self._true_negative_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_negative_sum is None:
            return 0.0
        else:
            # Because we just care about the class `positive_label`,
            # there is just one item in `self._true_negative_sum`.
            return self._true_negative_sum[0]

    @property
    def _false_positives(self):
        # When this metric is never called, `self._pred_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._pred_sum is None:
            return 0.0
        else:
            # `self._pred_sum` is the total number of instances under each _predicted_ class,
            # including true positives and false positives.
            return self._pred_sum[0] - self._true_positives

    @property
    def _false_negatives(self):
        # When this metric is never called, `self._true_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_sum is None:
            return 0.0
        else:
            # `self._true_sum` is the total number of instances under each _true_ class,
            # including true positives and false negatives.
            return self._true_sum[0] - self._true_positives

def _prf_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.
    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result

def calculate_single_sequence_task_metrics(predictions, labels):
    """Calculate accuracy for single sequence task evaluation.

    Args:
        predictions: a list of integers, each corresponding to a predicted label
        labels: a list of integers, goldstandard labels.

    Returns:
        accuracy: the percentage of correct prediction
        # f1: f1 score, computed as 2 * (prec * rec) / (prec + rec)
    """
    correctness = [1 if i == j else 0 for i, j in zip(predictions, labels)]
    accuracy = np.mean(correctness)
    return accuracy

def calculate_multiple_choice_task_metrics(pred_dict, labels_dict):
    """Calculate accuracy for multiple choice tasks.

    Args:
        pred_dict: mapping subinstance ids to prediction, where subinstance id is like "0_0", which stands for the 0-th
            option for the 0-th instance
        labels_dict: mapping subinstance ids to labels
    Return:
        accuracy: measuring the percentage of correct predicted instances
    """

    assert len(pred_dict) == len(labels_dict)

    instance = dict()
    for sub_id in pred_dict:
        ins_id, choice_id = sub_id.split("_")
        prob = pred_dict[sub_id]

        if not ins_id in instance:
            instance[ins_id] = (choice_id, prob)
        elif prob > instance[ins_id][1]:
            # whenever the new option has a higher probability, replace the choice
            instance[ins_id] = (choice_id, prob)

    correct_count = 0
    for sub_id in labels_dict:
        ins_id, choice_id = sub_id.split("_")
        label = int(labels_dict[sub_id])
        if label == 1 and choice_id == instance[ins_id][0]:
            correct_count += 1

    return correct_count / len(instance)


def evaluate_multirc(fpath):
    """
    Demo code to run evaluation on MultiRC, requires three fields to be 
    presented in the output jsonl file:
        `text`: a list of three strings: question, answer, passage
        `prob`: the probability [prob(false), prob(true)]
        `label`: the gold-standard label (0 or 1)
    Usage:
        evaluate_multirc(fpath="prediction_MultiRC_epoch_-1.jsonl")
    """
    scorer1 = F1Measure(positive_label=1)
    scorer2 = F1Measure(positive_label=1)
    score_tracker = collections.defaultdict(list)

    for ln in open(fpath):
        cur_obj = json.loads(ln)
        tokens = cur_obj["text"]
        question, answer, passage = tokens

        logits = cur_obj["prob"]
        label = cur_obj["label"]

        logits = torch.tensor([logits])
        label = torch.tensor([label])

        scorer1(logits, label)
        score_tracker[(passage, question)].append((logits, label))

    _, _, ans_f1 = scorer1.get_metric()

    ems, f1s = [], []
    for logits_and_labels in score_tracker.values():
        logits, labels = list(zip(*logits_and_labels))
        logits = torch.stack(logits)
        labels = torch.stack(labels)

        scorer2(logits, labels)
        _, _, ex1 = scorer2.get_metric(reset=True)

        f1s.append(ex1)

        preds = logits.argmax(dim=-1)
        ex_em = (torch.eq(preds, labels).sum() == len(preds)).item()
        ems.append(ex_em)
    em = sum(ems) / len(ems)
    qst_f1 = sum(f1s) / len(f1s)

    print("answer f1:", ans_f1)
    print("question f1:", qst_f1)
    print("EM: ", em)
    print("avg (of EM and answer f1):", (ans_f1 + em) / 2)
    return ans_f1, qst_f1, em


TASK_TO_EVALUATOR = {
    "BoolQ": calculate_single_sequence_task_metrics,
    "BoolQNLI": calculate_single_sequence_task_metrics,
    "SQuADNLI": calculate_single_sequence_task_metrics,
    "CB": calculate_single_sequence_task_metrics,
    "RTE": calculate_single_sequence_task_metrics,
    "COPA": calculate_multiple_choice_task_metrics,
    "WSC": calculate_single_sequence_task_metrics,
    "WiC": calculate_single_sequence_task_metrics,
}
