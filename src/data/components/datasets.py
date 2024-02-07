import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
import numpy as np
from collections.abc import Iterable
from typing import List, Tuple, Union
import numbers
from typing import List, Tuple
from omegaconf import DictConfig, OmegaConf
from transformers import BertTokenizer, GPT2Tokenizer
from tqdm import tqdm
from typing import List
import re

from src.utils.text_processing import python_remove_punctuation

# default separators for the tokenization
SEP = ["-", ".", ",", ";"]


def assign_labels(input_string, labels):
    # Create list to hold words and punctuation
    words_with_punctuation = re.findall(r"[\w']+|[.,!?;\"-]|'", input_string)

    # Create list to hold only words
    words_only = re.findall(r"\w+'?\w*", input_string)

    # Make sure the number of labels matches the number of words
    if not len(labels) == len(words_only):
        # print(
        #     f"Aligning labels: Number of labels ({len(labels)}) does not match number of words ({len(words_only)})"
        # )
        # alignmend or extraction failed, skip sample
        return None, None, None

    # Create a generator for word-label pairs
    word_label_pairs = ((word, label) for word, label in zip(words_only, labels))

    # Create list of tuples where each word is matched to a label and each punctuation is matched to None
    words_with_labels = []
    for token in words_with_punctuation:
        if re.match(r"\w+'?\w*", token):
            words_with_labels.append(next(word_label_pairs))
        else:
            words_with_labels.append((token, None))

    return words_only, words_with_punctuation, words_with_labels


def tokenize_text_with_labels(
    text: List[str],
    # aligned_words: List[str],
    labels: Union[List[float], List[List[float]]],
    tokenizer,
    model_type,
    invalid_label: int = -999,
    score_first_token: bool = False,
    score_last_token: bool = False,
    relative_to_prev: bool = False,
    remove_punctuation: bool = False,
    n_prev: int = 1,
    relative_to_mean=False,
    word_stats: dict = None,
    add_prefix_space: bool = True,
    min_words=4,
    max_words=60,
):
    """
    Tokenize the input text and associate labels with each token.

    Args:
        text (str): The input text to tokenize.
        labels (list): A list of labels corresponding to each word in the text.
        model_type (str): The type of the language model (e.g., 'gpt2', 'bert-uncased', 'bert-cased').
        invalid_label (int, optional): The label to assign to invalid tokens (e.g., punctuation and whitespace). Defaults to -1.
        score_first_token (bool, optional): If True, only the first token of a multi-token word will have a mask value of 1. Defaults to True.
        relative_to_prev (bool, optional): If True, adjust the labels to be relative to the average of the previous n_prev words. Defaults to False.
        n_prev (int, optional): The number of previous words to consider when adjusting labels. Only used if relative_to_prev is True. Defaults to 1.
        relative_to_mean (bool, optional): If True, adjust the labels to be relative to the mean of the word in the corpus passed by the dict. Defaults to False.
        word_stats (dict, optional): A dictionary containing word statistics and as such the mean prominence of each word in the corpus. Defaults to None.

    Returns:
        tuple: A tuple containing the following elements:
            - input_text (str): The input text.
            - tokenized_text (list): The tokenized text as a list of tokens.
            - tokenized_labels (list): The list of labels corresponding to each token in the tokenized_text.
            - token_ids (list): The list of token IDs corresponding to each token in the tokenized_text.
            - mask (list): A binary mask indicating whether a token should be scored (1) or not (0).
    """
    assert not (
        score_first_token and score_last_token
    ), "Only one of score_first_token and score_last_token can be True"

    # remove punctuation if specified
    if remove_punctuation:
        text = python_remove_punctuation(text)

    # check if we have vector-valued labels and if so, adapt the invalid label to it
    if isinstance(labels[0], Iterable) and not isinstance(labels[0], str):
        invalid_label = [invalid_label] * len(labels[0])

    # remove None labels
    labels = [l for l in labels if l is not None]
    # print(f"Text before : {text}")
    # print(f"Labels: {labels}")
    # print(f"Length labels: {len(labels)}")

    _, _, labeled_tokens = assign_labels(text, labels)
    # print(f"Labeled tokens: {labeled_tokens}")
    # check if label assignment is possible
    if labeled_tokens is None:
        return None

    # apply cleaning on the number of words to remove outliers
    if len(labeled_tokens) < min_words or len(labeled_tokens) > max_words:
        # print(f"Text: {text}")
        return None

    word_units, labels = zip(*labeled_tokens)
    words = list(word_units)
    labels = list(labels)
    original_labels = labels  # store them to return them later

    # from here on we assume that each word is a unit that has a (potentially None) label
    assert len(words) == len(labels), "The number of words and labels should be equal"

    # if relative_to_prev is True, we adjust the labels to be relative to the average of the previous n_prev words
    if relative_to_prev and not relative_to_mean:
        new_labels = []
        # print(f"labels before: {labels}")
        for i, label in enumerate(labels):
            if i < n_prev or label is None:
                new_labels.append(label)
            else:
                # Get the previous n_prev labels which are not None
                prev_labels = [
                    labels[j] for j in range(i - n_prev, i) if labels[j] is not None
                ]
                # print(f"i = {i}, label {label}, prev_labels = {prev_labels}")
                if prev_labels:
                    avg_prev = sum(prev_labels) / len(prev_labels)
                    new_labels.append(label - avg_prev)
                else:
                    new_labels.append(label)
        labels = new_labels
        # print(f"labels after: {labels}")

    # if relative_to_mean is True, we adjust the labels to be relative to the mean of the word in the corpus
    elif relative_to_mean:
        if word_stats is None:
            raise ValueError(
                "Word statistics are required for relative_to_mean method."
            )
        new_labels = []
        for word, label in zip(words, labels):
            if label is None:
                new_labels.append(label)
                continue

            if word in word_stats:
                mean_label = word_stats[word]["mean"]
            elif word.lower() in word_stats:
                mean_label = word_stats[word.lower()]["mean"]
            else:
                mean_label = word_stats["$global$"]["mean"]
            new_labels.append(label - mean_label)
        labels = new_labels

    tokenized_text, tokenized_labels, token_ids, mask, word_to_tokens = (
        [],
        [],
        [],
        [],
        [],
    )

    # if model is Bert we add a [CLS] token at the beginning
    if model_type.lower().startswith("bert"):
        tokenized_text.append(tokenizer.cls_token)
        tokenized_labels.append(invalid_label)
        token_ids.append(tokenizer.cls_token_id)
        mask.append(0)

    # we tokenize each word separately and keep track of the mapping between words and tokens
    for i, (word, label) in enumerate(zip(words, labels)):
        # TODO: remove this hardcoded hack for gpt, must be similar for llama
        if (
            "gpt" in model_type.lower()
            and i > 0
            and not np.any([s in word for s in SEP])  # check for punctuation
        ):
            word = " " + word
        # else:
        # print("word: ", word)
        # print(f"model name {model_type}")
        # print(f"i: {i}")
        # print(f"SEP: {SEP}")

        tokens = tokenizer.tokenize(word)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        tokenized_text.extend(tokens)
        token_ids.extend(ids)
        word_to_tokens.extend((word, ids))

        if score_first_token:
            mask.extend([1] + [0] * (len(tokens) - 1))
            tokenized_labels.extend([label] + [invalid_label] * (len(tokens) - 1))
        elif score_last_token:
            mask.extend([0] * (len(tokens) - 1) + [1])
            tokenized_labels.extend([invalid_label] * (len(tokens) - 1) + [label])
        else:
            mask.extend([1] * len(tokens))
            tokenized_labels.extend([label] * len(tokens))

    # if model is BERT we add a [SEP] token at the end
    if model_type.lower().startswith("bert"):
        tokenized_text.append(tokenizer.sep_token)
        tokenized_labels.append(invalid_label)
        token_ids.append(tokenizer.sep_token_id)
        mask.append(0)

    # substitute all None in labels (could not compute a label here) with invalid_label as well as set mask to 0 at these positions
    tokenized_labels = [
        label if label is not None else invalid_label for label in tokenized_labels
    ]
    # mask = [1 if label != invalid_label else 0 for label in tokenized_labels]
    mask = [1 if np.all(label != invalid_label) else 0 for label in tokenized_labels]

    # if mask is all 0 (no valid predicitons) we return None
    if np.all(mask == 0):
        return None

    return (
        text,
        tokenized_text,
        original_labels,
        tokenized_labels,
        token_ids,
        mask,
        word_to_tokens,
    )


class TokenTaggingDataset(Dataset):
    def __init__(
        self,
        input_texts,
        targets,
        tokenizer,
        model_name: str,
        score_first_token: bool = False,
        score_last_token: bool = False,
        relative_to_prev: bool = False,
        remove_punctuation: bool = False,
        n_prev: int = 1,
        relative_to_mean=False,
        word_stats: dict = None,
        debug: bool = False,
    ):
        """
        ::param inputs: list of strings
        ::param targets: list of lists of labels
        ::param model_name: name of the model to use
        ::param tokenizer: tokenizer object
        ::param score_first_token: whether to score only the first token of a word
        ::param relative_to_prev: whether to score relative to the previous token
        ::param n_prev: number of previous tokens to consider
        """
        self.inputs = input_texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.score_first_token = score_first_token
        self.score_last_token = score_last_token
        self.relative_to_prev = relative_to_prev
        self.remove_punctuation = remove_punctuation
        self.n_prev = n_prev
        self.relative_to_mean = relative_to_mean
        self.word_stats = word_stats
        self.debug = debug

        cnt_failed = 0
        # Perform preprocessing at initialization
        self.processed_data = []
        for text, labels_per_word in tqdm(
            zip(self.inputs, self.targets),
            total=len(self.inputs),
            desc="Preprocessing samples",
        ):
            result = tokenize_text_with_labels(
                text=text,
                labels=labels_per_word,
                tokenizer=self.tokenizer,
                model_type=self.model_name,
                score_first_token=self.score_first_token,
                score_last_token=self.score_last_token,
                relative_to_prev=self.relative_to_prev,
                remove_punctuation=self.remove_punctuation,
                n_prev=self.n_prev,
                relative_to_mean=self.relative_to_mean,
                word_stats=self.word_stats,
            )

            if not result:
                cnt_failed += 1
                continue

            (
                input_text,
                tokenized_text,
                original_labels,
                tokenized_labels,
                token_ids,
                mask,
                word_to_tokens,
            ) = result

            self.processed_data.append(
                {
                    "input_text": input_text,
                    "tokenized_text": tokenized_text,
                    "original_labels": original_labels,
                    "tokenized_labels": tokenized_labels,
                    "input_ids": token_ids,
                    "loss_mask": mask,
                    "word_to_tokens": word_to_tokens,
                }
            )

        print(f"Failed {cnt_failed}/{len(self.inputs)}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        if self.debug:
            print("---")
            print("input_text", item["input_text"])
            print("tokenized_text", item["tokenized_text"])
            print("original_labels", item["original_labels"])
            print("tokenized_labels", item["tokenized_labels"])
            print("input_ids", item["input_ids"])
            print("loss_mask", item["loss_mask"])
            print("word_to_tokens", item["word_to_tokens"])

        return item


# BACKUP
# class TokenTaggingDataset(Dataset):
#     def __init__(
#         self,
#         input_texts,
#         targets,
#         tokenizer,
#         model_name: str,
#         score_first_token: bool = False,
#         score_last_token: bool = False,
#         relative_to_prev: bool = False,
#         n_prev: int = 1,
#         relative_to_mean=False,
#         word_stats: dict = None,
#         debug: bool = False,
#     ):
#         """
#         ::param inputs: list of strings
#         ::param targets: list of lists of labels
#         ::param model_name: name of the model to use
#         ::param tokenizer: tokenizer object
#         ::param score_first_token: whether to score only the first token of a word
#         ::param relative_to_prev: whether to score relative to the previous token
#         ::param n_prev: number of previous tokens to consider
#         """
#         self.inputs = input_texts
#         self.targets = targets
#         self.tokenizer = tokenizer
#         self.model_name = model_name
#         self.score_first_token = score_first_token
#         self.score_last_token = score_last_token
#         self.relative_to_prev = relative_to_prev
#         self.n_prev = n_prev
#         self.relative_to_mean = relative_to_mean
#         self.word_stats = word_stats
#         self.debug = debug

#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, idx):
#         text = self.inputs[idx]
#         labels_per_word = self.targets[idx]

#         (
#             input_text,
#             tokenized_text,
#             original_labels,
#             tokenized_labels,
#             token_ids,
#             mask,
#             word_to_tokens,
#         ) = tokenize_text_with_labels(
#             text=text,
#             labels=labels_per_word,
#             tokenizer=self.tokenizer,
#             model_type=self.model_name,
#             score_first_token=self.score_first_token,
#             score_last_token=self.score_last_token,
#             relative_to_prev=self.relative_to_prev,
#             n_prev=self.n_prev,
#             relative_to_mean=self.relative_to_mean,
#             word_stats=self.word_stats,
#         )

#         if self.debug:
#             print("---")
#             print("input_text", input_text)
#             print("tokenized_text", tokenized_text)
#             print("original_labels", original_labels)
#             print("tokenized_labels", tokenized_labels)
#             print("input_ids", token_ids)
#             print("loss_mask", mask)
#             print("word_to_tokens", word_to_tokens)

#         assert len(tokenized_labels) == len(
#             token_ids
#         ), f"tokens and labels have different length: {len(tokenized_labels)} != {len(token_ids)}"

#         return {
#             "input_text": input_text,
#             "tokenized_text": tokenized_text,
#             "original_labels": original_labels,
#             "tokenized_labels": tokenized_labels,
#             "input_ids": token_ids,
#             "loss_mask": mask,
#             "word_to_tokens": word_to_tokens,
#         }


class TimeSeriesDataset(Dataset):
    def __init__(self, data, texts=None):
        self.data = data
        self.texts = texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = torch.tensor(self.data[index], dtype=torch.float32)
        if self.texts is not None:
            text = self.texts[index]
            return sequence, text
        return sequence
