import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import string
from typing import Union, List
import nltk
import re
from nltk.corpus import cmudict
import syllables
import pyphen
from g2p_en import G2p


dic = pyphen.Pyphen(lang="en")
g2p = G2p()

try:
    d = cmudict.dict()
except:
    print("Downloading nltk data...")
    nltk.download("cmudict")
    d = cmudict.dict()


def find_stress_syllable_start(
    syllables, stress_index, phoneme_lab_lines, word_start, word_end, verbose=False
):
    phoneme_lab_lines = [
        line for line in phoneme_lab_lines if len(line) > 2
    ]  # Remove empty lines

    # Find the lines that are in the range
    phoneme_lab_lines = [
        (float(start), float(end), phoneme)
        for start, end, phoneme in phoneme_lab_lines
        if start >= word_start and end <= word_end
    ]

    if verbose:
        print("phoneme lab lines", phoneme_lab_lines)

    # Extract the stress syllable phonemes
    stress_syllable = syllables[stress_index]
    if verbose:
        print(f"Syllables: {syllables}")
        print(f"stress syllable: {stress_syllable} at index {stress_index}")

    candidates = []

    # The current window of phonemes
    window_phonemes = []

    for start, end, phoneme in phoneme_lab_lines:
        # print(start, end, phoneme)
        window_phonemes.append((start, phoneme))

        # Build the current window string
        curr_str = "".join(p for _, p in window_phonemes)

        if curr_str == stress_syllable:
            # If the window matches the stress syllable, add the start time as a candidate
            candidates.append(window_phonemes[0][0])

        # If the window is larger than the stress syllable, remove phonemes from the start
        while len(curr_str) > len(stress_syllable):
            window_phonemes.pop(0)
            curr_str = "".join(p for _, p in window_phonemes)

        # Check if the window matches the stress syllable again after removing phonemes
        if curr_str == stress_syllable:
            candidates.append(window_phonemes[0][0])

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        return None  # no candidates found

    # remove ambiguity by using stress index
    try:
        return candidates[stress_index]  # stress index too big
    except IndexError:
        return candidates[-1]  # so let's just take the last one as approx


def extract_phonemes(
    phoneme_lab_lines, word_phonemes, start_time, end_time, verbose=False
):
    # Split the file content by lines and split each line by tabs
    # Clean the phoneme lines by removing empty strings
    phoneme_lab_lines = [
        line for line in phoneme_lab_lines if len(line) > 2
    ]  # Remove empty lines
    # Convert start times, end times, and phonemes to a list of tuples
    phonemes = [
        (float(start), float(end), phoneme) for start, end, phoneme in phoneme_lab_lines
    ]

    if verbose:
        print(phonemes)

    # Get the phonemes for the word of interest within the start and end times
    word_phonemes_data = [
        (start, end, phoneme)
        for start, end, phoneme in phonemes
        if start_time <= start <= end_time and phoneme in word_phonemes
    ]

    return word_phonemes_data


def nb_syllables(word):
    """
    Returns the number of syllables in a word.
    If the word is not in the CMU Pronouncing Dictionary, use syllables as a fallback.
    """
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except KeyError:
        return syllables.estimate(word)


def syllabify(word):
    """
    Syllabifies a word using the CMU Pronouncing Dictionary.
    If the word is not in the dictionary, use g2p_en as a fallback.
    Returns: a list of syllables
    """
    try:
        # Get the syllabified phonemes from the dictionary
        syllabified_phonemes = d[word.lower()][0]

        # Create syllables from the phonemes
        syllables = []
        syllable = ""

        for phoneme in syllabified_phonemes:
            # Phonemes with numbers are the end of a syllable
            if "0" in phoneme or "1" in phoneme or "2" in phoneme:
                syllable += phoneme
                syllables.append(syllable)
                syllable = ""
            else:
                syllable += phoneme

        # Catch any remaining phonemes as a syllable
        if syllable:
            syllables.append(syllable)

        return syllables

    except KeyError:
        print(
            f"Word '{word}' not in CMU Pronouncing Dictionary. Using g2p for ARPABET conversion."
        )
        # Use g2p_en as a fallback
        arpabet_phonemes = g2p(word)
        syllables = []
        syllable = ""

        for phoneme in arpabet_phonemes:
            if "0" in phoneme or "1" in phoneme or "2" in phoneme:
                syllable += phoneme
                syllables.append(syllable)
                syllable = ""
            else:
                syllable += phoneme

        # Catch any remaining phonemes as a syllable
        if syllable:
            syllables.append(syllable)

        return syllables


class CelexReader:
    def __init__(self, file_path):
        self.data = self._load_data(file_path)

    def _load_data(self, file_path):
        # Dictionary to store the information
        data = {}

        # Open the file and read line by line
        with open(file_path, "r") as file:
            # Skipping header line (assuming the first line is the header)
            # If there's no header, comment the line below
            # next(file)

            # Reading each line
            for line in file:
                # Splitting the line by tabs
                (
                    head,
                    cls,
                    strs_pat,
                    phon_syl_disc,
                    morph_status,
                    cob,
                ) = line.strip().split("\\")

                # Creating a dictionary for the current word
                info = {
                    "Class": cls,
                    "StressPattern": strs_pat,
                    "PhoneticSyllable": phon_syl_disc,
                    "MorphStatus": morph_status,
                    "Frequency": cob,
                }

                # Adding to the data dictionary
                data[head] = info

        return data

    def lookup(self, word):
        # Returning the information for the requested word
        return self.data.get(word, None)

    def get_stress_syllable(self, word):
        return self.lookup(word).get("StressPattern", None)

    def get_class(self, word):
        return self.lookup(word).get("Class", None)

    def get_phonetic_syllable(self, word):
        return self.lookup(word).get("PhoneticSyllable", None)

    def get_morph_status(self, word):
        return self.lookup(word).get("MorphStatus", None)

    def get_frequency(self, word):
        return self.lookup(word).get("Frequency", None)

    def get_stress_position(self, word):
        try:
            stress_syllable = self.get_stress_syllable(word)
            if stress_syllable is not None:
                start = stress_syllable.find("1") / len(stress_syllable)
                end = (stress_syllable.rfind("1") + 1) / len(stress_syllable)
                return start, end
        except:
            return 0, 1  # default to full word stressed

    def get_stress_index(self, word):
        try:
            stress_syllable = self.get_stress_syllable(word)
            if stress_syllable is not None:
                return stress_syllable.find("1")
        except:
            return 0


def assign_labels(input_string, labels):
    # Create list to hold words and punctuation
    words_with_punctuation = re.findall(r"[\w']+|[.,!?;\"-]|'", input_string)

    # Create list to hold only words
    words_only = re.findall(r"\w+'?\w*", input_string)

    # Make sure the number of labels matches the number of words
    if not len(labels) == len(words_only):
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


def assign_labels_to_sentences(sentences, labels):
    single_words = []
    single_labels = []
    for i in range(len(sentences)):
        words_only, words_with_punct, words_with_labels = assign_labels(
            sentences[i], labels[i]
        )
        # check if alignment failed
        if words_only is None:
            # print(f"Alignment failed for sentence {i}")
            continue

        # remove Nones
        words_with_labels = [(w, l) for w, l in words_with_labels if l is not None]
        if len(words_with_labels) == 0:
            # print("No labels for sentence {i}")
            continue
        # process words and labels
        words, word_labels = zip(
            *[(w, l) for w, l in words_with_labels if l is not None]
        )
        single_words.extend(words)
        single_labels.extend(word_labels)

    return single_words, single_labels


def get_wordlist_from_string(string: str) -> List[str]:
    return re.findall(r"\w+'?\w*", string)


def get_part_of_speech(word):
    global nltk_downloads_complete
    if not nltk_downloads_complete:
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk_downloads_complete = True

    tokens = nltk.word_tokenize(word)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags[0][1]


class WordRanking:
    def __init__(self, file_path):
        self.rank_data = {}
        self._read_file(file_path)

    def _read_file(self, file_path):
        with open(file_path, "r") as file:
            next(file)  # Skip header line
            for line in file:
                parts = line.split()
                rank = int(parts[0])
                word = parts[1]
                count = int(parts[2].replace(",", ""))
                percent = float(parts[3].strip("%"))
                cumulative = float(parts[4].strip("%"))

                self.rank_data[word] = {
                    "rank": rank,
                    "count": count,
                    "percent": percent,
                    "cumulative": cumulative,
                }

    def get_rank(self, word):
        word = word.lower()
        if word in self.rank_data:
            return self.rank_data[word]["rank"]
        return None

    def get_word(self, rank):
        for word, data in self.rank_data.items():
            if data["rank"] == rank:
                return word
        return None

    def get_count(self, word):
        word = word.lower()
        if word in self.rank_data:
            return self.rank_data[word]["count"]
        return None

    def get_percent(self, word):
        word = word.lower()
        if word in self.rank_data:
            return self.rank_data[word]["percent"]
        return None

    def get_cumulative(self, word):
        word = word.lower()
        if word in self.rank_data:
            return self.rank_data[word]["cumulative"]
        return None

    def is_in_top_100k(self, word):
        word = word.lower()
        if word in self.rank_data and self.rank_data[word]["rank"] <= 100000:
            return True
        return False

    def is_in_top_Xk(self, word, X=10):
        word = word.lower()
        if word in self.rank_data and self.rank_data[word]["rank"] <= X * 1000:
            return True
        return False


def python_remove_whitespace(string):
    return "".join(string.split())


def python_lowercase_remove_punctuation(
    input_text: Union[str, List[str]]
) -> Union[str, List[str]]:
    if isinstance(input_text, str):
        return input_text.lower().translate(str.maketrans("", "", string.punctuation))
    elif isinstance(input_text, list):
        return [python_lowercase_remove_punctuation(text) for text in input_text]
    else:
        raise ValueError("Input must be a string or a list of strings")


def python_lowercase(input_text: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(input_text, str):
        return input_text.lower()
    elif isinstance(input_text, list):
        return [python_lowercase(text) for text in input_text]
    else:
        raise ValueError("Input must be a string or a list of strings")


def python_remove_punctuation(
    input_text: Union[str, List[str]]
) -> Union[str, List[str]]:
    if isinstance(input_text, str):
        return input_text.translate(str.maketrans("", "", string.punctuation))
    elif isinstance(input_text, list):
        return [python_remove_punctuation(text) for text in input_text]
    else:
        raise ValueError("Input must be a string or a list of strings")


def distribute_word_label_to_token(
    text, label, tokenizer, model_name, relative_to_prev=False, n_prev=1
):
    """
    Tokenizes the text and distributes the corresponding labels to each of it's tokens
    ::param text: string of text
    ::param label: list of labels
    ::param tokenizer: tokenizer object
    ::param model_name: name of the model/tokenizer
    ::param relative_to_prev: if True, the labels are computted relative to the previous label(s)
    ::param n_prev: number of previous labels to consider
    ::return tokens: list of tokens
    ::return token_labels: list of labels of same length as tokens
    """

    # encode each word into its tokens
    if model_name == "gpt2":
        word_encodings = [
            tokenizer.encode(x, add_prefix_space=True) for x in text.split()
        ]
    elif model_name == "bert-base-uncased":
        word_encodings = [
            tokenizer.encode(x, add_special_tokens=True) for x in text.split()
        ]
    else:
        raise ValueError("Model not supported")
    # print(f"word encodings \n", word_encodings)

    # add the labels to the tokens
    word_to_token = []
    grouped_tokens = []
    idx = 0
    for word_tokens in word_encodings:
        token_output = []
        token_group = []
        for token in word_tokens:
            token_output.append(idx)
            idx += 1
            token_group.append(token)
        word_to_token.append(token_output)
        grouped_tokens.append(token_group)

    # print("word_to_token\n", word_to_token)

    # create the labels for each token
    token_labels = []
    for i, label in enumerate(label):
        tokens_of_word = word_to_token[i]
        token_labels += [label] * len(tokens_of_word)

    tokens = [item for sublist in word_encodings for item in sublist]
    return tokens, token_labels, word_to_token, grouped_tokens


import os

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def get_paths_from_root(root_dir, ends_with=".wav"):
    """
    Returns a list of paths to files in root_dir that end with ends_with.
    """
    paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ends_with):
                paths.append(os.path.join(root, file))
    return paths


def read_lab_file(lab_path):
    """
    Returns a list of lists, where each list lines[i] contains the start time (lines[i][0]), end time (lines[i][1]), and word/phoneme (lines[i][2]).
    Note that if there a pause, len(lines[i]) < 3, since there is no word/phoneme
    """
    with open(lab_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split("\t") for line in lines]
    # Cast start and end to float
    for line in lines:
        line[0], line[1] = float(line[0]), float(line[1])
    return lines


def remove_breaks_from_lab_lines(lines):
    """
    Returns a list of lists, where each list lines[i] contains the start time (lines[i][0]), end time (lines[i][1]), and word/phoneme (lines[i][2]).
    Note that if there a pause, len(lines[i]) < 3, since there is no word/phoneme
    """
    return [line for line in lines if len(line) == 3]


def get_parts_from_lab_path(lab_path):
    """
    Returns the name parts of the lab file path.
    works for LibriTTS
    """
    path = lab_path.split(".")[0].split("/")[-1]
    reader, book, ut1, ut2 = path.split("_")
    return reader, book, ut1 + "_" + ut2


def get_words_from_lab_lines(lines):
    """
    Returns a list of words from the lab file lines.
    """
    words = []
    for line in lines:
        if len(line) == 3:
            words.append(line[2])
    return words
