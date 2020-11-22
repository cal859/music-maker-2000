from copy import deepcopy
from collections import Counter
import re
import numpy as np
from nltk.stem import WordNetLemmatizer


class CleanTextData:
    # def __init__(self, clean_text_args: dict = {}):
    #     self.lower_text = clean_text_args.get("lower_text", True)
    #     self.lemma = clean_text_args.get("lemma", True)
    #     self.strip_punctuation = clean_text_args.get("strip_punctuation", True)
    #     self.keep_numbers = clean_text_args.get("keep_numbers", True)
    #     self.special_tokens_to_keep = clean_text_args.get(
    #         "special_tokens_to_keep", ["<SONGSTART>", "<SONGEND>", "\n"]
    #     )
    #     self.split_criteria = clean_text_args.get("split_criteria", " ")
    def __init__(
        self,
        lower_text: bool = True,
        lemma: bool = True,
        strip_punctuation: bool = True,
        keep_numbers: bool = True,
        special_tokens_to_keep: list = ["<SONGSTART>", "<SONGEND>", "\n"],
        split_criteria: str = " ",
    ):
        self.lower_text = lower_text
        self.lemma = lemma
        self.strip_punctuation = strip_punctuation
        self.keep_numbers = keep_numbers
        self.special_tokens_to_keep = special_tokens_to_keep
        self.split_criteria = split_criteria

    def process_text(self, text: str) -> list:
        clean_text = deepcopy(text)
        if self.special_tokens_to_keep:
            for w in self.special_tokens_to_keep:
                clean_text = clean_text.replace(w, f" {w} ")
        if not self.keep_numbers:
            clean_text = re.sub(r"\d+", "<NUMBER>")
        if self.lower_text:
            clean_text = clean_text.lower()
        if self.strip_punctuation:
            clean_text = re.sub(r"[\.,\-\"!\?\(\)\{\}\\\/]", "", clean_text)

        clean_text = clean_text.split(self.split_criteria)
        # mini cleaning here
        clean_text = [x for x in clean_text if len(x) > 0 and x != " "]
        if self.lemma:
            # TODO: Is there a faster way to do this, almost certainly
            # TODO: Create way to allow for different lemma models
            wnl = WordNetLemmatizer()
            clean_text = [wnl.lemmatize(x) for x in clean_text]

        self.clean_text = clean_text

        return clean_text

    def clean_data(self, text: str) -> list:
        return self.process_text(text)