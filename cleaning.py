import pandas as pd
import datetime as dt
import emoji
import pkg_resources
from symspellpy import SymSpell, Verbosity
import string
import re
import warnings
from functools import partial
import spacy
from spacy.language import Language
import multiprocessing
import numpy as np
import fasttext
from multiprocessing import Pool
from tqdm import tqdm

trump_df = pd.read_csv(r"Data/hashtag_donaldtrump.csv", lineterminator='\n')

trump_df['created_at'] = trump_df['created_at'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
trump_df['collected_at'] = trump_df['collected_at'].apply(lambda x: dt.datetime.strptime(x[:18], '%Y-%m-%d %H:%M:%S'))
warnings.filterwarnings("ignore")


class SpellChecker:
    def __init__(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    def lookup(self, input_term, max_edit_distance=2):
        suggestions = self.sym_spell.lookup(input_term, Verbosity.CLOSEST, max_edit_distance=max_edit_distance,
                                            transfer_casing=True, include_unknown=True)
        return suggestions[0].term

    def lookup_compound(self, input_term, max_edit_distance=2, **kwargs):
        suggestions = self.sym_spell.lookup_compound((input_term), max_edit_distance=max_edit_distance,
                                                     transfer_casing=True, ignore_non_words=True, **kwargs)
        return suggestions[0].term if suggestions else input_term

    def correct(self, text, **kwargs):
        result = ""
        start = 0
        changed = True
        for match in re.finditer(f"[{re.escape(string.punctuation)}]", text):
            end = match.start(0)
            spaces = re.search(r"^(\s+)", text[start: end])
            corrected_text = self.lookup_compound(text[start: end], **kwargs)
            corrected_text = spaces.group(0) + corrected_text if spaces else corrected_text
            spaces = re.search(r"(\s+)$", text[start: end])
            corrected_text = corrected_text + spaces.group(0) if spaces else corrected_text
            corrected_text += match.group(0)
            result += corrected_text
            start = match.end(0)
        spaces = re.search(r"^(\s+)", text[start:])
        corrected_text = self.lookup_compound(text[start:], **kwargs)
        corrected_text = spaces.group(0) + corrected_text if spaces else corrected_text
        result += corrected_text
        if corrected_text.lower() == text.lower():
            changed = False
        return result


def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)


def preprocess_tweet(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@(\w+)', '', text)
    text = re.sub(r'#\w+\s?', '', text)
    text = re.sub(r'[^\w\s\.,!?]', '', text)
    text = ' '.join(text.split())
    text = re.sub(r'^RT[\s]', '', text)
    text = text.lower()
    text = text.strip()
    return text


def clean_text_pipeline(text):
    text = remove_emojis(text)
    text = preprocess_tweet(text)
    return text


def correct_text(text):
    try:
        return SYM_SPELL.correct(text)
    except:
        return text


class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "/kaggle/working/lid.176.ftz"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1)
        return predictions

    def is_english(self, text):
        return self.predict_lang(text)[0][0] == "__label__en"


LANGUAGE = LanguageIdentification()
SYM_SPELL = SpellChecker()
