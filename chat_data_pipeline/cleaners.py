import re
import ftfy


def fix_utf8_encoding(text):
    if text is None:
        return ""
    return ftfy.fix_text(text)


# Adapted from:
# https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/filtering.py#L95
whitespace = {" ", " ", " ", " ", " ", "　", " ", " ", " ", " ", "￼", ""}


def normalize_whitespace(text):
    chars = [char if char not in whitespace else " " for char in text]
    text = "".join(chars)
    return text


unicode_punctuation = {
    "，": ",",
    "。": ".",
    "、": ",",
    "„": '"',
    "”": '"',
    "“": '"',
    "«": '"',
    "»": '"',
    "１": '"',
    "」": '"',
    "「": '"',
    "《": '"',
    "》": '"',
    "´": "'",
    "∶": ":",
    "：": ":",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "；": ";",
    "–": "-",
    "—": " - ",
    "．": ". ",
    "～": "~",
    "’": "'",
    "…": "...",
    "━": "-",
    "〈": "<",
    "〉": ">",
    "【": "[",
    "】": "]",
    "％": "%",
    "►": "-",
}


def normalize_punctuation(text):
    chars = [unicode_punctuation.get(char, char) for char in text]
    text = "".join(chars)
    return text


def remove_empty_lines(text):
    lines = text.splitlines()
    func = lambda x: not re.match(r'^\s*$', x)
    filtered = filter(func, lines)
    text = "\n".join(filtered)
    if text is None or isinstance(text, str):
        text = ""
    return text


def clean_new_lines(text):
    text = text.strip()
    text = text.replace("\n", "")
    return text
