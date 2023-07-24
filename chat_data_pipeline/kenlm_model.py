"""
Adapted from KenLM repository: https://huggingface.co/edugp/kenlm
"""

import os
import re
import unicodedata

from huggingface_hub import cached_download, hf_hub_url
import sentencepiece
import kenlm
from requests.exceptions import HTTPError
from typing import Dict

KENLM_MODEL_REPO = "edugp/kenlm"


class SentencePiece:
    def __init__(
            self,
            model: str,
    ):
        super().__init__()
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(model))

    def do(self, text: dict) -> dict:
        tokenized = self.sp.encode_as_pieces(text)
        return " ".join(tokenized)


class KenlmModel:
    digit_re: re.Pattern = re.compile(r"\d")
    unicode_punct: Dict[str, str] = {
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
    unicode_punct_re = re.compile(f"[{''.join(unicode_punct.keys())}]")
    non_printing_chars_re = re.compile(
        f"[{''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))}]"
    )
    kenlm_model_dir = None
    sentence_piece_model_dir = None

    def __init__(
            self,
            model_dataset: str,
            language: str,
            lower_case: bool = False,
            remove_accents: bool = False,
            normalize_numbers: bool = True,
            punctuation: int = 1,
    ):
        self.download_kenlm_model(model_dataset, language)
        try:
            self.model = kenlm.Model(self.kenlm_model_dir)
            self.tokenizer = SentencePiece(self.sentence_piece_model_dir)
        except OSError:
            os.remove(self.kenlm_model_dir)
            if os.path.exists(self.sentence_piece_model_dir):
                os.remove(self.sentence_piece_model_dir)
            raise OSError(
                "File was corrupt and should have been removed. Please, retry."
            )
        self.accent = remove_accents
        self.case = lower_case
        self.numbers = normalize_numbers
        self.punct = punctuation

    @classmethod
    def from_pretrained(
            cls,
            *,
            model_dataset: str,
            language: str,
            lower_case: bool,
            remove_accents: bool,
            normalize_numbers: bool,
            punctuation: int,
    ):
        return cls(
            model_dataset,
            language,
            lower_case,
            remove_accents,
            normalize_numbers,
            punctuation,
        )

    def pp(self, log_score, length):
        return 10.0 ** (-log_score / length)

    def get_perplexity(self, doc: str, normalize_cc_net: bool = True):
        if normalize_cc_net:
            doc = self.normalize(
                doc,
                accent=self.accent,
                case=self.case,
                numbers=self.numbers,
                punct=self.punct,
            )
        # Tokenize (after normalizing): See https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/mine.py#L352 for full pipeline
        doc = self.tokenizer.do(doc)
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return round(self.pp(doc_log_score, doc_length), 1)

    def normalize(
            self,
            line: str,
            accent: bool = True,
            case: bool = True,
            numbers: bool = True,
            punct: int = 1,
    ) -> str:
        line = line.strip()
        if not line:
            return line
        if case:
            line = line.lower()
        if accent:
            line = self.strip_accents(line)
        if numbers:
            line = self.digit_re.sub("0", line)
        if punct == 1:
            line = self.replace_unicode_punct(line)
        elif punct == 2:
            line = self.remove_unicode_punct(line)
        line = self.remove_non_printing_char(line)
        return line

    def strip_accents(self, line: str) -> str:
        """Strips accents from a piece of text."""
        nfd = unicodedata.normalize("NFD", line)
        output = [c for c in nfd if unicodedata.category(c) != "Mn"]
        if len(output) == line:
            return line
        return "".join(output)

    def replace_unicode_punct(self, text: str) -> str:
        return "".join(self.unicode_punct.get(c, c) for c in text)

    def remove_unicode_punct(self, text: str) -> str:
        """More aggressive version of replace_unicode_punct but also faster."""
        return self.unicode_punct_re.sub("", text)

    def remove_non_printing_char(self, text: str) -> str:
        return self.non_printing_chars_re.sub("", text)

    def download_kenlm_model(self, model_dataset: str, language: str):
        try:
            kenlm_model_url = hf_hub_url(
                KENLM_MODEL_REPO, filename=f"{model_dataset}/{language}.arpa.trie.bin"
            )
            self.kenlm_model_dir = cached_download(kenlm_model_url)
        except HTTPError:
            kenlm_model_url = hf_hub_url(
                KENLM_MODEL_REPO, filename=f"{model_dataset}/{language}.arpa.bin"
            )
            self.kenlm_model_dir = cached_download(kenlm_model_url)
        sentence_piece_model_url = hf_hub_url(
            KENLM_MODEL_REPO, filename=f"{model_dataset}/{language}.sp.model"
        )
        self.sentence_piece_model_dir = cached_download(sentence_piece_model_url)
