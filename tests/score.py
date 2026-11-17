"""Word accuracy of an OCR output against a hand transcription.
Words = runs of letters/digits, lowercased; LaTeX commands are dropped first.
Reports word recall (GT words found in order) and word precision."""
import re, sys
from difflib import SequenceMatcher

import unicodedata

def words(s: str) -> list[str]:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\\(frac|left|right|mathrm|text|mathbf|Delta|cdot|quad|qquad|,|;|!)", " ", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    return re.findall(r"[a-z0-9]+", s.lower())

def score(pred: str, gt: str) -> dict:
    p, g = words(pred), words(gt)
    m = sum(b.size for b in SequenceMatcher(None, g, p, autojunk=False).get_matching_blocks())
    return {"recall": m / max(1, len(g)), "precision": m / max(1, len(p)), "gt_words": len(g)}

if __name__ == "__main__":
    print(score(open(sys.argv[1], encoding="utf-8").read(), open(sys.argv[2], encoding="utf-8").read()))
