"""
文の出力の番号の並びを, データセットのレシピの語彙で文字の文に戻す

語彙と分け方 (語, 文字, 部分語) は ML-EDA のレシピの最後の vocabulary の
手順が当てはめた状態 (state.vocabulary か, 部分語の区切りの JSON) と設定
(unit, length) から読む
"""
import json
import re
from typing import Any, Optional

# 埋め草の番号 (ML-EDA の語彙の 0)
PAD = 0

# 並びの長さの既定 (ML-EDA の vocabulary の length の既定)
DEFAULT_LENGTH = 128

# 部分語の空白の印 (ML-EDA の部分語の区切りの Metaspace)
METASPACE = "\u2581"

# 語の分け方で, 空白を挟んで並べる語 (かなと漢字を除く文字と数字の並び.
# ML-EDA の語の分け方と同じ文字の組) と, 語彙の外の印
SPACED = re.compile(r"[^\W_\u3040-\u30ff\u3400-\u9fff]+|<[^<>]+>")

# 後ろの語との間に空白を入れる記号と, 前の語との間に空白を入れる記号
AFTER_SPACE = (",", ".", "!", "?", ";", ":", ")")
BEFORE_SPACE = ("(",)


def subword_pieces(serialized: Any) -> Optional[list[str]]:
    """
    部分語の区切りの JSON から, 番号の順の部分語を返す関数

    Parameters
    ----------
    serialized: Any
        区切りの JSON (tokenizers の Tokenizer.to_str の文字)

    Returns
    ----------
    pieces: Optional[list[str]]
        番号の順の部分語 (読めなければ None)
    """
    # BPE の model.vocab (部分語から番号) を番号の順に (tokenizers は不要)
    try:
        vocab = json.loads(str(serialized))["model"]["vocab"]
        ordered = sorted(vocab.items(), key=lambda item: int(item[1]))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    pieces = [str(piece) for piece, _ in ordered]

    return pieces


def text_vocabulary(recipe: dict) -> Optional[dict]:
    """
    レシピの最後の vocabulary の手順から, 語彙と分け方を返す関数

    Parameters
    ----------
    recipe: dict
        当てはめたレシピ

    Returns
    ----------
    decoding: Optional[dict]
        vocabulary (番号の順の語), unit (word, char, subword), length
        (並びの長さ). 最後の手順に当てはめた語彙が無ければ None
    """
    # 前処理がかける最後の vocabulary の手順だけを見る
    steps = [
        step for step in recipe.get("item_steps") or []
        if isinstance(step, dict) and step.get("op") == "vocabulary"
    ]
    if not steps:
        return None
    step = steps[-1]
    state = step.get("state") or {}
    unit = str(step.get("unit", "word"))

    # 語彙の並び (部分語で並びの無い古いレシピは区切りの JSON から)
    vocabulary = state.get("vocabulary")
    if not vocabulary and unit == "subword" and state.get("tokenizer"):
        vocabulary = subword_pieces(state["tokenizer"])
    if not vocabulary:
        return None
    decoding = {
        "vocabulary": [str(v) for v in vocabulary],
        "unit": unit,
        "length": int(step.get("length", DEFAULT_LENGTH)),
    }

    return decoding


def trim_ids(
    ids: list[int], prepend: list[int], start: int, eos: Any
) -> list[int]:
    """
    番号の並びから, 先頭に足した番号, 終わりの番号の後, 埋め草を除く関数

    Parameters
    ----------
    ids: list[int]
        1 件の番号の並び
    prepend: list[int]
        前処理の外で先頭に足した番号 (無ければ空)
    start: int
        先頭に足した番号の後の, 渡した文の番号の並びの長さ (終わりの
        番号はこれより後ろでだけ探す)
    eos: Any
        終わりの番号 (None なら切らない)

    Returns
    ----------
    kept: list[int]
        文に戻す番号
    """
    # 先頭に足した番号は語彙の外なので除く
    if prepend and ids[:len(prepend)] == prepend:
        ids = ids[len(prepend):]

    # 終わりの番号は足した番号の中だけで探す (渡した文や埋め草では切らない)
    if eos is not None and int(eos) in ids[start:]:
        ids = ids[:start + ids[start:].index(int(eos))]
    kept = [v for v in ids if v != PAD]

    return kept


def spaced(before: str, after: str, earlier: Optional[str]) -> bool:
    """
    語の分け方で, 2 つの語の間に空白を入れるかを返す関数

    Parameters
    ----------
    before: str
        前の語
    after: str
        後ろの語
    earlier: Optional[str]
        前の語のさらに前の語 (無ければ None)

    Returns
    ----------
    result: bool
        文字と数字の並びどうし, 区切りの記号の後ろの語と開き括弧, 語の後ろ
        の開き括弧なら True. ただし数字の間の区切りの記号 (3.5, 10:30,
        1,000) の後ろは False
    """
    # 数字に挟まれた区切りの記号は数の書き方
    if before in AFTER_SPACE and after.isdigit() and (
        earlier is not None and earlier.isdigit()
    ):
        return False

    # 英語のような書き方の空白だけを戻す
    word_before = bool(SPACED.fullmatch(before))
    word_after = bool(SPACED.fullmatch(after))
    result = (word_before and word_after) or (
        before in AFTER_SPACE and (word_after or after in BEFORE_SPACE)
    ) or (word_before and after in BEFORE_SPACE)

    return result


def join_tokens(tokens: list[str], unit: str) -> str:
    """
    語の並びを, 分け方に合わせて文字の文にする関数

    Parameters
    ----------
    tokens: list[str]
        語の並び
    unit: str
        分け方 (word, char, subword)

    Returns
    ----------
    text: str
        文
    """
    # 文字はそのまま, 部分語は空白の印を空白に
    if unit == "char":
        text = "".join(tokens)
        return text
    if unit == "subword":
        text = "".join(tokens).replace(METASPACE, " ").strip()
        return text

    # 語は決まった組の間だけに空白
    parts: list[str] = []
    for index, token in enumerate(tokens):
        earlier = tokens[index - 2] if index >= 2 else None
        if index and spaced(tokens[index - 1], token, earlier):
            parts.append(" ")
        parts.append(token)
    text = "".join(parts)

    return text


def ids_to_text(
    ids: list[int], decoding: dict, prepend: list[int], eos: Any
) -> str:
    """
    1 件の番号の並びを, 語彙で文字の文に戻す関数

    Parameters
    ----------
    ids: list[int]
        番号の並び
    decoding: dict
        語彙, 分け方, 並びの長さ (text_vocabulary の結果)
    prepend: list[int]
        前処理の外で先頭に足した番号
    eos: Any
        終わりの番号

    Returns
    ----------
    text: str
        文 (語彙の外の番号は <番号>)
    """
    # 番号から語へ (語彙の外は番号を見せる)
    vocabulary = decoding["vocabulary"]
    start = int(decoding.get("length", DEFAULT_LENGTH))
    tokens = [
        vocabulary[v] if 0 <= v < len(vocabulary) else f"<{v}>"
        for v in trim_ids(ids, prepend, start, eos)
    ]
    text = join_tokens(tokens, decoding["unit"])

    return text
