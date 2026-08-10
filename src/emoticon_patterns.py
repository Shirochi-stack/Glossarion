# -*- coding: utf-8 -*-
"""Curated text-emoticon defaults used by the QA foreign-script scanner.

Unicode does not define an exhaustive set of text emoticons.  These are
literal, user-editable defaults covering common Western emoticons, kaomoji,
Korean Jamo faces, and mixed-script faces that contain characters a
translation QA pass would otherwise correctly classify as foreign text.
"""

import re
from functools import lru_cache


def _deduplicate(patterns):
    return tuple(dict.fromkeys(patterns))


DEFAULT_EMOTICON_PATTERNS = _deduplicate((
    # Western / ASCII-style emoticons
    ":-)", ":)", ":]", "=)", ":}", ":^)", ":-D", ":D", "=D", "x-D",
    "XD", "xD", ":-P", ":P", ":p", "=P", ":-3", ":3", ":c)", ":o)",
    ":-(", ":(", ":[", "=(", ":{", ":'(", ":'-(", "D:", "D-:", "D:<",
    ":-/", ":/", ":\\", ":-\\", ":-|", ":|", ";-)", ";)", "*-)", "*)",
    ";-]", ";]", ";D", ";^)", ":-*", ":*", ":×", "<3", "</3", "8-)",
    "8)", "B-)", "B)", "O:-)", "O:)", "0:-3", "0:3", ">:-)", ">:)",
    "}:-)", "}:)", "3:-)", "3:)", "o_O", "O_o", "O_O", "0_0", "^_^",
    "^-^", "^.^", "-_-", "._.", "T_T", "T.T", "Q_Q", "Q.Q", ";_;",
    "u_u", "U_U", "UwU", "OwO", "D;", "DX",

    # Happy / friendly kaomoji
    "(＾▽＾)", "(＾ω＾)", "(⌒▽⌒)", "(⌒ω⌒)", "(★ω★)", "(☆▽☆)",
    "(☆ω☆)", "(✯ᴗ✯)", "(◕‿◕)", "(◠‿◠)", "(◡‿◡)", "(≧▽≦)",
    "(≧◡≦)", "(*^‿^*)", "(*^▽^*)", "(*≧ω≦*)", "(*⌒▽⌒*)", "(*´▽`*)",
    "(o^▽^o)", "(o´▽`o)", "ヽ(・∀・)ﾉ", "ヽ(o＾▽＾o)ノ", "＼(＾▽＾)／",
    "＼(^o^)／", "(^人^)", "(o´∀`o)", "(´• ω •`)", "(｡•̀ᴗ-)✧",
    "(✧ω✧)", "(￣▽￣)", "(￣ω￣)", "(つ≧▽≦)つ", "(づ｡◕‿‿◕｡)づ",
    "༼ つ ◕_◕ ༽つ", "(つ✧ω✧)つ", "(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧",
    "☆*:.｡.o(≧▽≦)o.｡.:*☆", "｡ﾟ( ﾟ^∀^ﾟ)ﾟ｡", "(๑˃ᴗ˂)ﻭ", "٩(◕‿◕｡)۶",
    "(b ᵔ▽ᵔ)b", "(っ˘ω˘ς )", "(*￣▽￣)b", "ヾ(・ω・*)",
    "(＠＾◡＾)", "(─‿‿─)", "(✿◠‿◠)", "(◕ᴗ◕✿)", "(｡•̀ᴗ-)✧",

    # Sad / crying / tired kaomoji
    "(╥﹏╥)", "(ಥ﹏ಥ)", "(ಥ_ಥ)", "(T_T)", "(T⌓T)", "(个_个)",
    "(╯︵╰,)", "(っ˘̩╭╮˘̩)っ", "(｡•́︿•̀｡)", "(｡╯︵╰｡)", "(ノ_<。)",
    "(μ_μ)", "(ﾉД`)", "(っ- ‸ - ς)", "(´-ω-`)", "(´；ω；`)",
    "｡ﾟ･ (>﹏<) ･ﾟ｡", "。゜゜(´Ｏ`) ゜゜。", "･ﾟ･(｡>ω<｡)･ﾟ･", "(ᗒᗣᗕ)՞",
    "(ಢ_ಢ)", "(ಥ益ಥ)", "(இ﹏இ`)", "(つ﹏⊂)", "(｡•́︵•̀｡)",
    "(っ˘̩╭╮˘̩)っ", "(ノД`)・゜・。", "｡･ﾟﾟ*(>д<)*ﾟﾟ･｡",

    # Angry / determined kaomoji
    "(＃`Д´)", "(`皿´＃)", "ヽ(`д´*)ノ", "(・`ω´・)", "(`ー´)",
    "ヽ(｀⌒´メ)ノ", "凸(￣ヘ￣)", "凸(`△´＃)", "(｀ε´)", "(￣^￣)",
    "(¬_¬\")", "(＃￣ω￣)", "(╬ಠ益ಠ)", "٩(╬ʘ益ʘ╬)۶", "(ノಠ益ಠ)ノ",
    "(ง'̀-'́)ง", "(ง •̀_•́)ง", "ᕦ(ò_óˇ)ᕤ", "ᕙ(⇀‸↼‶)ᕗ", "щ(ﾟДﾟщ)",
    "щ(ಠ益ಠщ)", "ლ(ಠ益ಠლ)", "(╯`Д´)╯", "щ(゜ロ゜щ)", "щ(ಥДಥщ)",

    # Surprise / confusion
    "(⊙_⊙)", "(o_O)", "(O_O;)", "(°ロ°) !", "(°Д°)", "(゜ロ゜)",
    "(ﾟДﾟ)", "Σ(°△°|||)", "Σ(O_O)", "w(°ｏ°)w", "ヽ(°〇°)ﾉ",
    "＼(〇_ｏ)／", "( : ౦ ‸ ౦ : )", "(๑•́ ₃ •̀๑)", "(・・;)",
    "(•ิ_•ิ)?", "(・・ ) ?", "(￣ω￣;)", "(＠_＠)", "(⊙_◎)", "(☉_☉)",
    "(⊙﹏⊙)", "(・・;φ", "(◎ ◎)ゞ", "(；￣Д￣)", "(￣Д￣)",
    "(Дﾟ≡ﾟДﾟ)", "(ﾟДﾟ≡ﾟДﾟ)", "ヽ(ﾟДﾟ)ﾉ",

    # Love, hugs, and animal faces
    "(♡°▽°♡)", "(❤ω❤)", "(´,,•ω•,,)♡", "(´• ω •`) ♡", "(ღ˘⌣˘ღ)",
    "(っ˘з(˘⌣˘ )", "( ˘⌣˘)♡(˘⌣˘ )", "(づ￣ ³￣)づ", "(っ˘з(˘⌣˘ ) ♡",
    "(´ε｀ )♡", "(*♡∀♡)", "(｡♥‿♥｡)", "(♥ω♥*)", "(´♡‿♡`)",
    "♡( ◡‿◡ )", "(◍•ᴗ•◍)❤", "( ˘ ³˘)♥", "ʕ•ᴥ•ʔ", "ʕっ•ᴥ•ʔっ",
    "ʕ￫ᴥ￩ʔ", "ฅ^•ﻌ•^ฅ", "(=^･ω･^=)", "(^・ω・^ )", "(=①ω①=)",
    "ଲ(ⓛ ω ⓛ)ଲ", "／(≧ x ≦)＼", "(=｀ω´=)", "(=;ェ;=)",

    # Shrugs, gestures, table flips, dancing, and sleeping
    "¯\\_(ツ)_/¯", "¯\\(°_o)/¯", "┐('～`;)┌", "┐(￣ヘ￣)┌",
    "╮(￣ω￣;)╭", "╮(︶▽︶)╭", "┐(シ)┌", "┐(´д`)┌", "ヽ(ー_ー )ノ",
    "┐(‘～` )┌", "┐( ˘_˘ )┌", "╮( ˘ ､ ˘ )╭", "(╯°□°）╯︵ ┻━┻",
    "(ノಠ益ಠ)ノ彡┻━┻", "┻━┻ ︵ヽ(`Д´)ﾉ︵ ┻━┻", "(ﾉಥ益ಥ）ﾉ ┻━┻",
    "┬─┬ ノ( ゜-゜ノ)", "┬─┬ ﾉ(° -°ﾉ)", "(ヘ･_･)ヘ┳━┳",
    "┬──┬ ¯\\_(ツ)", "(っ•́｡•́)♪♬", "♪(´▽｀)", "ヾ(´〇`)ﾉ♪♪♪",
    "ヘ(￣ω￣ヘ)", "(〜￣▽￣)〜", "〜(꒪꒳꒪)〜", "(－‸ლ)",
    "(－ω－) zzZ", "(￣o￣) zzZZzzZZ", "(－_－) zzZ", "(∪｡∪)｡｡｡zzZ",
    "(*￣m￣)", "(￣ε￣＠)", "(￣3￣)",

    # Korean Jamo-style emoticons and common repeated reactions
    "ㅠㅠ", "ㅠㅠㅠ", "ㅠㅠㅠㅠ", "ㅜㅜ", "ㅜㅜㅜ", "ㅜㅜㅜㅜ",
    "ㅋㅋ", "ㅋㅋㅋ", "ㅋㅋㅋㅋ", "ㅎㅎ", "ㅎㅎㅎ", "ㅎㅎㅎㅎ",
    "ㄷㄷ", "ㄷㄷㄷ", "ㄷㄷㄷㄷ", "ㅇㅅㅇ", "ㅇㅁㅇ", "ㅇㅂㅇ", "ㅇ_ㅇ",
    "ㅇㅈㅇ", "ㅎㅅㅎ", "ㅎ_ㅎ", "ㅅ_ㅅ", "ㅂ_ㅂ", "ㅍ_ㅍ", "ㅋ_ㅋ",
    "ㄱㅅㄱ", "ㄴㅇㄱ", "ㅜ_ㅜ", "ㅠ_ㅠ", "ㅡ_ㅡ", "-ㅅ-", "^ㅅ^",
    ">ㅅ<", "◉ㅅ◉", "•ㅅ•", "⊙ㅅ⊙", "(ㅇㅅㅇ)", "(ㅇㅁㅇ)", "(ㅇㅂㅇ)",
    "(ㅇ_ㅇ)", "(ㅎㅅㅎ)", "(ㅎ_ㅎ)", "(ㅅ_ㅅ)", "(ㅂ_ㅂ)", "(ㅍ_ㅍ)",
    "(ㅋ_ㅋ)", "(ㄱㅅㄱ)", "(ㄴㅇㄱ)", "(ㅜ_ㅜ)", "(ㅠ_ㅠ)", "(ㅡ_ㅡ)",
    "(-ㅅ-)", "(^ㅅ^)", "(>ㅅ<)", "(◉ㅅ◉)", "(•ㅅ•)", "(⊙ㅅ⊙)",

    # Cyrillic and other mixed-script faces commonly seen in kaomoji
    "(Ф_Ф)", "(Ж_Ж)", "(Д_Д)", "(ш_ш)", "(ц_ц)", "ヽ༼ຈل͜ຈ༽ﾉ",
    "༼ ºل͟º ༼ ºل͟º ༽ ºل͟º ༽", "༼ つ ಥ_ಥ ༽つ", "ಠ_ಠ", "ಥ_ಥ",
    "ಠ‿ಠ", "ಠ⌣ಠ", "ಠ益ಠ", "ಥ﹏ಥ", "ಥ‿ಥ", "ಡ_ಡ", "ಢ_ಢ",
))


def normalize_emoticon_patterns(patterns):
    """Return non-empty string patterns, preserving user order and uniqueness."""
    if not isinstance(patterns, (list, tuple)):
        return ()
    normalized = []
    seen = set()
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            continue
        if pattern in seen:
            continue
        seen.add(pattern)
        normalized.append(pattern)
    return tuple(normalized)


@lru_cache(maxsize=32)
def _compiled_emoticon_matchers(patterns, patterns_are_regex=False):
    if patterns_are_regex:
        matchers = []
        for pattern in patterns:
            try:
                matchers.append(re.compile(pattern))
            except re.error:
                continue
        return tuple(matchers)

    alternatives = [
        re.escape(pattern)
        for pattern in sorted(patterns, key=len, reverse=True)
    ]
    if not alternatives:
        return ()
    return (re.compile("|".join(alternatives)),)


def mask_whitelisted_emoticons(text, patterns=None, patterns_are_regex=False):
    """Replace complete whitelisted emoticon matches with same-length spaces."""
    if not isinstance(text, str) or not text:
        return text
    if patterns is None:
        patterns = DEFAULT_EMOTICON_PATTERNS
    normalized = normalize_emoticon_patterns(patterns)
    matchers = _compiled_emoticon_matchers(normalized, bool(patterns_are_regex))
    masked = text
    for matcher in matchers:
        masked = matcher.sub(lambda match: " " * len(match.group(0)), masked)
    return masked
