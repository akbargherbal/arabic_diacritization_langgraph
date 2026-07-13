"""
config/meter_tables.py
=======================
Ported verbatim from arabic_prosody_feedback.py's module-level constants.

This file is marked deny(write, edit) in main.py's permissions list. It is
ground truth for meter names/templates, not something an agent should ever
propose changing to make a verse "pass." If a meter definition here is
genuinely wrong, that's a human edit, not an agent one.
"""

CANONICAL_PATTERNS: dict[str, str] = {
    "Fawlon": "11010", "Faelon": "10110", "Faelaton": "1011010",
    "Mafaeelon": "1101010", "Mustafelon": "1010110", "Mutafaelon": "1110110",
    "Mafaelaton": "1101110", "Mafoolato": "1010101", "Mustafe_lon": "1010110",
    "Fae_laton": "1011010",
}

METER_ARABIC_NAMES: dict[str, str] = {
    "taweel": "الطويل", "madeed": "المديد", "baseet": "البسيط", "wafer": "الوافر",
    "kamel": "الكامل", "hazaj": "الهزج", "rajaz": "الرجز", "ramal": "الرمل",
    "saree": "السريع", "munsareh": "المنسرح", "khafeef": "الخفيف",
    "mudhare": "المضارع", "muqtadheb": "المقتضب", "mujtath": "المجتث",
    "mutakareb": "المتقارب", "mutadarak": "المتدارك",
}

METER_TEMPLATES: dict[str, str] = {
    "taweel": "فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِلُ",
    "madeed": "فَاعِلَاتُنْ فَاعِلُنْ فَاعِلَاتُ",
    "baseet": "مُسْتَفْعِلُنْ فَاعِلُنْ مُسْتَفْعِلُنْ فَعِلُ",
    "wafer": "مُفَاعَلَتُنْ مُفَاعَلَتُنْ فَعُولُ",
    "kamel": "مُتَفَاعِلُنْ مُتَفَاعِلُنْ مُتَفَاعِلُ",
    "hazaj": "مَفَاعِيلُنْ مَفَاعِيلُ",
    "rajaz": "مُسْتَفْعِلُنْ مُسْتَفْعِلُنْ مُسْتَفْعِلُ",
    "ramal": "فَاعِلَاتُنْ فَاعِلَاتُنْ فَاعِلَاتُ",
    "saree": "مُسْتَفْعِلُنْ مُسْتَفْعِلُنْ فَاعِلُ",
    "munsareh": "مُسْتَفْعِلُنْ مَفْعُولَاتُ مُفْتَعِلُ",
    "khafeef": "فَاعِلَاتُنْ مُسْتَفْعِلُنْ فَاعِلَاتُ",
    "mudhare": "مَفَاعِيلُ فَاعِلَاتُ",
    "muqtadheb": "مَفْعُولَاتُ مُفْتَعِلُ",
    "mujtath": "مُسْتَفْعِلُنْ فَاعِلَاتُ",
    "mutakareb": "فَعُولُنْ فَعُولُنْ فَعُولُنْ فَعُولُ",
    "mutadarak": "فَعِلُنْ فَعِلُنْ فَعِلُنْ فَعِلُ",
}

_ZIHAF_MAP: dict[tuple[str, str], str] = {
    ("11010", "1101"): "Qabadh", ("11010", "110"): "Hadhf", ("11010", "10"): "Batr",
    ("10110", "1110"): "Khaban",
    ("1011010", "111010"): "Khaban", ("1011010", "101101"): "Kaff",
    ("1011010", "10110"): "Hadhf", ("1011010", "11101"): "Shakal",
    ("1011010", "1011"): "Waqf",
    ("1101010", "110110"): "Qabadh", ("1101010", "110101"): "Kaff",
    ("1101010", "11010"): "Hadhf", ("1101010", "11011"): "Shakl_alt",
    ("1010110", "110110"): "Khaban", ("1010110", "101110"): "Tay",
    ("1010110", "11110"): "Khabal", ("1010110", "101010"): "Kasf",
    ("1110110", "1010110"): "Edmaar", ("1110110", "110110"): "Waqas",
    ("1110110", "101110"): "Khazal",
    ("1101110", "110110"): "Akal", ("1101110", "1101010"): "Asab",
    ("1101110", "11010"): "Qatf",
    ("1010101", "110101"): "Khaban", ("1010101", "101101"): "Tay",
    ("1010101", "10101"): "Kasf",
}

_ALIASES: dict[str, str] = {
    "tawil": "tawil", "al-tawil": "tawil", "ṭawīl": "tawil", "طويل": "tawil", "الطويل": "tawil",
    "basit": "basit", "al-basit": "basit", "basīṭ": "basit", "بسيط": "basit", "البسيط": "basit",
    "kamil": "kamil", "al-kamil": "kamil", "kāmil": "kamil", "كامل": "kamil", "الكامل": "kamil",
    "wafir": "wafir", "al-wafir": "wafir", "wāfir": "wafir", "وافر": "wafir", "الوافر": "wafir",
    "ramal": "ramal", "al-ramal": "ramal", "رمل": "ramal", "الرمل": "ramal",
    "mutaqarib": "mutaqarib", "al-mutaqarib": "mutaqarib", "mutaqārib": "mutaqarib",
    "متقارب": "mutaqarib", "المتقارب": "mutaqarib",
    "mutadarak": "mutadarak", "al-mutadarak": "mutadarak", "mutadārak": "mutadarak",
    "متدارك": "mutadarak", "المتدارك": "mutadarak", "الخبب": "mutadarak",
    "rajaz": "rajaz", "al-rajaz": "rajaz", "رجز": "rajaz", "الرجز": "rajaz",
    "khafif": "khafif", "al-khafif": "khafif", "khafīf": "khafif", "خفيف": "khafif", "الخفيف": "khafif",
}

_METER_TABLE_TO_PYARUD: dict[str, str] = {
    "tawil": "taweel", "basit": "baseet", "kamil": "kamel", "wafir": "wafer",
    "ramal": "ramal", "mutaqarib": "mutakareb", "mutadarak": "mutadarak",
    "rajaz": "rajaz", "khafif": "khafeef",
}
