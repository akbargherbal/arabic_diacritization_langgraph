import sys
import os

# Add package root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subagents.formatter import heuristic_salvage, strip_diacritics, clean_and_normalize

def test_strip_diacritics():
    assert strip_diacritics("كَاتِبٌ") == "كاتب"
    assert strip_diacritics("الشَّعْرُ") == "الشعر"

def test_heuristic_salvage_exact_match():
    expected = [
        {"verse_id": "687-1", "sadr": "قفا نبك من ذكرى حبيب ومنزل", "ajuz": "بسقط اللوى بين الدخول فحومل"},
        {"verse_id": "687-2", "sadr": "كأن عيون الوحش حول كناتنا", "ajuz": "جزع سراة لم يحلل"},
    ]
    
    # 1. Exact list, matching length but missing/incorrect verse_ids
    parsed = [
        {"sadr": "قَفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ", "ajuz": "بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"},
        {"sadr": "كَأَنَّ عُيُونَ الْوَحْشِ حَوْلَ كِنَاتِنَا", "ajuz": "جِزْعُ سَرَاةِ لَمْ يُحَلِّلِ"},
    ]
    
    res = heuristic_salvage(parsed, expected)
    assert res is not None
    assert "687-1" in res
    assert "687-2" in res
    assert res["687-1"]["sadr"] == "قَفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
    assert res["687-2"]["sadr"] == "كَأَنَّ عُيُونَ الْوَحْشِ حَوْلَ كِنَاتِنَا"

def test_heuristic_salvage_mismatched_length_closeness():
    expected = [
        {"verse_id": "687-1", "sadr": "قفا نبك من ذكرى حبيب ومنزل", "ajuz": "بسقط اللوى بين الدخول فحومل"},
        {"verse_id": "687-2", "sadr": "كأن عيون الوحش حول كناتنا", "ajuz": "جزع سراة لم يحلل"},
    ]
    
    # Mismatched length: parsed has a different count, but we can match text
    parsed = [
        {"sadr": "قَفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ", "ajuz": "بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"},
    ]
    
    res = heuristic_salvage(parsed, expected)
    assert res is not None
    assert "687-1" in res
    assert "687-2" in res  # filled with fallback
    assert res["687-1"]["sadr"] == "قَفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
    assert res["687-2"]["sadr"] == "كأن عيون الوحش حول كناتنا"

if __name__ == "__main__":
    test_strip_diacritics()
    test_heuristic_salvage_exact_match()
    test_heuristic_salvage_mismatched_length_closeness()
    print("ALL TESTS PASSED SUCCESSFULLY!")
