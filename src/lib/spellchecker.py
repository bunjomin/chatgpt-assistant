import re

from spellchecker import SpellChecker

class SpellChecker:
    @staticmethod
    def correct_spelling(text="")->str:
        spell = SpellChecker()
        unknown = spell.unknown(text.split())
        for word in unknown:
            text = text.replace(word, spell.correction(word))
        return text

    # Replace any consonant followed by a space and the same consonant starting a word
    # with just one instance of that consonant
    @staticmethod
    def correct_consonants(text)->str:
        corrected_text = re.sub(r'([bcdfghjklmnpqrstvwxyz])\s\1', r'\1', text)
        return corrected_text
