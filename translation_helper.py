import deepl
import io
from deep_translator import GoogleTranslator
from fairseq.models.transformer import TransformerModel
from contextlib import redirect_stdout
import logging
import warnings
from enum import Enum
# import json

class TranslatorService(Enum):
    GOOGLE = 'google'
    DEEPL = 'deepl'
    SUGOI = 'sugoi'

class Language(Enum):
    ENGLISH = 'en'
    ENGLISH_DEEPL = 'EN-US'
    JAPANESE = 'ja'

Languages = {
    "english": "en",
    "japanese": "ja"
}

class Translator:

    def __init__(self, translator=None, source_lang=None, target_lang=None):
        self.translator = translator if translator is not None else TranslatorService.GOOGLE
        self.source_lang = source_lang if source_lang is not None else Language.JAPANESE
        self.source_abrv = "ja"
        self.target_lang = target_lang if target_lang is not None else Language.ENGLISH
        self.target_abrv = "en"
        self.total_character_count_translated = 0

    def translate(self, input_text_list):
        # Concatenate the text with a unique separator
        separator = "\n"
        concatenated_text = separator.join(input_text_list)

        # count characters translated
        self.total_character_count_translated += len(concatenated_text)

        # translate
        if (self.translator == TranslatorService.GOOGLE):
            translated_text = self.translate_google(concatenated_text)
        elif (self.translator == TranslatorService.DEEPL):
            translated_text = self.translate_deepL(concatenated_text)
        elif (self.translator == TranslatorService.SUGOI):
            return self.translate_sugoi(separator, input_text_list)
        else:
            print("No translator selected!")

        
        # Split the translated text back into individual translations
        translated_text_list = translated_text.split(separator)

        # Save translated text to file

        return translated_text, translated_text_list


    def translate_google(self, input_text):
        if self.source_lang == Language.JAPANESE: self.source_abrv = "ja"
        if self.target_lang == Language.ENGLISH: self.target_abrv = "en"

        translated_text = GoogleTranslator(source_lang=self.source_abrv, target_lang=self.target_abrv).translate(input_text)
        return translated_text

    def translate_deepL(self, input_text, auth_key):
        if self.source_lang == Language.JAPANESE: self.source_abrv = "JA"
        if self.target_lang == Language.ENGLISH: self.target_abrv = "EN-US"

        auth_key = ""  
        translator = deepl.Translator(auth_key)
        translated_text = translator.translate_text(input_text, source_lang=self.source_abrv, target_lang=self.target_abrv).text
        return translated_text
    

    # https://www.patreon.com/mingshiba
    def translate_sugoi(self, separator, input_text_list):

        logging.getLogger().setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore")

        ja2en = TransformerModel.from_pretrained(
            './Sugoi-Translator-Toolkit-V4.0-Public/Code/backendServer/Program-Backend/Sugoi-Japanese-Translator/offlineTranslation/fairseq/japaneseModel',
            checkpoint_file='big.pretrain.pt',
            source_lang = "ja",
            target_lang = "en",
            bpe = 'sentencepiece',
            sentencepiece_model = './Sugoi-Translator-Toolkit-V4.0-Public/Code/backendServer/Program-Backend/Sugoi-Japanese-Translator/offlineTranslation/fairseq/spmModels/spm.ja.nopretok.model',
            is_gpu = True
        )
        ja2en.cuda()
        
        translated_text_list = []
        for text in input_text_list:
            translated_text_list.append(ja2en.translate(text))
        translated_text = separator.join(translated_text_list)

        warnings.filterwarnings("default")
        logging.getLogger().setLevel(20) # standard logging level

        return translated_text, translated_text_list
        
        #result = ja2en.translate(input_text)
        #return result
        #json.dumps(result)
        #print("Result: " + result)


    def getTranslatorService(self):
        if self.translator == TranslatorService.GOOGLE:
            return "google"
        elif self.translator == TranslatorService.DEEPL:
            return "deepl"
        elif self.translator == TranslatorService.SUGOI:
            return "sugoi"
        else:
            return "nothing"
        






