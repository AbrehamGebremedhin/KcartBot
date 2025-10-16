"""Language detection and translation utilities for multilingual support."""

import re
import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    AMHARIC = "am"
    AMHARIC_LATIN = "am_latin"  # Phonetic Amharic in Latin script


class LanguageDetector:
    """Detects the language of input text."""

    # Amharic Unicode range
    AMHARIC_RANGE = re.compile(r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]')

    # Common Amharic words in Latin script (phonetic)
    AMHARIC_LATIN_WORDS = {
        'neger', 'yene', 'yemay', 'yemata', 'yemibal', 'yemibalew', 'yemibalewot',
        'min', 'betam', 'bet', 'menor', 'yihonal', 'yihonalew', 'yihonalewot',
        'lela', 'lelit', 'lelitnew', 'lelitnewot', 'lelitnewotegna', 'lelitnewotegnachihu',
        'kemer', 'kemere', 'kemerachihu', 'kemerachihun', 'kemerachihunet',
        'dehna', 'dehene', 'dehenachin', 'dehenachihu', 'dehenachihun',
        'meskerem', 'tekemt', 'hedar', 'tahsas', 'ter', 'yekatit', 'megabit', 'miazia', 'ginbot', 'sene', 'hamle', 'nehase', 'pagumen'
    }

    @staticmethod
    def detect_language(text: str) -> Language:
        """
        Detect the language of the input text.

        Args:
            text: Input text to analyze

        Returns:
            Detected language
        """
        if not text or not text.strip():
            return Language.ENGLISH  # Default fallback

        text_lower = text.lower().strip()

        # Check for Amharic Unicode characters
        if LanguageDetector.AMHARIC_RANGE.search(text):
            return Language.AMHARIC

        # Check for phonetic Amharic (Latin script)
        words = set(re.findall(r'\b\w+\b', text_lower))
        amharic_latin_matches = words.intersection(LanguageDetector.AMHARIC_LATIN_WORDS)

        # If we have multiple Amharic-like words, it's likely phonetic Amharic
        if len(amharic_latin_matches) >= 2:
            return Language.AMHARIC_LATIN

        # Check for single Amharic word that might be a name or greeting
        if len(amharic_latin_matches) == 1:
            # Additional context check - if it's a common greeting or standalone
            if any(greeting in text_lower for greeting in ['selam', 'tena', 'dehna']):
                return Language.AMHARIC_LATIN

        # Default to English
        return Language.ENGLISH


class TranslationService:
    """Service for translating text between languages."""

    # Basic Amharic translations for common responses
    AMHARIC_TRANSLATIONS = {
        # Greetings and onboarding
        "Hello! Welcome to KCartBot, your fresh produce marketplace assistant. Are you a customer looking to place an order, or a supplier managing inventory?": "ሰላም! ወደ KCartBot እንኳን ደህና መጡ፣ የእርሾ ምርቶች የመሸጫ ቦታ ረዳት። የተያያዥ ለማዘዝ የምትሄድ የእርሾ ምርት ለማስተያየት የምትሄድ አቅራቢ ነህ?",
        "Welcome! What's your name?": "እንኳን ደህና መጡ! ስምህ ማን ነው?",
        "Great! What's your phone number?": "ጥሩ! የስልክ ቁጥርህ ምን ነው?",
        "Perfect! What's your default delivery location?": "ተለይቶአል! የነባሪ የማድረስ ቦታህ ምን ነው?",
        "Welcome {name}! Your account has been created. How can I help you today?": "እንኳን ደህና መጡ {name}! መለያህ ተሰራ። ዛሬ እንዴት ልረዳህ?",
        "I couldn't create your account. Please try again.": "መለያህን ልሰራ አልቻልኩም። እባክህ እንደገና ሞክር።",

        # Product availability
        "What product are you looking for?": "ምን ምርት እየለመህ ነህ?",
        "Yes, {product} is available from:": "አዎ፣ {product} ከዚህ ያለ ሊገኝ ይችላል፡",
        "Sorry, {product} is currently out of stock.": "ያልተለመድ፣ {product} በአሁኑ ጊዜ አልተለመደም።",
        "Sorry, {product} is not currently available.": "ያልተለመድ፣ {product} በአሁኑ ጊዜ አልተለመደም።",

        # Orders
        "What would you like to order?": "ምን ለማዘዝ ትፈልጋለህ?",
        "How much would you like to order?": "ምን ያህል ለማዘዝ ትፈልጋለህ?",
        "When would you like your delivery?": "ማድረስ መቼ ትፈልጋለህ?",
        "Order placed successfully! Total: {total} ETB. Payment will be Cash on Delivery.": "ትዕዛዝ ተሳካ! ጠቅላላ፡ {total} ብር። ክፍያ በማድረስ ላይ ያለ ገንዘብ ይሆናል።",

        # General
        "I'm not sure what you mean. Are you a customer or supplier?": "ምን ማለትህ እንጂ አልረዳኝም። የተያያዥ ነህ ወይም አቅራቢ?",
        "I couldn't check availability right now. Please try again.": "በአሁኑ ጊዜ መገኘት ልፈትሽ አልቻልኩም። እባክህ እንደገና ሞክር።",
        "I encountered an issue processing your request. Please try again.": "ጥያቄህን ለማስተካከል ችግር ተለመደልኝ። እባክህ እንደገና ሞክር።",
    }

    def __init__(self, llm_service):
        """Initialize translation service with LLM service for dynamic translations."""
        self.llm_service = llm_service

    async def translate_to_amharic(self, text: str) -> str:
        """
        Translate English text to Amharic.

        Args:
            text: English text to translate

        Returns:
            Amharic translation
        """
        # Check if we have a pre-defined translation
        if text in self.AMHARIC_TRANSLATIONS:
            return self.AMHARIC_TRANSLATIONS[text]

        # Use LLM for dynamic translation
        try:
            prompt = f"""
Translate the following English text to Amharic. Provide only the Amharic translation without any additional text or explanations:

English: {text}

Amharic:"""

            translation = await self.llm_service.acomplete(prompt)
            return translation.strip() if translation else text

        except Exception as exc:
            logger.error(f"Failed to translate text: {exc}")
            return text  # Return original text if translation fails

    def get_language_from_user(self, user_data: Optional[Dict[str, Any]]) -> Language:
        """
        Get user's preferred language from user data.

        Args:
            user_data: User data dictionary

        Returns:
            User's preferred language
        """
        if not user_data:
            return Language.ENGLISH

        preferred_lang = user_data.get("preferred_language", "").lower()
        if "amharic" in preferred_lang:
            return Language.AMHARIC
        elif "english" in preferred_lang:
            return Language.ENGLISH
        else:
            return Language.ENGLISH  # Default


class MultilingualResponseFormatter:
    """Formats responses based on detected or preferred language."""

    def __init__(self, translation_service: TranslationService):
        """Initialize with translation service."""
        self.translation_service = translation_service

    async def format_response(
        self,
        english_text: str,
        detected_language: Language,
        user_preferred_language: Optional[Language] = None
    ) -> str:
        """
        Format response based on language preferences.

        Args:
            english_text: The response text in English
            detected_language: Language detected from user input
            user_preferred_language: User's preferred language from profile

        Returns:
            Response in appropriate language
        """
        # Determine target language
        target_language = user_preferred_language or detected_language

        # If English is preferred or detected, return English
        if target_language in [Language.ENGLISH, Language.AMHARIC_LATIN]:
            return english_text

        # If Amharic is preferred, translate
        if target_language == Language.AMHARIC:
            return await self.translation_service.translate_to_amharic(english_text)

        # Default to English
        return english_text