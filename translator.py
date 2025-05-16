# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Translator Module

Provides language translation capabilities using external translation APIs or LLM prompting.
"""

import logging
import json
import re
import requests
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class Translator:
    """Tool for language translation."""

    def __init__(self, llm_translator=None):
        """
        Initialize the translator.

        Args:
            llm_translator: Optional LLM-based translator function
        """
        self.llm_translator = llm_translator
        self.language_codes = {
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'italian': 'it',
            'portuguese': 'pt',
            'russian': 'ru',
            'japanese': 'ja',
            'chinese': 'zh',
            'korean': 'ko',
            'arabic': 'ar',
            'hindi': 'hi',
            'dutch': 'nl',
            'swedish': 'sv',
            'finnish': 'fi',
            'turkish': 'tr',
            'polish': 'pl',
            'vietnamese': 'vi',
            'thai': 'th',
            'greek': 'el'
        }

    def translate_text(self, text: str,
                      source_lang: str = "auto",
                      target_lang: str = "en",
                      api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Translate text from source language to target language.

        Args:
            text: Text to translate
            source_lang: Source language code or name (auto for detection)
            target_lang: Target language code or name
            api_key: Optional API key for external translation service

        Returns:
            Dict containing translated text and metadata
        """
        # Normalize language codes
        source_code = self._normalize_language(source_lang)
        target_code = self._normalize_language(target_lang)

        logger.info(f"Translating from {source_code} to {target_code}")

        # Try different translation methods in order of preference

        # 1. Use external API if API key is provided
        if api_key:
            try:
                result = self._translate_with_api(text, source_code, target_code, api_key)
                if result["status"] == "success":
                    return result
            except Exception as e:
                logger.error(f"API translation failed: {str(e)}")

        # 2. Use LLM-based translation if available
        if self.llm_translator:
            try:
                result = self._translate_with_llm(text, source_code, target_code)
                if result["status"] == "success":
                    return result
            except Exception as e:
                logger.error(f"LLM translation failed: {str(e)}")

        # 3. Use LibreTranslate as fallback (no API key required)
        try:
            result = self._translate_with_libre(text, source_code, target_code)
            if result["status"] == "success":
                return result
        except Exception as e:
            logger.error(f"LibreTranslate failed: {str(e)}")

        # 4. Last resort: use a simple pattern-based translation for common phrases
        result = self._simple_translation(text, source_code, target_code)

        return result

    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of the given text.

        Args:
            text: Text to analyze

        Returns:
            Dict containing detected language and confidence
        """
        # Try to detect using pattern matching first
        detected = self._pattern_detect_language(text)

        if detected["confidence"] > 0.7:
            return detected

        # Fallback to LibreTranslate
        try:
            return self._libre_detect_language(text)
        except Exception as e:
            logger.error(f"LibreTranslate language detection failed: {str(e)}")
            return detected

    def _normalize_language(self, lang: str) -> str:
        """
        Normalize language name or code to standard language code.

        Args:
            lang: Language name or code

        Returns:
            Normalized language code
        """
        if lang.lower() == "auto":
            return "auto"

        # If it's already a valid 2-letter code, return it
        if lang.lower() in self.language_codes.values():
            return lang.lower()

        # Try to find the language name
        if lang.lower() in self.language_codes:
            return self.language_codes[lang.lower()]

        # Check for partial matches
        for name, code in self.language_codes.items():
            if lang.lower() in name:
                return code

        # Fallback to English if not found
        logger.warning(f"Unknown language '{lang}', defaulting to English")
        return "en"

    def _translate_with_api(self, text: str, source_lang: str, target_lang: str, api_key: str) -> Dict[str, Any]:
        """
        Translate using a commercial translation API.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            api_key: API key for the service

        Returns:
            Translation result
        """
        # This is a generic implementation for demonstration
        # In a real-world scenario, you would implement a specific API like Google Translate, DeepL, etc.

        api_url = "https://translation-api-example.com/translate"

        payload = {
            "text": text,
            "source": source_lang,
            "target": target_lang,
            "api_key": api_key
        }

        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()

            data = response.json()

            if "translatedText" in data:
                return {
                    "status": "success",
                    "translated_text": data["translatedText"],
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "method": "api"
                }
            else:
                return {
                    "status": "error",
                    "error": "API response missing translatedText field",
                    "method": "api"
                }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"API translation request failed: {str(e)}",
                "method": "api"
            }

    def _translate_with_llm(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Translate using LLM-based translation.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translation result
        """
        if not self.llm_translator:
            return {
                "status": "error",
                "error": "LLM translator not available",
                "method": "llm"
            }

        try:
            # Create a translation prompt
            prompt = f"""Translate the following text from {source_lang} to {target_lang}:

            {text}

            Translation:"""

            # Call the LLM translator function
            translation = self.llm_translator(prompt)

            if translation:
                return {
                    "status": "success",
                    "translated_text": translation,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "method": "llm"
                }
            else:
                return {
                    "status": "error",
                    "error": "LLM translator returned empty result",
                    "method": "llm"
                }

        except Exception as e:
            return {
                "status": "error",
                "error": f"LLM translation failed: {str(e)}",
                "method": "llm"
            }

    def _translate_with_libre(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Translate using LibreTranslate API (open source).

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translation result
        """
        api_url = "https://libretranslate.de/translate"

        # LibreTranslate doesn't support auto-detection in API
        if source_lang == "auto":
            # Detect language first
            detected = self._libre_detect_language(text)
            source_lang = detected.get("detected_lang", "en")

        payload = {
            "q": text,
            "source": source_lang,
            "target": target_lang
        }

        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()

            data = response.json()

            if "translatedText" in data:
                return {
                    "status": "success",
                    "translated_text": data["translatedText"],
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "method": "libretranslate"
                }
            else:
                return {
                    "status": "error",
                    "error": "LibreTranslate response missing translatedText field",
                    "method": "libretranslate"
                }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"LibreTranslate request failed: {str(e)}",
                "method": "libretranslate"
            }

    def _libre_detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language using LibreTranslate API.

        Args:
            text: Text to analyze

        Returns:
            Detection result
        """
        api_url = "https://libretranslate.de/detect"

        payload = {
            "q": text
        }

        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()

            data = response.json()

            if data and isinstance(data, list) and len(data) > 0:
                return {
                    "status": "success",
                    "detected_lang": data[0]["language"],
                    "confidence": data[0]["confidence"],
                    "method": "libretranslate"
                }
            else:
                return {
                    "status": "error",
                    "error": "Invalid response from LibreTranslate detect API",
                    "detected_lang": "en",
                    "confidence": 0,
                    "method": "libretranslate"
                }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"LibreTranslate language detection failed: {str(e)}",
                "detected_lang": "en",
                "confidence": 0,
                "method": "libretranslate"
            }

    def _pattern_detect_language(self, text: str) -> Dict[str, Any]:
        """
        Simple pattern-based language detection.

        Args:
            text: Text to analyze

        Returns:
            Detection result
        """
        # Define language patterns
        patterns = {
            "en": (r'\b(the|and|is|in|to|of|that|for|it|with)\b', 0.05),  # English
            "es": (r'\b(el|la|de|en|que|y|a|los|por|con)\b', 0.05),       # Spanish
            "fr": (r'\b(le|la|de|et|est|en|que|une|pour|qui)\b', 0.05),   # French
            "de": (r'\b(der|die|und|ist|von|zu|das|mit|sich|auf)\b', 0.05) # German
        }

        best_lang = "en"
        best_score = 0

        for lang, (pattern, weight) in patterns.items():
            matches = re.findall(pattern, text.lower())
            score = len(matches) * weight / (len(text.split()) + 1)  # Normalize by word count

            if score > best_score:
                best_score = score
                best_lang = lang

        # Check for specific character sets
        if re.search(r'[\u4e00-\u9fff]', text):  # Chinese characters
            return {"status": "success", "detected_lang": "zh", "confidence": 0.9, "method": "pattern"}

        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):  # Japanese characters
            return {"status": "success", "detected_lang": "ja", "confidence": 0.9, "method": "pattern"}

        if re.search(r'[\uAC00-\uD7AF]', text):  # Korean characters
            return {"status": "success", "detected_lang": "ko", "confidence": 0.9, "method": "pattern"}

        if re.search(r'[\u0600-\u06FF]', text):  # Arabic characters
            return {"status": "success", "detected_lang": "ar", "confidence": 0.9, "method": "pattern"}

        if re.search(r'[\u0400-\u04FF]', text):  # Cyrillic (Russian) characters
            return {"status": "success", "detected_lang": "ru", "confidence": 0.9, "method": "pattern"}

        # Return best match from word patterns
        return {
            "status": "success",
            "detected_lang": best_lang,
            "confidence": min(best_score * 10, 0.7),  # Cap confidence at 0.7 for pattern matching
            "method": "pattern"
        }

    def _simple_translation(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Very basic pattern-based translation for common phrases.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translation result
        """
        # This is a fallback method with extremely limited capabilities
        # Only useful for a small set of common phrases

        # Only implement English to Spanish and reverse as example
        translations = {
            "en_es": {
                "hello": "hola",
                "goodbye": "adiós",
                "thank you": "gracias",
                "how are you": "cómo estás",
                "my name is": "me llamo",
                "what time is it": "qué hora es",
                "good morning": "buenos días",
                "good afternoon": "buenas tardes",
                "good evening": "buenas noches",
                "please": "por favor",
                "sorry": "lo siento"
            },
            "es_en": {
                "hola": "hello",
                "adiós": "goodbye",
                "gracias": "thank you",
                "cómo estás": "how are you",
                "me llamo": "my name is",
                "qué hora es": "what time is it",
                "buenos días": "good morning",
                "buenas tardes": "good afternoon",
                "buenas noches": "good evening",
                "por favor": "please",
                "lo siento": "sorry"
            }
        }

        # Check if we have a translation dictionary for the language pair
        translation_key = f"{source_lang}_{target_lang}"
        if translation_key not in translations:
            # Try to use English as an intermediate language
            text_in_english = text

            if source_lang != "en":
                # Try to translate to English first using pattern matching
                src_to_en_key = f"{source_lang}_en"
                if src_to_en_key in translations:
                    for src, tgt in translations[src_to_en_key].items():
                        text_in_english = re.sub(
                            r'\b' + re.escape(src) + r'\b',
                            tgt,
                            text_in_english,
                            flags=re.IGNORECASE
                        )

            # Then translate from English to target
            if target_lang != "en":
                en_to_tgt_key = f"en_{target_lang}"
                if en_to_tgt_key in translations:
                    translated_text = text_in_english
                    for src, tgt in translations[en_to_tgt_key].items():
                        translated_text = re.sub(
                            r'\b' + re.escape(src) + r'\b',
                            tgt,
                            translated_text,
                            flags=re.IGNORECASE
                        )

                    return {
                        "status": "partial",
                        "translated_text": translated_text,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "method": "pattern",
                        "warning": "Limited phrase translation only. Result may be inaccurate."
                    }

            # If we can't use English as intermediate, return failure
            return {
                "status": "error",
                "error": f"No translation available for {source_lang} to {target_lang}",
                "method": "pattern"
            }

        # Direct translation using our dictionary
        translated_text = text
        for src, tgt in translations[translation_key].items():
            translated_text = re.sub(
                r'\b' + re.escape(src) + r'\b',
                tgt,
                translated_text,
                flags=re.IGNORECASE
            )

        # See if we actually changed anything
        if translated_text != text:
            return {
                "status": "partial",
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "method": "pattern",
                "warning": "Limited phrase translation only. Result may be inaccurate."
            }
        else:
            return {
                "status": "error",
                "error": "Could not translate text with available patterns",
                "method": "pattern"
            }

# Create a translator instance
translator = Translator()

def translator_tool_handler(user_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle translation requests.

    Args:
        user_input: Parsed user input
        context: Session context

    Returns:
        Translation result
    """
    raw_input = user_input.get("raw_input", "")

    # Extract the text to translate
    # Look for "translate" or "translation" patterns
    translate_match = re.search(
        r'translat(?:e|ion)(?:[:\s]+)(?:of\s+)?["\']?([^"\']+)["\']?',
        raw_input,
        re.IGNORECASE
    )

    text_to_translate = None
    source_lang = "auto"
    target_lang = "en"

    if translate_match:
        text_to_translate = translate_match.group(1).strip()
    else:
        # Look for "to {language}" pattern
        to_lang_match = re.search(
            r'(?:in|to|into)\s+([a-z]+)(?:\s+language)?',
            raw_input,
            re.IGNORECASE
        )

        if to_lang_match:
            target_lang = to_lang_match.group(1).strip().lower()
            # Remove the "to {language}" part to get the text
            text = re.sub(
                r'(?:translate|translation)(?:\s+|:)\s*',
                '',
                raw_input,
                flags=re.IGNORECASE
            )
            text = re.sub(
                r'(?:in|to|into)\s+([a-z]+)(?:\s+language)?',
                '',
                text,
                flags=re.IGNORECASE
            )
            text_to_translate = text.strip()
        else:
            # Fallback: just remove "translate" and use the rest
            text = re.sub(
                r'(?:translate|translation)(?:\s+|:)\s*',
                '',
                raw_input,
                flags=re.IGNORECASE
            )
            text_to_translate = text.strip()

    # Look for "from {language}" pattern
    from_lang_match = re.search(
        r'from\s+([a-z]+)(?:\s+language)?',
        raw_input,
        re.IGNORECASE
    )

    if from_lang_match:
        source_lang = from_lang_match.group(1).strip().lower()

    # Look for "to {language}" pattern if not found above
    if target_lang == "en":
        to_lang_match = re.search(
            r'(?:in|to|into)\s+([a-z]+)(?:\s+language)?',
            raw_input,
            re.IGNORECASE
        )

        if to_lang_match:
            target_lang = to_lang_match.group(1).strip().lower()

    # Get API key from context if available
    translation_settings = context.get("preferences", {}).get("translation", {})
    api_key = translation_settings.get("api_key", None)

    # Check if we actually have something to translate
    if not text_to_translate or len(text_to_translate) < 1:
        return {
            "status": "error",
            "content": "Could not identify text to translate. Please specify the text clearly.",
            "error": "No text to translate"
        }

    # Perform the translation
    result = translator.translate_text(
        text=text_to_translate,
        source_lang=source_lang,
        target_lang=target_lang,
        api_key=api_key
    )

    # Format the response
    if result["status"] == "success":
        response = {
            "status": "success",
            "content": f"Translation ({result['source_lang']} → {result['target_lang']}):\n\n{result['translated_text']}",
            "translated_text": result['translated_text'],
            "source_lang": result['source_lang'],
            "target_lang": result['target_lang'],
            "method": result.get("method", "unknown")
        }
    elif result["status"] == "partial":
        response = {
            "status": "partial",
            "content": f"Partial translation ({result['source_lang']} → {result['target_lang']}):\n\n{result['translated_text']}\n\nWarning: {result.get('warning', 'Translation may be incomplete.')}",
            "translated_text": result['translated_text'],
            "source_lang": result['source_lang'],
            "target_lang": result['target_lang'],
            "method": result.get("method", "unknown"),
            "warning": result.get("warning")
        }
    else:
        response = {
            "status": "error",
            "content": f"Translation failed: {result.get('error', 'Unknown error')}",
            "error": result.get("error")
        }

    return response

def detect_language_handler(user_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle language detection requests.

    Args:
        user_input: Parsed user input
        context: Session context

    Returns:
        Language detection result
    """
    raw_input = user_input.get("raw_input", "")

    # Extract the text to analyze
    detect_match = re.search(
        r'(?:detect|identify)\s+(?:language|what language)(?:[:\s]+)(?:of\s+)?["\']?([^"\']+)["\']?',
        raw_input,
        re.IGNORECASE
    )

    if detect_match:
        text_to_analyze = detect_match.group(1).strip()
    else:
        # Fallback: remove the "detect language" part
        text = re.sub(
            r'(?:detect|identify)\s+(?:language|what language)(?:[:\s]+)(?:of\s+)?',
            '',
            raw_input,
            flags=re.IGNORECASE
        )
        text_to_analyze = text.strip()

    # Check if we have text to analyze
    if not text_to_analyze or len(text_to_analyze) < 1:
        return {
            "status": "error",
            "content": "Could not identify text for language detection. Please specify the text clearly.",
            "error": "No text for language detection"
        }

    # Detect the language
    result = translator.detect_language(text_to_analyze)

    # Format the response
    if result["status"] == "success":
        # Get the full language name
        lang_code = result["detected_lang"]
        lang_name = "Unknown"

        for name, code in translator.language_codes.items():
            if code == lang_code:
                lang_name = name.capitalize()
                break

        confidence = result.get("confidence", 0) * 100

        response = {
            "status": "success",
            "content": f"Detected language: {lang_name} ({lang_code}) with {confidence:.1f}% confidence",
            "detected_lang": lang_code,
            "language_name": lang_name,
            "confidence": result.get("confidence", 0),
            "method": result.get("method", "unknown")
        }
    else:
        response = {
            "status": "error",
            "content": f"Language detection failed: {result.get('error', 'Unknown error')}",
            "error": result.get("error")
        }

    return response

if __name__ == "__main__":
    # Test the translator
    test_text = "Hello, how are you today?"

    # Test translation
    result = translator.translate_text(test_text, "en", "es")
    print(f"Translation: {result}")

    # Test language detection
    result = translator.detect_language(test_text)
    print(f"Language detection: {result}")
