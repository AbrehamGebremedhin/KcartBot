"""Multilingual testing tool for validating language detection, translation, and response formatting."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.tools.base import ToolBase
from app.utils.language_utils import LanguageDetector, TranslationService, MultilingualResponseFormatter, Language

logger = logging.getLogger(__name__)


class MultilingualTestingTool(ToolBase):
    """Tool for comprehensive testing of multilingual functionality."""

    def __init__(
        self,
        llm_service: Optional["LLMService"] = None,
    ) -> None:
        super().__init__(
            name="multilingual_testing",
            description=(
                "Test and validate multilingual functionality including language detection, "
                "translation services, and response formatting. Use this to ensure the system "
                "properly handles Amharic, phonetic Amharic, and English inputs/outputs."
            ),
        )

        # Initialize language utilities
        self.language_detector = LanguageDetector()

        if llm_service:
            self._llm = llm_service
            self.translation_service = TranslationService(llm_service)
            self.response_formatter = MultilingualResponseFormatter(self.translation_service)
        else:
            from app.services.llm_service import LLMService
            self._llm = LLMService()
            self.translation_service = TranslationService(self._llm)
            self.response_formatter = MultilingualResponseFormatter(self.translation_service)

        if llm_service:
            self._llm = llm_service
        else:
            from app.services.llm_service import LLMService
            self._llm = LLMService()

    async def run(self, input: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run comprehensive multilingual testing."""
        test_type = input.get("test_type", "comprehensive") if isinstance(input, dict) else "comprehensive"

        if test_type == "language_detection":
            return await self._test_language_detection(input)
        elif test_type == "translation":
            return await self._test_translation(input)
        elif test_type == "response_formatting":
            return await self._test_response_formatting(input)
        elif test_type == "comprehensive":
            return await self._run_comprehensive_tests(input)
        else:
            return {
                "error": f"Unknown test type: {test_type}",
                "available_types": ["language_detection", "translation", "response_formatting", "comprehensive"]
            }

    async def _test_language_detection(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Test language detection with various inputs."""
        test_texts = input.get("test_texts", [
            "Hello, I want to buy tomatoes",
            "ሰላም፣ ቲማቲም መግዛት እፈልጋለሁ",
            "selam, timatim megzat efelgalew",
            "I need storage advice for apples",
            "አብሎን ለማስቀመጥ እንጂ እፈልጋለሁ",
            "ablon lemaskemteg enji efelgalew"
        ])

        results = []
        for text in test_texts:
            detected_lang = self.language_detector.detect_language(text)
            results.append({
                "text": text,
                "detected_language": str(detected_lang)  # Convert enum to string
            })

        # Use LLM to validate results
        validation_prompt = f"""
        Analyze these language detection results and provide feedback:

        Test Results:
        {results}

        Please evaluate:
        1. Are the detections accurate?
        2. Any patterns or issues you notice?
        3. Suggestions for improvement?

        Respond with a JSON object containing:
        - accuracy_score: float between 0-1
        - issues: list of any problems found
        - suggestions: list of improvement suggestions
        - overall_assessment: brief summary
        """

        try:
            llm_response = await self._llm.acomplete(validation_prompt)
            validation = self._parse_llm_validation(llm_response)
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            validation = {
                "accuracy_score": 0.0,
                "issues": [f"LLM validation error: {str(e)}"],
                "suggestions": ["Check LLM service configuration"],
                "overall_assessment": "Unable to validate with LLM"
            }

        return {
            "test_type": "language_detection",
            "results": results,
            "validation": validation,
            "total_tests": len(results)
        }

    async def _test_translation(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Test translation services."""
        test_phrases = input.get("test_phrases", [
            "Welcome to our store!",
            "How can I help you?",
            "What products are available?",
            "Thank you for your order",
            "Please provide your location"
        ])

        results = []
        for phrase in test_phrases:
            try:
                translated = await self.translation_service.translate_to_amharic(phrase)
                results.append({
                    "original": phrase,
                    "translated": translated,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "original": phrase,
                    "translated": None,
                    "success": False,
                    "error": str(e)
                })

        # Use LLM to evaluate translations
        successful_translations = [r for r in results if r["success"]]
        validation_prompt = f"""
        Evaluate these English to Amharic translations:

        Translations:
        {successful_translations}

        For each translation, assess:
        1. Is the translation accurate?
        2. Is the Amharic natural and correct?
        3. Any cultural or contextual issues?

        Provide a JSON response with:
        - quality_score: average quality (0-1)
        - translation_quality: list of quality assessments
        - issues_found: any problems identified
        - recommendations: suggestions for improvement
        """

        try:
            llm_response = await self._llm.acomplete(validation_prompt)
            validation = self._parse_llm_validation(llm_response)
        except Exception as e:
            validation = {
                "quality_score": 0.0,
                "issues_found": [f"LLM evaluation error: {str(e)}"],
                "recommendations": ["Check LLM service"]
            }

        return {
            "test_type": "translation",
            "results": results,
            "validation": validation,
            "success_rate": len([r for r in results if r["success"]]) / len(results)
        }

    async def _test_response_formatting(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Test response formatting with different languages."""
        test_scenarios = input.get("test_scenarios", [
            {
                "response": "Welcome! How can I help you today?",
                "detected_language": Language.ENGLISH,
                "preferred_language": Language.ENGLISH
            },
            {
                "response": "Welcome! How can I help you today?",
                "detected_language": Language.AMHARIC,
                "preferred_language": Language.AMHARIC
            },
            {
                "response": "Product added to your cart",
                "detected_language": Language.AMHARIC_LATIN,
                "preferred_language": Language.AMHARIC
            }
        ])

        results = []
        for scenario in test_scenarios:
            try:
                formatted = await self.response_formatter.format_response(
                    scenario["response"],
                    scenario["detected_language"],
                    scenario["preferred_language"]
                )
                results.append({
                    "scenario": {
                        "response": scenario["response"],
                        "detected_language": str(scenario["detected_language"]),
                        "preferred_language": str(scenario["preferred_language"])
                    },
                    "formatted_response": formatted,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "scenario": {
                        "response": scenario["response"],
                        "detected_language": str(scenario["detected_language"]),
                        "preferred_language": str(scenario["preferred_language"])
                    },
                    "formatted_response": None,
                    "success": False,
                    "error": str(e)
                })

        return {
            "test_type": "response_formatting",
            "results": results,
            "success_rate": len([r for r in results if r["success"]]) / len(results)
        }

    async def _run_comprehensive_tests(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Run all multilingual tests comprehensively."""
        print("Running comprehensive multilingual tests...")

        # Run all test types
        detection_results = await self._test_language_detection({})
        translation_results = await self._test_translation({})
        formatting_results = await self._test_response_formatting({})

        # Generate comprehensive report using LLM
        report_prompt = f"""
You are a technical testing assistant analyzing multilingual system performance. This is NOT a customer service interaction.

Based on the following test results, generate a comprehensive technical report:

Language Detection Results:
- Total tests: {detection_results['total_tests']}
- Accuracy score: {detection_results.get('validation', {}).get('accuracy_score', 'N/A')}

Translation Results:
- Success rate: {translation_results['success_rate']:.2f}
- Quality score: {translation_results.get('validation', {}).get('quality_score', 'N/A')}

Response Formatting Results:
- Success rate: {formatting_results['success_rate']:.2f}

Provide a JSON report with:
- overall_health_score: float 0-1
- critical_issues: list of critical problems
- recommendations: prioritized list of fixes
- readiness_assessment: "ready", "needs_work", or "critical_issues"
- summary: brief overall assessment

IMPORTANT: Respond ONLY with valid JSON. Do not include any conversational text, greetings, or marketplace assistance.
"""

        try:
            llm_response = await self._llm.acomplete(report_prompt)
            report = self._parse_llm_validation(llm_response)
        except Exception as e:
            report = {
                "overall_health_score": 0.5,
                "critical_issues": [f"Report generation failed: {str(e)}"],
                "recommendations": ["Fix LLM service integration"],
                "readiness_assessment": "needs_work",
                "summary": "Unable to generate comprehensive report"
            }

        return {
            "test_type": "comprehensive",
            "language_detection": detection_results,
            "translation": translation_results,
            "response_formatting": formatting_results,
            "comprehensive_report": report,
            "timestamp": "2025-10-16"  # Current date
        }

    def _parse_llm_validation(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM validation response."""
        try:
            # Try to extract JSON from response
            import json
            import re

            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"error": "No JSON found in LLM response", "raw_response": llm_response}
        except Exception as e:
            return {"error": f"Failed to parse LLM response: {str(e)}", "raw_response": llm_response}