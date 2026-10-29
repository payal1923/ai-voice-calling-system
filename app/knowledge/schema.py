"""
app/knowledge/schema.py
Standardized Pydantic v2 schema for Universal Multi-Business Voice Agent profiles.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class BusinessMetadata(BaseModel):
    """Core tenant identification and categorization."""
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    company_id: str = Field(
        ...,
        description="Unique URL-safe and filesystem-safe identifier (e.g. 'pullman_aerocity', 'apex_car_rental')",
        min_length=3,
        max_length=64,
    )
    company_name: str = Field(
        ...,
        description="Human-readable commercial name of the business",
        min_length=2,
        max_length=128,
    )
    industry: str = Field(
        default="general",
        description="Primary industry sector: e.g. 'hospitality', 'automotive_rental', 'healthcare', 'general'",
        min_length=2,
        max_length=64,
    )
    brand: Optional[str] = Field(
        default=None,
        description="Parent brand or group affiliation (e.g. 'Accor Hotels')",
    )
    description: Optional[str] = Field(
        default=None,
        description="Concise description of the business for agent grounding",
    )

    @field_validator("company_id")
    @classmethod
    def validate_company_id_slug(cls, v: str) -> str:
        clean = v.strip().lower()
        if not re.match(r"^[a-z0-9_-]+$", clean):
            raise ValueError(f"company_id '{v}' must be alphanumeric with underscores or hyphens only (^[a-z0-9_-]+$)")
        return clean


class LocationConfig(BaseModel):
    """Physical address, geographic context, and operational time zone."""
    model_config = ConfigDict(extra="ignore")

    address: str = Field(..., description="Full physical postal address")
    proximity: Optional[str] = Field(default=None, description="Nearby landmarks, airports, or transit hubs")
    city: Optional[str] = Field(default=None, description="City of operation")
    country: Optional[str] = Field(default="India", description="Country of operation")
    timezone: str = Field(default="Asia/Kolkata", description="IANA timezone name")


class OperatingHoursConfig(BaseModel):
    """Operating schedule and 24/7 availability markers."""
    model_config = ConfigDict(extra="ignore")

    is_24_7: bool = Field(default=False, description="Whether the business or reception operates 24/7")
    regular_hours: Dict[str, str] = Field(
        default_factory=dict,
        description="Day-of-week schedule mapping",
    )
    special_notes: Optional[str] = Field(default=None, description="Exceptions, holidays, or emergency contact hours")


class VoiceLanguageConfig(BaseModel):
    """Audio synthesis configuration and language preferences."""
    model_config = ConfigDict(extra="ignore")

    default_language: str = Field(default="en-US", description="Primary BCP-47 language tag")
    supported_languages: List[str] = Field(default_factory=lambda: ["en-US"], description="Supported language codes")
    tts_voice_id: str = Field(default="en_US-lessac-medium", description="Piper ONNX voice model identifier")
    speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0, description="TTS playback speed multiplier")


class ConversationalPreferences(BaseModel):
    """Greetings, fallback responses, and LLM system prompt instructions."""
    model_config = ConfigDict(extra="ignore")

    greeting_message: str = Field(
        ...,
        description="Initial voice greeting spoken by the agent upon call connection",
        min_length=5,
    )
    system_instruction: str = Field(
        ...,
        description="Foundational system persona and behavioral guidelines passed to the LLM",
        min_length=20,
    )
    fallback_message: str = Field(
        default="I apologize, but I did not catch that clearly. Could you please repeat your question?",
        description="Standard fallback spoken when speech cannot be interpreted",
    )
    clarification_message: str = Field(
        default="I want to make sure I assist you correctly. Could you please clarify what you need?",
        description="Polite prompt requesting caller clarification",
    )
    max_response_tokens: int = Field(default=120, ge=20, le=512, description="Upper bound on LLM response length")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0, description="LLM sampling temperature for grounding")


class AcousticNormalizationRule(BaseModel):
    """Declarative regex acoustic/phonetic mistranscription correction rule."""
    model_config = ConfigDict(extra="ignore")

    rule_id: str = Field(..., description="Unique descriptive identifier for the rule")
    pattern: str = Field(..., description="Regex pattern matching the acoustic mistranscription")
    replacement: str = Field(..., description="Target clean transcript replacement")
    case_sensitive: bool = Field(default=False, description="Whether pattern matching is case-sensitive")
    description: Optional[str] = Field(default=None, description="Explanation of why this acoustic error occurs")

    @field_validator("pattern")
    @classmethod
    def validate_regex_pattern(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as err:
            raise ValueError(f"Invalid regex pattern '{v}': {err}")
        return v


class SpeechRecognitionConfig(BaseModel):
    """Faster-Whisper STT vocabulary biasing and tenant acoustic corrections."""
    model_config = ConfigDict(extra="ignore")

    stt_prompt: str = Field(
        ...,
        description="Domain-anchored initial prompt biasing Faster-Whisper decoder toward industry vocabulary",
    )
    vocabulary_boost: List[str] = Field(
        default_factory=list,
        description="Key proper nouns, venue names, or brand terms for STT guidance",
    )
    acoustic_normalization_rules: List[AcousticNormalizationRule] = Field(
        default_factory=list,
        description="Tenant-specific acoustic phonetic mistranscription rewrite rules",
    )


class KnowledgeFact(BaseModel):
    """Individual atomic knowledge fact used for grounded retrieval and LLM context."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique fact identifier")
    category: str = Field(..., description="Topical classification")
    keywords: List[str] = Field(..., min_length=1, description="Caller query triggers for scoring relevance")
    fact: str = Field(..., min_length=5, description="Approved, factual canonical statement")
    priority: int = Field(default=1, ge=1, le=10, description="Ranking boost priority")
    followup_cues: List[str] = Field(
        default_factory=list,
        description="Phrases indicating caller is asking a follow-up question regarding this fact",
    )
    negation_patterns: List[str] = Field(
        default_factory=list,
        description="Phrases indicating caller explicitly does NOT want this topic",
    )

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, v: List[str]) -> List[str]:
        cleaned = [k.strip().lower() for k in v if k and k.strip()]
        if not cleaned:
            raise ValueError("Fact must contain at least one valid keyword")
        return cleaned


class FAQItem(BaseModel):
    """Standardized frequently asked question pair."""
    model_config = ConfigDict(extra="ignore")

    question: str = Field(..., min_length=5, description="Common caller question phrasing")
    answer: str = Field(..., min_length=5, description="Concise, verified spoken answer")
    category: Optional[str] = Field(default="general")


class KnowledgeBaseConfig(BaseModel):
    """Collection of facts, FAQs, and retrieval parameters."""
    model_config = ConfigDict(extra="ignore")

    facts: List[KnowledgeFact] = Field(..., min_length=1, description="Atomic factual knowledge items")
    faqs: List[FAQItem] = Field(default_factory=list, description="Common Q&A pairs")

    @model_validator(mode="after")
    def verify_unique_fact_ids(self) -> KnowledgeBaseConfig:
        ids = [f.id for f in self.facts]
        if len(ids) != len(set(ids)):
            duplicates = [x for x in ids if ids.count(x) > 1]
            raise ValueError(f"Duplicate fact IDs detected in knowledge base: {set(duplicates)}")
        return self


class BusinessPolicies(BaseModel):
    """Standardized operational rules, guarantees, and legal disclosures."""
    model_config = ConfigDict(extra="ignore")

    cancellation_policy: Optional[str] = Field(default=None)
    payment_methods: List[str] = Field(default_factory=lambda: ["Credit Card", "Debit Card", "UPI", "Cash"])
    identification_requirements: Optional[str] = Field(default=None)
    custom_policies: Dict[str, str] = Field(
        default_factory=dict,
        description="Industry-specific policy items",
    )


class DomainExtensionConfig(BaseModel):
    """Industry-specific domain plugin configuration."""
    model_config = ConfigDict(extra="ignore")

    domain_module: str = Field(
        default="general",
        description="Name of the domain handler module",
    )
    deterministic_handlers_enabled: bool = Field(
        default=True,
        description="Whether deterministic fast-path handlers are evaluated before LLM fallback",
    )
    custom_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary domain-specific parameters consumed by domain plugins",
    )


class BusinessProfile(BaseModel):
    """
    Top-Level Universal Business Profile Model.
    Encompasses identity, speech configuration, grounded knowledge, and domain extensions.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    metadata: BusinessMetadata = Field(..., description="Core business identification")
    location: Optional[LocationConfig] = Field(default=None, description="Physical location and time zone")
    operating_hours: OperatingHoursConfig = Field(
        default_factory=OperatingHoursConfig,
        description="Business hours and 24/7 status",
    )
    voice_and_language: VoiceLanguageConfig = Field(
        default_factory=VoiceLanguageConfig,
        description="Voice synthesis and language preferences",
    )
    conversational_preferences: ConversationalPreferences = Field(
        ...,
        description="Greetings, persona, and response parameters",
    )
    speech_recognition: SpeechRecognitionConfig = Field(
        ...,
        description="Faster-Whisper vocabulary biasing and phonetic rules",
    )
    knowledge_base: KnowledgeBaseConfig = Field(
        ...,
        description="Grounded facts and Q&A pairs",
    )
    policies: BusinessPolicies = Field(
        default_factory=BusinessPolicies,
        description="Operational and legal policies",
    )
    domain_extensions: Optional[DomainExtensionConfig] = Field(
        default=None,
        description="Pluggable domain logic configuration",
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BusinessProfile:
        """
        Factory method that parses either a modern structured profile dict
        or an existing legacy flat profile dict (like pullman_aerocity.json).
        """
        if "metadata" in data and "knowledge_base" in data:
            return cls.model_validate(data)
        return cls._from_legacy_dict(data)

    @classmethod
    def _from_legacy_dict(cls, data: Dict[str, Any]) -> BusinessProfile:
        """Adapts legacy flat JSON profiles into the standardized BusinessProfile."""
        company_id = data.get("company_id", "pullman_aerocity")
        company_name = data.get("company_name", "Pullman New Delhi Aerocity")
        brand = data.get("brand")
        industry = data.get("industry", "hospitality")

        # Location
        loc_data = data.get("location")
        loc_model = None
        if isinstance(loc_data, dict):
            loc_model = LocationConfig(
                address=loc_data.get("address", "New Delhi, India"),
                proximity=loc_data.get("proximity"),
                city=loc_data.get("city", "New Delhi"),
                country=loc_data.get("country", "India"),
                timezone=loc_data.get("timezone", "Asia/Kolkata"),
            )

        # Conversational preferences
        prefs = ConversationalPreferences(
            greeting_message=data.get("greeting_message") or "Hello, thank you for calling. How may I assist you today?",
            system_instruction=data.get("system_instruction") or "You are a helpful voice assistant.",
            fallback_message=data.get("fallback_message") or "I apologize, but I did not catch that clearly. Could you please repeat your question?",
            clarification_message=data.get("clarification_message") or "I want to make sure I assist you correctly. Could you please clarify what you need?",
        )

        # Speech recognition
        stt_prompt = data.get("stt_prompt") or f"{company_name} front desk conversation."
        speech = SpeechRecognitionConfig(
            stt_prompt=stt_prompt,
            vocabulary_boost=data.get("vocabulary_boost", []),
            acoustic_normalization_rules=[],
        )

        # Knowledge base
        raw_facts = data.get("facts", [])
        kb_facts: List[KnowledgeFact] = []
        for idx, f in enumerate(raw_facts):
            f_id = f.get("id") or f"fact_{idx}"
            f_cat = f.get("category") or "general"
            f_kws = f.get("keywords") or [f_id]
            f_text = f.get("fact") or ""
            kb_facts.append(KnowledgeFact(
                id=f_id,
                category=f_cat,
                keywords=f_kws,
                fact=f_text,
                priority=f.get("priority", 1),
                followup_cues=f.get("followup_cues", []),
                negation_patterns=f.get("negation_patterns", []),
            ))

        if not kb_facts:
            kb_facts.append(KnowledgeFact(
                id="default_fact",
                category="general",
                keywords=["general"],
                fact=f"{company_name} is dedicated to serving our valued customers.",
            ))

        kb = KnowledgeBaseConfig(facts=kb_facts, faqs=[])

        # Metadata
        meta = BusinessMetadata(
            company_id=company_id,
            company_name=company_name,
            industry=industry,
            brand=brand,
            description=data.get("description"),
        )

        # Domain extensions
        dom = DomainExtensionConfig(
            domain_module="hospitality" if industry == "hospitality" else "general",
            deterministic_handlers_enabled=True,
        )

        return cls(
            metadata=meta,
            location=loc_model,
            operating_hours=OperatingHoursConfig(is_24_7=True),
            voice_and_language=VoiceLanguageConfig(),
            conversational_preferences=prefs,
            speech_recognition=speech,
            knowledge_base=kb,
            policies=BusinessPolicies(),
            domain_extensions=dom,
        )

