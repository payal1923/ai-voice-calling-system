import os
import json
import re
from typing import Dict, Any, List, Optional
from app.knowledge.schema import BusinessProfile

_KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge", "companies")
_CACHE: Dict[str, Dict[str, Any]] = {}
_PROFILE_CACHE: Dict[str, BusinessProfile] = {}


class KnowledgeService:
    """
    Lightweight, in-memory knowledge retrieval service for company-specific voice profiles.
    Loads approved facts and personas from JSON profiles, performing fast keyword-anchored
    fact retrieval to ground LLM voice responses without external vector databases or migrations.
    Supports validated BusinessProfile schema with two-way legacy backward compatibility.
    """

    def __init__(self, knowledge_dir: str = _KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir

    def load_company_profile(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Loads and caches the company knowledge profile from disk as a dict."""
        if not company_id:
            return None

        clean_id = str(company_id).strip().lower()
        if not re.match(r"^[a-z0-9_-]+$", clean_id):
            return None

        if clean_id in _CACHE:
            return _CACHE[clean_id]

        file_path = os.path.join(self.knowledge_dir, f"{clean_id}.json")
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Two-way compatibility: ensure legacy flat keys exist if loaded from modern schema
            if "metadata" in data and isinstance(data["metadata"], dict):
                meta = data["metadata"]
                if "company_id" not in data:
                    data["company_id"] = meta.get("company_id")
                if "company_name" not in data:
                    data["company_name"] = meta.get("company_name")
                if "brand" not in data:
                    data["brand"] = meta.get("brand")
            if "conversational_preferences" in data and isinstance(data["conversational_preferences"], dict):
                prefs = data["conversational_preferences"]
                if "greeting_message" not in data:
                    data["greeting_message"] = prefs.get("greeting_message")
                if "system_instruction" not in data:
                    data["system_instruction"] = prefs.get("system_instruction")
            if "speech_recognition" in data and isinstance(data["speech_recognition"], dict):
                speech = data["speech_recognition"]
                if "stt_prompt" not in data:
                    data["stt_prompt"] = speech.get("stt_prompt")
            if "knowledge_base" in data and isinstance(data["knowledge_base"], dict):
                kb = data["knowledge_base"]
                if "facts" not in data:
                    data["facts"] = kb.get("facts", [])

            _CACHE[clean_id] = data
            return data
        except Exception as e:
            print(f"[KNOWLEDGE_SERVICE ERROR] Failed loading profile for '{clean_id}': {e}")
            return None

    def load_profile(self, company_id: str) -> Optional[BusinessProfile]:
        """Loads, validates, and caches the standardized BusinessProfile model."""
        if not company_id:
            return None

        clean_id = str(company_id).strip().lower()
        if not re.match(r"^[a-z0-9_-]+$", clean_id):
            return None

        if clean_id in _PROFILE_CACHE:
            return _PROFILE_CACHE[clean_id]

        raw_data = self.load_company_profile(clean_id)
        if not raw_data:
            return None

        try:
            profile = BusinessProfile.from_dict(raw_data)
            _PROFILE_CACHE[clean_id] = profile
            return profile
        except Exception as e:
            print(f"[KNOWLEDGE_SERVICE ERROR] Failed validating BusinessProfile for '{clean_id}': {e}")
            return None

    def get_acoustic_rules(self, company_id: Optional[str]) -> List[Any]:
        """Returns the acoustic normalization rules for the given company profile."""
        if not company_id:
            return []
        clean_id = str(company_id).strip().lower()
        profile = self.load_profile(clean_id)
        if profile and profile.speech_recognition:
            return profile.speech_recognition.acoustic_normalization_rules or []
        return []

    def clear_cache(self, company_id: Optional[str] = None):
        """Clears in-memory profile and raw dictionary caches."""
        global _CACHE, _PROFILE_CACHE
        if company_id:
            clean_id = str(company_id).strip().lower()
            _CACHE.pop(clean_id, None)
            _PROFILE_CACHE.pop(clean_id, None)
        else:
            _CACHE.clear()
            _PROFILE_CACHE.clear()

    def reload_profile(self, company_id: str) -> Optional[BusinessProfile]:
        """Evicts and reloads the profile for the given company from disk."""
        self.clear_cache(company_id)
        return self.load_profile(company_id)

    def get_system_instruction(self, company_id: str) -> Optional[str]:
        """Returns the system instruction / persona prompt for the company."""
        profile = self.load_company_profile(company_id)
        if profile:
            return profile.get("system_instruction")
        return None

    def get_greeting(self, company_id: str) -> Optional[str]:
        """Returns the custom opening greeting for the company."""
        profile = self.load_company_profile(company_id)
        if profile:
            return profile.get("greeting_message")
        return None

    def get_company_name(self, company_id: str) -> Optional[str]:
        """Returns the human-readable company name."""
        profile = self.load_company_profile(company_id)
        if profile:
            return profile.get("company_name")
        return None

    def get_stt_prompt(self, company_id: Optional[str] = None) -> str:
        """
        Returns a domain-anchored Faster-Whisper vocabulary prompt tailored to the company.
        Avoids biasing hospitality or general queries with irrelevant telecom jargon.
        """
        if not company_id or not str(company_id).strip():
            from app.services.stt_service import DEFAULT_CONVERSATION_PROMPT
            return DEFAULT_CONVERSATION_PROMPT

        target_id = str(company_id).strip().lower()
        profile = self.load_company_profile(target_id)
        if profile:
            custom_prompt = profile.get("stt_prompt")
            if custom_prompt and str(custom_prompt).strip():
                return str(custom_prompt).strip()

            c_name = profile.get("company_name") or "Hotel"
            return (
                f"{c_name} front desk conversation: check-in, check-out, "
                f"room reservations, dining options, venues, amenities, and guest assistance."
            )

        from app.services.stt_service import DEFAULT_CONVERSATION_PROMPT
        return DEFAULT_CONVERSATION_PROMPT

    def get_relevant_knowledge(
        self,
        company_id: str,
        query: str,
        max_facts: int = 3,
        conversation_context: str = "",
    ) -> str:
        """
        Retrieves the most relevant facts from the company profile matching the caller query.
        Returns a concise bulleted list of approved facts, or an empty string if no match.
        """
        if not query or not query.strip():
            return ""

        # Genuinely garbled queries should not falsely match generic facts
        q_clean = query.lower().strip()
        if any(g in q_clean for g in ["traffic block", "booyo", "help off", "asdf"]):
            return ""

        target_company_id = (company_id or "").strip().lower() or "pullman_aerocity"
        profile = self.load_company_profile(target_company_id)
        if not profile or "facts" not in profile:
            return ""

        facts: List[Dict[str, Any]] = profile.get("facts", [])
        if not facts:
            return ""

        q_lower = query.lower()
        # Normalize common STT phonetic mistranscriptions for Pullman venues and check-in
        q_norm = re.sub(r"\b(?:luck|plug)\s+restaurant\b", "pluck restaurant", q_lower)
        q_norm = re.sub(r"\b(?:hong|hon)\s+restaurant\b", "honk restaurant", q_norm)
        q_norm = re.sub(r"\b(?:the\s+)?tak(?:ing|e[- ]in)\s+tim(?:e|ing)(?:s)?\b", "check-in time", q_norm)
        q_norm = re.sub(r"\bchecking\s+tim(?:e|ing)(?:s)?\b", "check-in time", q_norm)
        q_norm = re.sub(r"\bchicken(?:\s+in)?\s+tim(?:e|ing)(?:s)?\b", "check-in time", q_norm)
        q_norm = re.sub(r"\bchicken\s+in\b", "check-in", q_norm)
        q_norm = re.sub(r"\bchecking(?:\s+in)?\b", "check-in", q_norm)
        q_norm = re.sub(r"\bcheck\s+tim(?:e|ing)(?:s)?\b", "check-in time", q_norm)
        q_norm = re.sub(r"\bwork(?:ing)?\s+out\b", "workout", q_norm)
        q_norm = re.sub(r"\bcheckout\b", "check-out", q_norm)
        q_norm = re.sub(r"\bcheckin\b", "check-in", q_norm)

        # Normalized tokens
        q_tokens = set(re.findall(r"\b\w+\b", q_norm))
        STOPWORDS = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
            "with", "is", "are", "was", "were", "be", "been", "can", "could",
            "do", "does", "did", "have", "has", "had", "i", "you", "he", "she",
            "it", "we", "they", "me", "my", "your", "what", "where", "when",
            "how", "there", "this", "that", "please", "tell", "about"
        }
        content_tokens = {t for t in q_tokens if t not in STOPWORDS}

        GENERIC_KEYWORDS = {"time", "times"}

        # Distinct venue & cuisine identification
        targeted_venue_ids = set()

        # Track active topic and venue from recent conversation history for follow-up questions
        hist_lower = (conversation_context or "").lower()
        recent_venue_id = None
        recent_fact_id = None
        if hist_lower:
            hist_lines = [l.strip().lower() for l in (conversation_context or "").split("\n") if l.strip()]
            recent_hist_text = " ".join(hist_lines[-4:]) if hist_lines else hist_lower
            FACT_HISTORY_CUES = [
                ("restaurant_pluck", ["pluck", "modern european", "contemporary indian"]),
                ("restaurant_honk", ["honk", "pan-asian", "dim sum", "sushi"]),
                ("restaurant_cafe_pluck", ["café pluck", "cafe pluck", "delicatessen", "bakery", "specialty coffee", "artisanal pastries"]),
                ("bar_chidya_ghar", ["chidya ghar", "the bar", "artisanal cocktail", "fine wine"]),
                ("dining_overview", ["dining venues", "dining options", "four distinct dining"]),
                ("wellness_spa_pool", ["woo wellness", "swimming pool", "heated pool", "heated outdoor pool", "the pool", "wellness spa"]),
                ("fitness_gym", ["fit lounge", "24/7 fitness", "fitness center", "cardiovascular", "the gym"]),
                ("wifi_internet", ["high-speed wi-fi", "wi-fi internet", "wifi internet"]),
                ("airport_transport", ["airport transfer", "terminal 3", "metro station", "aerocity station"]),
                ("concierge_luggage_service", ["luggage storage", "leave bags", "hold luggage", "baggage", "suitcases", "suitcase", "concierge"]),
                ("accessibility_facilities", ["wheelchair accessible", "wheelchair accessibility", "elevator", "elevators", "lift", "lifts"]),
                ("laundry_dry_cleaning", ["laundry", "dry cleaning", "pressing", "ironing", "iron clothes", "wash clothes", "washing clothes"]),
                ("parking_facilities", ["valet parking", "parking facilities"]),
                ("pet_policy", ["service animals", "traveling with other pets"]),
                ("smoking_policy", ["100% non-smoking", "outdoor smoking zones"]),
                ("check_in_id_policy", ["government-issued photo id", "passport, aadhaar", "valid address proof"]),
                ("check_in_out", ["check-in time at pullman", "check-out time is"]),
                ("hotel_services_overview", ["hotel services", "services overview", "luxury accommodations, four distinct dining", "services do you offer"]),
            ]
            fact_indices = []
            for f_id, cues in FACT_HISTORY_CUES:
                max_idx = max(
                    (m.start() for c in cues for m in re.finditer(rf"\b{re.escape(c)}\b", recent_hist_text)),
                    default=-1
                )
                if max_idx != -1:
                    fact_indices.append((max_idx, f_id))
            if fact_indices:
                fact_indices.sort(key=lambda x: x[0], reverse=True)
                recent_fact_id = fact_indices[0][1]
                if recent_fact_id in ["restaurant_pluck", "restaurant_honk", "restaurant_cafe_pluck", "bar_chidya_ghar"]:
                    recent_venue_id = recent_fact_id

        # Check if caller clearly changes topic or venue in the new query
        has_explicit_other_venue = any(v in q_norm for v in ["honk", "asian", "sushi", "dim sum", "cafe", "coffee", "pastry", "bar", "cocktail", "wine", "chidya"])
        has_topic_change = any(t in q_norm for t in [
            "check-in", "check in", "checkin", "check-out", "check out", "checkout",
            "pool", "swimming", "swim", "heated", "spa", "gym", "fitness",
            "wifi", "internet", "room rate", "room rates", "book a room", "reserve a room",
            "rooms", "room", "stay", "luggage", "bags", "parking", "pet", "smoking",
            "elevator", "lift", "laundry", "iron", "ironing"
        ])
        if has_topic_change:
            recent_venue_id = None
            recent_fact_id = None

        has_dining_context = any(w in q_norm for w in ["eat", "food", "dining", "restaurant", "bite", "bites", "coffee", "cafe", "café", "snack", "snacks", "meal", "drink", "drinks", "order", "dishes", "menu"])

        # Follow-up cues referring to the active venue
        FOLLOW_UP_VENUE_CUES = [
            "what kind of food", "what type of food", "what food", "what cuisine",
            "they serve", "do they serve", "what do they serve", "their food", "their menu",
            "tell me about them", "tell me more about it", "what do they have", "what they have",
            "do they have", "they have", "do they offer", "they offer",
            "visible options", "visitor options", "visual options", "vegetable options",
            "great options", "good options", "nice options", "options",
            "vegetarian", "veg", "vegan", "dishes", "regional dishes", "they prepare", "do they prepare", "what dishes"
        ]
        is_venue_followup = any(c in q_norm for c in FOLLOW_UP_VENUE_CUES)
        if recent_venue_id and is_venue_followup and not has_explicit_other_venue and not has_topic_change:
            targeted_venue_ids.add(recent_venue_id)

        # General follow-up cues referring to the active facility or entity
        GENERAL_FOLLOW_UP_CUES = [
            "is it", "does it", "can it", "what time does it", "what time is it", "what time do they",
            "how much is it", "how much does it cost", "is there any", "is it open", "is it closed",
            "when does it open", "when does it close", "is it heated", "is it free", "is it complimentary",
            "can i use it", "where is it", "what about it", "tell me more", "how about that",
            "what are the hours", "what are the timings", "what time", "what are their hours",
            "what are their timings", "are they open", "when do they open", "when do they close", "what hours"
        ]
        is_general_followup = any(c in q_norm for c in GENERAL_FOLLOW_UP_CUES) or (
            len(q_tokens) <= 5 and bool({"it", "there", "that", "they", "their", "them"}.intersection(q_tokens))
        )
        targeted_fact_id = None
        if recent_fact_id and (is_general_followup or is_venue_followup) and not has_topic_change:
            targeted_fact_id = recent_fact_id

        # Negative venue negation handling (e.g. "not asking about Pluck, tell me about Honk")
        is_negated_pluck = any(p in q_norm for p in ["not pluck", "not asking about pluck", "not about pluck", "instead of pluck", "no pluck"])
        is_negated_honk = any(p in q_norm for p in ["not honk", "not asking about honk", "not about honk", "instead of honk", "no honk"])
        is_negated_cafe = any(p in q_norm for p in ["not cafe", "not café", "not asking about cafe", "not asking about café", "instead of cafe", "no cafe"])
        is_negated_bar = any(p in q_norm for p in ["not bar", "not asking about bar", "not the bar", "instead of the bar", "no bar", "not chidya"])

        # Pluck cues (modern European and contemporary Indian, farm-to-table)
        pluck_cues = [
            "contemporary indian", "modern european", "farm to table", "farm-to-table",
            "herb garden", "indian food", "indian cuisine", "indian restaurant", "indian dishes",
            "european food", "european cuisine", "european restaurant", "signature restaurant"
        ]
        if not is_negated_pluck:
            if "cafe pluck" not in q_norm and "café pluck" not in q_norm and "pluck" in q_norm:
                targeted_venue_ids.add("restaurant_pluck")
            elif any(cue in q_norm for cue in pluck_cues):
                targeted_venue_ids.add("restaurant_pluck")
            elif has_dining_context and ("indian" in content_tokens or ("european" in content_tokens and "cafe" not in content_tokens)):
                targeted_venue_ids.add("restaurant_pluck")

        # Honk cues (pan-Asian street food, dim sum, sushi, regional China, Japan, Thailand, Vietnam)
        honk_cues = [
            "honk", "asian", "pan-asian", "pan asian", "sushi", "dim sum", "dimsum",
            "noodles", "chinese", "japanese", "thai", "vietnamese", "oriental", "street food"
        ]
        if not is_negated_honk:
            if any(cue in q_norm for cue in honk_cues) or (has_dining_context and any(cue in content_tokens for cue in ["asian", "sushi", "noodles", "chinese", "japanese", "thai", "vietnamese"])):
                targeted_venue_ids.add("restaurant_honk")

        # Café Pluck cues (24-hour delicatessen, bakery, coffee, pastries, late-night dining)
        cafe_direct_cues = [
            "cafe pluck", "café pluck", "cafe", "café", "coffee", "pastry", "pastries",
            "bakery", "cake", "cakes", "tea", "deli", "delicatessen", "patisserie",
            "croissant", "espresso", "latte", "cappuccino", "light bites"
        ]
        cafe_timing_cues = [
            "24 hour", "24 hours", "24/7", "late night", "midnight", "middle of the night",
            "open at night", "eat late", "late food", "2 am", "3 am", "night dining"
        ]
        if not is_negated_cafe:
            if any(cue in q_norm for cue in cafe_direct_cues) or (any(cue in q_norm for cue in cafe_timing_cues) and has_dining_context):
                targeted_venue_ids.add("restaurant_cafe_pluck")

        # The Chidya Ghar Bar cues (artisanal cocktails, spirits, fine wines, contemporary bar)
        bar_cues = [
            "chidya ghar", "chidya", "bar", "cocktail", "cocktails", "wine", "wines",
            "spirits", "beer", "alcohol", "liquor", "lounge", "pub", "drinks", "drinking",
            "for a drink", "grab a drink", "have a drink", "spots for a drink", "spot for a drink"
        ]
        if not is_negated_bar:
            if any(cue in q_norm for cue in bar_cues):
                targeted_venue_ids.add("bar_chidya_ghar")

        # Check-in / check-out cues
        CHECKIN_OUT_CUES = {
            "check-in", "check in", "checkin", "check-out", "check out", "checkout",
            "arrive", "arriving", "arrives", "arrival", "depart", "departure", "leave", "leaving", "what time is check",
            "when is check", "early check", "late check"
        }

        # Room service cues (unsupported in hotel facts)
        ROOM_SERVICE_CUES = [
            "room service", "in-room dining", "in room dining", "delivered to my room",
            "deliver to my room", "deliver to room", "delivery to my room", "delivery to room",
            "food to my room", "order to my room", "order to room"
        ]
        is_room_service_query = any(cue in q_norm for cue in ROOM_SERVICE_CUES)

        # Dining cues to avoid matching live_rates_and_availability (which is only for hotel rooms/stays)
        DINING_PRICE_OR_BOOKING_CUES = {
            "dining", "table", "restaurant", "restaurants", "dinner", "lunch", "breakfast",
            "dish", "dishes", "meal", "food", "pluck", "honk", "cafe", "bar", "drinks"
        }
        ROOM_STAY_WORDS = {
            "stay", "night", "nights", "suite", "suites", "occupancy", "vacant",
            "vacancy", "room rate", "room rates", "tariff", "tariffs", "live rate", "live rates"
        }

        # Attribute keywords that should NOT trigger a fact match without relevant category context
        ATTRIBUTE_KEYWORDS = {"24 hour", "24 hours", "24/7"}
        DINING_CATEGORY_WORDS = {
            "cafe", "café", "coffee", "pastry", "bakery", "cake", "tea", "deli",
            "patisserie", "food", "eat", "dining", "restaurant", "restaurants"
        }

        scored_facts = []
        for item in facts:
            fact_id = item.get("id", "")

            # Exclude negated dining venues
            if fact_id == "restaurant_pluck" and is_negated_pluck:
                continue
            if fact_id == "restaurant_honk" and is_negated_honk:
                continue
            if fact_id == "restaurant_cafe_pluck" and is_negated_cafe:
                continue
            if fact_id == "bar_chidya_ghar" and is_negated_bar:
                continue

            # If room service is specifically queried, exclude dining venues and rates so we don't hallucinate delivery
            if is_room_service_query and fact_id in ["dining_overview", "restaurant_cafe_pluck", "live_rates_and_availability", "hotel_services_overview"]:
                continue

            # If specific venue(s) are targeted, exclude the generic overview fact unless user asks for all options
            asks_all_dining = any(p in q_norm for p in ["all restaurant", "all dining", "all venues", "overview", "every restaurant"])
            if targeted_venue_ids and fact_id == "dining_overview" and not asks_all_dining:
                continue

            # check_in_out fact must only match when query actually pertains to arrival/departure/check-in/out
            if fact_id == "check_in_out":
                # Do not match if asking about ID documents, luggage storage, or parking
                ID_WORDS = {"id", "identity", "identification", "proof", "document", "documents", "passport", "aadhaar", "pan", "license", "voter"}
                if content_tokens.intersection(ID_WORDS) or "pan card" in q_norm:
                    continue
                LUGGAGE_WORDS = {"luggage", "baggage", "bags", "bag", "store", "storage", "cloakroom", "suitcases", "suitcase", "rucksack", "rucksacks", "backpack", "backpacks", "belongings"}
                if content_tokens.intersection(LUGGAGE_WORDS):
                    continue
                has_checkin_out_cue = any(c in q_norm for c in CHECKIN_OUT_CUES)
                if not has_checkin_out_cue:
                    continue

            # live_rates_and_availability is strictly for hotel room bookings, not restaurant tables, dish prices, or amenities
            if fact_id == "live_rates_and_availability":
                has_dining_word = any(w in q_norm for w in DINING_PRICE_OR_BOOKING_CUES)
                has_room_stay_word = any(w in q_norm for w in ROOM_STAY_WORDS)
                has_amenity_word = any(w in q_norm for w in ["wifi", "wi-fi", "internet", "wireless", "broadband", "pool", "swimming", "gym", "fitness", "workout", "luggage", "parking"])
                if (has_dining_word or has_amenity_word) and not has_room_stay_word:
                    continue

            # check_in_id_policy must match when query asks for ID proof, documents, or check-in requirements
            if fact_id == "check_in_id_policy":
                ID_PROOF_CUES = [
                    "id proof", "identity proof", "identification", "photo id", "government id",
                    "id card", "id requirement", "id required", "passport", "aadhaar",
                    "driving license", "voter id", "documents required", "document required",
                    "what documents", "which documents", "proof of identity", "address proof",
                    "pan", "pan card", "verification", "what to bring", "documents to show",
                    "documents needed", "id needed", "id must", "mandatory id"
                ]
                has_direct_id_cue = any(c in q_norm for c in ID_PROOF_CUES)
                has_checkin_id_cue = (
                    bool({"id", "document", "documents", "proof", "passport", "aadhaar", "pan", "license"}.intersection(content_tokens))
                    and any(c in q_norm for c in ["check-in", "check in", "reception", "desk", "room", "guest", "stay", "register"])
                )
                if not (has_direct_id_cue or has_checkin_id_cue):
                    continue

            score = 0
            keywords = item.get("keywords", [])
            fact_text = item.get("fact", "").lower()
            category = item.get("category", "").lower()

            # Boost targeted venue(s) or targeted topic
            if targeted_venue_ids and fact_id in targeted_venue_ids:
                score += 10
            if targeted_fact_id and fact_id == targeted_fact_id:
                score += 10

            # 1. Multi-word or hyphenated phrase match (high weight: 6 points)
            for kw in keywords:
                kw_lower = kw.lower()
                kw_spaced = kw_lower.replace("-", " ")
                kw_hyphen = kw_lower.replace(" ", "-")

                # If it's an attribute keyword like "24 hours", match only if dining or fitness category words are present
                if kw_lower in ATTRIBUTE_KEYWORDS:
                    is_dining_q = bool(content_tokens.intersection(DINING_CATEGORY_WORDS)) or any(c in q_norm for c in ["eat", "food", "drink"])
                    is_fitness_q = any(c in q_norm for c in ["gym", "fitness", "workout", "fit lounge", "exercise"])
                    if fact_id == "fitness_gym" and not is_fitness_q and fact_id != targeted_fact_id:
                        continue
                    elif fact_id == "restaurant_cafe_pluck" and not is_dining_q and fact_id != targeted_fact_id:
                        continue
                    elif fact_id not in ["fitness_gym", "restaurant_cafe_pluck"]:
                        continue

                # Generic question starters like "where is" must not match hotel address unless location or hotel nouns are present
                if kw_lower in ["where is", "where are", "where located"]:
                    LOCATION_NOUNS = {"hotel", "pullman", "located", "location", "address", "situated", "directions", "landmark", "find", "terminal", "airport", "station", "reach", "property", "place"}
                    if not any(n in q_norm for n in LOCATION_NOUNS):
                        continue

                if (" " in kw_lower or "-" in kw_lower) and (kw_lower in q_norm or kw_spaced in q_norm or kw_hyphen in q_norm):
                    score += 6
                elif (
                    kw_lower in content_tokens
                    or (kw_lower in ("arrive", "arrival", "arriving") and any(t.startswith("arriv") for t in content_tokens))
                    or ((" " in kw_lower or "-" in kw_lower) and kw_spaced in q_norm)
                ):
                    if kw_lower in GENERIC_KEYWORDS:
                        score += 1
                    else:
                        score += 3

            # 2. Category match (2 points)
            if category and category in content_tokens:
                score += 2

            # 3. Content term overlap with fact body (1 point per meaningful word, max 3)
            if content_tokens:
                fact_tokens = set(re.findall(r"\b\w+\b", fact_text)) - STOPWORDS
                overlap = len(content_tokens.intersection(fact_tokens))
                score += min(overlap, 3)

            # Minimum score threshold of 3 to avoid spurious generic single-word noise
            if score >= 3:
                scored_facts.append((score, item["fact"]))

        if not scored_facts:
            return ""

        # Sort descending by relevance score
        scored_facts.sort(key=lambda x: x[0], reverse=True)
        top_facts = [fact for _, fact in scored_facts[:max_facts]]

        return "\n".join(f"- {f}" for f in top_facts)
