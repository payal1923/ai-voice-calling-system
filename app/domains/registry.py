"""
app/domains/registry.py
Pluggable Domain Registry for Universal Multi-Business Voice Agent.
"""

from typing import Dict, Any, Optional


class DomainRegistry:
    """Central registry mapping industry domain slugs to domain handler providers."""
    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, domain_module: str, handler_instance: Any):
        clean_key = domain_module.strip().lower()
        cls._registry[clean_key] = handler_instance

    @classmethod
    def get(cls, domain_module: Optional[str]) -> Optional[Any]:
        if not domain_module:
            return None
        return cls._registry.get(domain_module.strip().lower())

    @classmethod
    def list_domains(cls) -> list:
        return list(cls._registry.keys())

