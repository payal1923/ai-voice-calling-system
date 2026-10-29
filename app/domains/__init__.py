"""
app/domains package.
Provides modular domain-specific business logic extensions.
"""
from app.domains.registry import DomainRegistry
from app.domains.hospitality import HospitalityDomain
from app.domains.automotive_rental import AutomotiveRentalDomain

# Register default domains
DomainRegistry.register("hospitality", HospitalityDomain)
DomainRegistry.register("automotive_rental", AutomotiveRentalDomain)

__all__ = ["DomainRegistry", "HospitalityDomain", "AutomotiveRentalDomain"]

