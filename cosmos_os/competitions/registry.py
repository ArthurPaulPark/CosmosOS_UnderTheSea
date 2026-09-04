from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .base import BaseCompetitionAdapter

LOGGER = logging.getLogger(__name__)


class CompetitionRegistry:
    """Registry for managing competition adapters."""
    
    _adapters: dict[str, Type["BaseCompetitionAdapter"]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a competition adapter.
        
        Args:
            name (str): The unique name of the competition (e.g., 'ocean').
        """
        def decorator(adapter_cls: Type["BaseCompetitionAdapter"]):
            if name in cls._adapters:
                LOGGER.warning("Adapter for competition '%s' is being overwritten.", name)
            cls._adapters[name] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> "BaseCompetitionAdapter":
        """Create an instance of a registered adapter.
        
        Args:
            name (str): The unique name of the competition.
            **kwargs: Additional arguments to pass to the adapter's constructor.
            
        Returns:
            BaseCompetitionAdapter: An instance of the registered adapter.
            
        Raises:
            KeyError: If the adapter name is not found in the registry.
        """
        if name not in cls._adapters:
            raise KeyError(f"Competition adapter '{name}' is not registered. Available: {list(cls._adapters.keys())}")
        return cls._adapters[name](**kwargs)
