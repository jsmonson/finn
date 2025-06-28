"""
FINN Plugin Registry

Simple registry for FINN plugins, focusing on transforms for now.
"""

import logging
from typing import Dict, Type, Optional, List, Tuple, Any

logger = logging.getLogger(__name__)


class FinnPluginRegistry:
    """
    Registry for FINN plugins including transforms, kernels, and backends.
    
    Supports dynamic discovery of available backends for each kernel through
    backend declarations.
    """
    
    _instance = None
    _plugins = {}
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize registry if not already done."""
        if self._initialized:
            return
        
        self._plugins = {
            "transform": {},
            "kernel": {},
            "backend": {}
        }
        self._initialized = True
        logger.debug("FinnPluginRegistry initialized")
    
    @classmethod
    def register(cls, plugin_class: Type) -> None:
        """
        Register a plugin class.
        
        Args:
            plugin_class: Class with _plugin_metadata attribute
        """
        instance = cls()
        
        # Get metadata
        if not hasattr(plugin_class, '_plugin_metadata'):
            raise ValueError(f"Plugin class {plugin_class.__name__} missing _plugin_metadata")
        
        metadata = plugin_class._plugin_metadata
        plugin_type = metadata.get("type")
        name = metadata.get("name")
        
        if plugin_type not in instance._plugins:
            raise ValueError(f"Unknown plugin type: {plugin_type}")
        
        # Register the plugin
        if plugin_type == "backend":
            # Store backend with additional metadata
            instance._plugins[plugin_type][name] = {
                "class": plugin_class,
                "kernel": metadata.get("kernel"),
                "backend_type": metadata.get("backend_type")
            }
        else:
            instance._plugins[plugin_type][name] = plugin_class
        logger.debug(f"Registered {plugin_type}: {name}")
    
    def get_transform(self, name: str) -> Optional[Type]:
        """Get a transform by name."""
        return self._plugins["transform"].get(name)
    
    def list_transforms(self, category: Optional[str] = None) -> List[Tuple[str, Type]]:
        """
        List all transforms, optionally filtered by category.
        
        Args:
            category: Optional category to filter by (streamline, fpgadataflow, etc.)
            
        Returns:
            List of (name, class) tuples
        """
        transforms = []
        for name, cls in self._plugins["transform"].items():
            metadata = cls._plugin_metadata
            if category is None or metadata.get("category") == category:
                transforms.append((name, cls))
        return sorted(transforms, key=lambda x: x[0])
    
    def get_categories(self) -> List[str]:
        """Get all unique transform categories."""
        categories = set()
        for cls in self._plugins["transform"].values():
            category = cls._plugin_metadata.get("category")
            if category:
                categories.add(category)
        return sorted(list(categories))
    
    def get_kernel(self, name: str) -> Optional[Type]:
        """Get a kernel by name."""
        return self._plugins["kernel"].get(name)
    
    def get_backend(self, kernel_name: str, backend_type: str) -> Optional[Type]:
        """Get a specific backend for a kernel."""
        for backend_name, backend_info in self._plugins["backend"].items():
            if (backend_info["kernel"] == kernel_name and 
                backend_info["backend_type"] == backend_type):
                return backend_info["class"]
        return None
    
    def get_backends_for_kernel(self, kernel_name: str) -> Dict[str, Type]:
        """Get all available backends for a kernel."""
        backends = {}
        for backend_name, backend_info in self._plugins["backend"].items():
            if backend_info["kernel"] == kernel_name:
                backend_type = backend_info["backend_type"]
                backends[backend_type] = backend_info["class"]
        return backends
    
    def list_kernels(self) -> List[Tuple[str, Type]]:
        """List all registered kernels."""
        kernels = [(name, cls) for name, cls in self._plugins["kernel"].items()]
        return sorted(kernels, key=lambda x: x[0])
    
    def list_backends(self, kernel_name: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """List all backends, optionally filtered by kernel."""
        backends = []
        for name, info in self._plugins["backend"].items():
            if kernel_name is None or info["kernel"] == kernel_name:
                backends.append((name, {
                    "kernel": info["kernel"],
                    "backend_type": info["backend_type"],
                    "class": info["class"]
                }))
        return sorted(backends, key=lambda x: x[0])
    
    def get_kernels_by_op_type(self, op_type: str) -> List[Tuple[str, Type]]:
        """Get kernels that implement a specific ONNX op type."""
        kernels = []
        for name, cls in self._plugins["kernel"].items():
            metadata = getattr(cls, '_plugin_metadata', {})
            if metadata.get("op_type") == op_type:
                kernels.append((name, cls))
        return sorted(kernels, key=lambda x: x[0])
    
    def clear(self):
        """Clear the registry (mainly for testing)."""
        self._plugins = {"transform": {}, "kernel": {}, "backend": {}}


# Global instance for convenience
_registry_instance = None


def get_finn_registry() -> FinnPluginRegistry:
    """Get the global FINN registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = FinnPluginRegistry()
    return _registry_instance