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
        
        # Flat storage model like BrainSmith
        self._plugins = {}  # {type:name -> metadata}
        self._initialized = True
        logger.debug("FinnPluginRegistry initialized")
    
    def register(self, plugin_type: str, name: str, cls: Type, **metadata):
        """
        Register a plugin. Overwrites any existing registration.
        
        Permissive like BrainSmith - no validation, always succeeds.
        
        Args:
            plugin_type: Type of plugin (transform, kernel, backend, etc.)
            name: Plugin name
            cls: The plugin class
            **metadata: Additional metadata
        """
        key = f"{plugin_type}:{name}"
        
        # Store everything in consistent format
        self._plugins[key] = {
            "class": cls,
            "type": plugin_type,
            "name": name,
            "framework": "finn",  # Always finn for FINN plugins
            **metadata
        }
        
        logger.debug(f"Registered {key}")
    
    @classmethod
    def register_legacy(cls, plugin_class: Type) -> None:
        """
        Legacy registration method for backward compatibility.
        
        Args:
            plugin_class: Class with _plugin_metadata attribute
        """
        instance = cls()
        
        # Get metadata
        if not hasattr(plugin_class, '_plugin_metadata'):
            logger.debug(f"Plugin class {plugin_class.__name__} missing _plugin_metadata")
            return
        
        metadata = plugin_class._plugin_metadata
        plugin_type = metadata.get("type")
        name = metadata.get("name")
        
        if not plugin_type or not name:
            logger.debug(f"Plugin class {plugin_class.__name__} missing type or name")
            return
        
        # Extract metadata without duplicating fields
        reg_metadata = {k: v for k, v in metadata.items() 
                       if k not in ["type", "name"]}
        
        # Use new register method
        instance.register(plugin_type, name, plugin_class, **reg_metadata)
    
    def get(self, plugin_type: str, name: str) -> Optional[Type]:
        """
        Get plugin by type and name.
        
        Same signature as BrainSmith for compatibility.
        """
        key = f"{plugin_type}:{name}"
        entry = self._plugins.get(key)
        return entry["class"] if entry else None
    
    def get_transform(self, name: str) -> Optional[Type]:
        """Get a transform by name (legacy method)."""
        return self.get("transform", name)
    
    def list_transforms(self, category: Optional[str] = None) -> List[Tuple[str, Type]]:
        """
        List all transforms, optionally filtered by category.
        
        Args:
            category: Optional category to filter by (streamline, fpgadataflow, etc.)
            
        Returns:
            List of (name, class) tuples
        """
        transforms = []
        for key, entry in self._plugins.items():
            if entry["type"] == "transform":
                if category is None or entry.get("category") == category:
                    transforms.append((entry["name"], entry["class"]))
        return sorted(transforms, key=lambda x: x[0])
    
    def get_categories(self) -> List[str]:
        """Get all unique transform categories."""
        categories = set()
        for entry in self._plugins.values():
            if entry["type"] == "transform":
                category = entry.get("category")
                if category:
                    categories.add(category)
        return sorted(list(categories))
    
    def get_kernel(self, name: str) -> Optional[Type]:
        """Get a kernel by name (legacy method)."""
        return self.get("kernel", name)
    
    def get_backend(self, kernel_name: str, backend_type: str) -> Optional[Type]:
        """Get a specific backend for a kernel."""
        for entry in self._plugins.values():
            if (entry["type"] == "backend" and 
                entry.get("kernel") == kernel_name and 
                entry.get("backend_type") == backend_type):
                return entry["class"]
        return None
    
    def get_backends_for_kernel(self, kernel_name: str) -> Dict[str, Type]:
        """Get all available backends for a kernel."""
        backends = {}
        for entry in self._plugins.values():
            if entry["type"] == "backend" and entry.get("kernel") == kernel_name:
                backend_type = entry.get("backend_type")
                if backend_type:
                    backends[backend_type] = entry["class"]
        return backends
    
    def list_kernels(self) -> List[Tuple[str, Type]]:
        """List all registered kernels."""
        kernels = []
        for entry in self._plugins.values():
            if entry["type"] == "kernel":
                kernels.append((entry["name"], entry["class"]))
        return sorted(kernels, key=lambda x: x[0])
    
    def list_backends(self, kernel_name: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """List all backends, optionally filtered by kernel."""
        backends = []
        for entry in self._plugins.values():
            if entry["type"] == "backend":
                if kernel_name is None or entry.get("kernel") == kernel_name:
                    backends.append((entry["name"], {
                        "kernel": entry.get("kernel"),
                        "backend_type": entry.get("backend_type"),
                        "class": entry["class"]
                    }))
        return sorted(backends, key=lambda x: x[0])
    
    def get_kernels_by_op_type(self, op_type: str) -> List[Tuple[str, Type]]:
        """Get kernels that implement a specific ONNX op type."""
        kernels = []
        for entry in self._plugins.values():
            if entry["type"] == "kernel" and entry.get("op_type") == op_type:
                kernels.append((entry["name"], entry["class"]))
        return sorted(kernels, key=lambda x: x[0])
    
    def query(self, **filters) -> List[Dict[str, Any]]:
        """
        Query plugins by any metadata field.
        
        This enables FINN to work as a standalone plugin system with
        rich discovery capabilities.
        
        Args:
            **filters: Field=value pairs to match
            
        Returns:
            List of matching plugin entries with full metadata
            
        Example:
            # Get all transforms for a specific stage
            transforms = registry.query(type="transform", stage="topology_opt")
            
            # Get all components related to a kernel
            layernorm_all = registry.query(kernel="LayerNorm")
        """
        results = []
        
        # Simple field matching like BrainSmith
        for key, entry in self._plugins.items():
            # Check if all filters match
            match = True
            for field, value in filters.items():
                if entry.get(field) != value:
                    match = False
                    break
            
            if match:
                results.append(entry.copy())
        
        return results
    
    def get_plugin_info(self, plugin_type: str, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific plugin."""
        results = self.query(type=plugin_type, name=name)
        return results[0] if results else None
    
    def clear(self):
        """Clear the registry (mainly for testing)."""
        self._plugins.clear()
        logger.debug("Registry cleared")


# Global instance for convenience
_registry_instance = None


def get_finn_registry() -> FinnPluginRegistry:
    """Get the global FINN registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = FinnPluginRegistry()
    return _registry_instance