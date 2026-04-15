from ontology import place_hierarchy as _place_hierarchy_module

globals().update(
    {
        name: getattr(_place_hierarchy_module, name)
        for name in dir(_place_hierarchy_module)
        if not name.startswith("__")
    }
)

__all__ = [name for name in globals() if not name.startswith("__")]
