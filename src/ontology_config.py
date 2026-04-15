from ontology import config as _ontology_config_module

globals().update(
    {
        name: getattr(_ontology_config_module, name)
        for name in dir(_ontology_config_module)
        if not name.startswith("__")
    }
)

__all__ = [name for name in globals() if not name.startswith("__")]
