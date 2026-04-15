from runtime import entity_resolver as _entity_resolver_module

globals().update(
    {
        name: getattr(_entity_resolver_module, name)
        for name in dir(_entity_resolver_module)
        if not name.startswith("__")
    }
)

__all__ = [name for name in globals() if not name.startswith("__")]
