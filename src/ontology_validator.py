from ontology import validator as _ontology_validator_module

globals().update(
    {
        name: getattr(_ontology_validator_module, name)
        for name in dir(_ontology_validator_module)
        if not name.startswith("__")
    }
)

__all__ = [name for name in globals() if not name.startswith("__")]


if __name__ == "__main__":
    raise SystemExit(main())
