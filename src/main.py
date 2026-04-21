import sys

from runtime import main as _main_module

globals().update(
    {
        name: getattr(_main_module, name)
        for name in dir(_main_module)
        if not name.startswith("__")
    }
)

__all__ = [name for name in globals() if not name.startswith("__")]


def _sync_runtime_main_module() -> None:
    if "neo4j_loader" in sys.modules:
        sys.modules["graph.neo4j_loader"] = sys.modules["neo4j_loader"]

    for name in (
        "AnalystPipelineResult",
        "ExtractionError",
        "ExtractionPipelineResult",
        "LLMExtractor",
        "PipelineConsole",
        "ZeroShotPipelineResult",
        "_console_print",
        "_format_duration",
        "_format_token_visual",
        "_infer_company_name",
        "_mode_name",
        "_prepare_output_layout",
        "resolve_entities",
        "resolve_model_settings",
        "validate_triples",
    ):
        if name in globals():
            setattr(_main_module, name, globals()[name])


def main(argv=None):
    _sync_runtime_main_module()
    if argv is None:
        return _main_module.main()
    return _main_module.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
