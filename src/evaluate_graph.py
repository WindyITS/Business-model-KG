from graph import evaluate_graph as _evaluate_graph_module

globals().update(
    {
        name: getattr(_evaluate_graph_module, name)
        for name in dir(_evaluate_graph_module)
        if not name.startswith("__")
    }
)

__all__ = [name for name in globals() if not name.startswith("__")]


if __name__ == "__main__":
    raise SystemExit(main())
