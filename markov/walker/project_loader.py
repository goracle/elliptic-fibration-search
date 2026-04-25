from sage.all import *
from pathlib import Path
from .curve_helpers import *
from .curve_helpers import _coerce_base_ring

PROJECT_REGISTRY: Dict[str, Any] = {}

def resolve_project_symbol(name: str, default: Any = None, required: bool = False):
    """Resolve a symbol loaded from tower.sage / search7_genus2.sage.

    Looks in PROJECT_REGISTRY first (populated by load_project_sources),
    then falls back to the module globals for any symbol that was already
    defined before load_project_sources was called.
    """
    if name in PROJECT_REGISTRY:
        return PROJECT_REGISTRY[name]
    g = globals()
    if name in g:
        return g[name]
    if required:
        raise RuntimeError(
            f"Required project symbol {name!r} not found. "
            "Call load_project_sources() before using this function."
        )
    return default


def load_project_sources(base_dir: Optional[Path] = None, verbose: bool = True) -> Dict[str, bool]:
    """Load tower.sage and search7_genus2.sage into PROJECT_REGISTRY."""
    if base_dir is not None:
        here = Path(base_dir)
    else:
        # Walk up from this file's location until we find tower.sage (repo root).
        candidate = Path(__file__).resolve().parent
        while candidate != candidate.parent:
            if (candidate / "tower.sage").exists():
                break
            candidate = candidate.parent
        else:
            raise FileNotFoundError(
                "Could not find tower.sage by walking up from "
                f"{Path(__file__).resolve().parent}"
            )
        here = candidate
    loaded: Dict[str, bool] = {}
    for name in ("tower.sage", "search7_genus2.sage"):
        path = here / name
        if verbose:
            print(f"[bootstrap] loading {path}")
        try:
            with open(path, "r") as f:
                src = f.read()
        except FileNotFoundError:
            if verbose:
                print(f"[bootstrap] WARNING: {path} not found, skipping")
            loaded[name] = False
            raise

        src = src.replace("    main_genus2()", "    pass # main_genus2() disabled")

        # Mutate the shared dict in-place so walkerclass.resolve_project_symbol sees it
        PROJECT_REGISTRY.update(
            {k: v for k, v in exec_namespace(preparse(src)).items()
             if not k.startswith('__')}
        )
        globals().update({k: v for k, v in PROJECT_REGISTRY.items()
                          if not k.startswith('__')})
        loaded[name] = True

    return loaded

def project_base_points_from_globals(current_x=None, current_y=None, p: Optional[int] = None):
    """Build a base-point list from project globals such as DATA_PTS_GENUS2."""
    data_pts = DATA_PTS_GENUS2
    yfun = get_y_unshifted_genus2
    finite_field = FINITE_FIELD

    pts = []
    for x in data_pts or []:
        y = None
        try:
            y = yfun(x) if yfun is not None else None
        except Exception:
            y = None
            raise
        if y is None:
            continue
        try:
            if finite_field is not None and p is not None:
                pts.append((GF(int(p))(x), GF(int(p))(y)))
            else:
                pts.append((QQ(x), QQ(y)))
        except Exception:
            try:
                pts.append((x, y))
            except Exception:
                raise
            raise

    if current_x is not None and current_y is not None:
        try:
            cx = GF(int(p))(current_x) if (p is not None and finite_field is not None) else QQ(current_x)
            cy = GF(int(p))(current_y) if (p is not None and finite_field is not None) else QQ(current_y)
            if (cx, cy) not in pts:
                pts.insert(0, (cx, cy))
        except Exception:
            raise

    if not pts and current_x is not None and current_y is not None:
        pts = [(current_x, current_y)]

    return pts


def exec_namespace(src: str) -> Dict[str, Any]:
    """Execute preparsed sage source and return the resulting namespace."""
    ns: Dict[str, Any] = {}
    exec(src, ns)
    return ns



