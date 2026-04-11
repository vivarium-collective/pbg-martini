"""Custom bigraph-schema type registrations for pbg-martini.

No custom types are needed at present — the wrapper uses built-in types
(string, integer, float, list, map).
"""


def register_types(core):
    """Register any custom types with *core* (currently a no-op)."""
    return core
