"""AI Customer Support Desk — customers, tickets, agents, tags, notes, history.

This package extends the inquiry processor into a lightweight support-desk
core: same SQLite database, same layered style (API -> service -> storage),
same LLM abstraction for all AI operations.
"""

__all__ = ["models", "storage", "workflow", "ai", "auth", "service", "seed"]
