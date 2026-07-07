"""
swarm_memory/core/log.py

Logging configuration with correlation IDs.
"""

import contextvars
import logging

# Context variable to hold the current run_id.
# The default is "-" for logs emitted outside of a specific agent run.
current_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_run_id", default="-")


class RunIdFilter(logging.Filter):
    """
    Injects the `run_id` from the current context into log records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Fetch the context variable and attach it to the log record
        record.run_id = current_run_id.get()
        return True


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures standard logging for the application, injecting the context run_id.

    Call this once at the application startup (e.g., inside the MCP server init).
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [run:%(run_id)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Apply filter to the handler so that record.run_id is populated for every log
    handler.addFilter(RunIdFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates (e.g. during testing or hot-reloads)
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)
