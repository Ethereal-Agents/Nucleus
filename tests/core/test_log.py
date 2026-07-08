import logging

from swarm_memory.core.log import RunIdFilter, current_run_id, setup_logging


def test_run_id_filter():
    filt = RunIdFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=(), exc_info=None
    )

    # Default is "-"
    current_run_id.set("-")
    assert filt.filter(record) is True
    assert record.run_id == "-"

    # Set to a specific run id
    current_run_id.set("run-xyz")
    filt.filter(record)
    assert record.run_id == "run-xyz"


def test_setup_logging():
    # Verify setup_logging sets up the handler and filter properly without crashing
    setup_logging(level=logging.DEBUG)

    root_logger = logging.getLogger()
    assert len(root_logger.handlers) == 1
    handler = root_logger.handlers[0]

    # Check that our custom filter is in the handler
    assert any(isinstance(f, RunIdFilter) for f in handler.filters)

    # Cleanup to avoid polluting other tests' logs if run concurrently
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
