import pytest

from swarm_memory.core.utils import timed


class TestTimedUtility:
    def test_timed_does_not_suppress_exceptions(self):
        """Exceptions inside `with timed(...)` should propagate."""
        with pytest.raises(ValueError), timed("test_block"):
            raise ValueError("intentional error")

    def test_timed_yields_control(self):
        """The timed block should execute the inner code."""
        executed = []
        with timed("test"):
            executed.append(True)
        assert executed == [True]
