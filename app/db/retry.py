from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from sqlalchemy.exc import DBAPIError, OperationalError

T = TypeVar("T")


def is_deadlock_error(exc: BaseException) -> bool:
    """True for MySQL 1213 (deadlock) / 1205 (lock wait timeout)."""
    codes: list[int] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        args = getattr(cur, "args", ())
        if args and isinstance(args[0], int):
            codes.append(args[0])
        orig = getattr(cur, "orig", None)
        cur = orig if isinstance(orig, BaseException) else None
    return any(c in {1213, 1205} for c in codes)


def run_with_deadlock_retry(
    fn: Callable[[], T],
    *,
    retries: int = 6,
    base_delay: float = 0.05,
) -> T:
    last: BaseException | None = None
    for attempt in range(retries):
        try:
            return fn()
        except (OperationalError, DBAPIError) as e:
            last = e
            if not is_deadlock_error(e) or attempt >= retries - 1:
                raise
            time.sleep(base_delay * (2**attempt) + random.random() * base_delay)
        except Exception as e:
            # pymysql may surface OperationalError unwrapped in some paths
            last = e
            if not is_deadlock_error(e) or attempt >= retries - 1:
                raise
            time.sleep(base_delay * (2**attempt) + random.random() * base_delay)
    assert last is not None
    raise last
