import multiprocessing
import multiprocessing.context
import multiprocessing.sharedctypes
import multiprocessing.synchronize


class ExclusiveLock(object):
    def __init__(
        self, mp_context: multiprocessing.context.BaseContext, *args, **kwargs
    ) -> "ExclusiveLock":
        super().__init__(*args, **kwargs)
        self._mp_context: multiprocessing.context.BaseContext = mp_context
        self._lock: multiprocessing.synchronize.Lock = mp_context.Lock()
        self._condition: multiprocessing.synchronize.Condition = mp_context.Condition(
            self._lock
        )
        self._red_count: multiprocessing.sharedctypes.Synchronized = mp_context.Value(
            "i", 0
        )
        self._blue_count: multiprocessing.sharedctypes.Synchronized = mp_context.Value(
            "i", 0
        )

    def acquire_blue(self) -> None:
        with self._condition:
            while self._red_count.value > 0:
                self._condition.wait()
            self._blue_count.value += 1

    def release_blue(self) -> None:
        with self._condition:
            self._blue_count.value -= 1
            if self._blue_count.value == 0:
                self._condition.notify_all()

    def acquire_red(self) -> None:
        with self._condition:
            while self._blue_count.value > 0:
                self._condition.wait()
            self._red_count.value += 1

    def release_red(self) -> None:
        with self._condition:
            self._red_count.value -= 1
            if self._red_count.value == 0:
                self._condition.notify_all()

    def blue(self) -> "BlueState":
        return BlueState(self)

    def red(self) -> "RedState":
        return RedState(self)


class BlueState(object):
    def __init__(self, exlock: ExclusiveLock, *args, **kwargs) -> "BlueState":
        super().__init__(*args, **kwargs)
        self._exlock: ExclusiveLock = exlock

    def __enter__(self) -> None:
        self._exlock.acquire_blue()

    def __exit__(self, *_) -> None:
        self._exlock.release_blue()


class RedState(object):
    def __init__(self, exlock: ExclusiveLock, *args, **kwargs) -> "RedState":
        super().__init__(*args, **kwargs)
        self._exlock: ExclusiveLock = exlock

    def __enter__(self) -> None:
        self._exlock.acquire_red()

    def __exit__(self, *_) -> None:
        self._exlock.release_red()
