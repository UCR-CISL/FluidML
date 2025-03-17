import multiprocessing
import multiprocessing.context
import multiprocessing.sharedctypes
import multiprocessing.synchronize


class RWLock(object):
    def __init__(
        self, mp_context: multiprocessing.context.BaseContext, *args, **kwargs
    ) -> "RWLock":
        super().__init__(*args, **kwargs)
        self._mp_context: multiprocessing.context.BaseContext = mp_context
        self._readers: multiprocessing.sharedctypes.Value = mp_context.Value("i", 0)
        self._readers_lock: multiprocessing.synchronize.Lock = mp_context.Lock()
        self._writers_lock: multiprocessing.synchronize.Lock = mp_context.Lock()

    def acquire_read(self) -> None:
        with self._readers_lock:
            self._readers.value += 1
            if self._readers.value == 1:
                self._writers_lock.acquire()

    def release_read(self) -> None:
        with self._readers_lock:
            self._readers.value -= 1
            if self._readers.value == 0:
                self._writers_lock.release()

    def acquire_write(self) -> None:
        self._writers_lock.acquire()

    def release_write(self) -> None:
        self._writers_lock.release()

    def read(self) -> "ReadState":
        return ReadState(self)

    def write(self) -> "WriteState":
        return WriteState(self)


class ReadState(object):
    def __init__(self, rwlock: RWLock, *args, **kwargs) -> "ReadState":
        super().__init__(*args, **kwargs)
        self._rwlock: RWLock = rwlock

    def __enter__(self) -> None:
        self._rwlock.acquire_read()

    def __exit__(self, *_) -> None:
        self._rwlock.release_read()


class WriteState(object):
    def __init__(self, rwlock: RWLock, *args, **kwargs) -> "WriteState":
        super().__init__(*args, **kwargs)
        self._rwlock: RWLock = rwlock

    def __enter__(self) -> None:
        self._rwlock.acquire_write()

    def __exit__(self, *_) -> None:
        self._rwlock.release_write()
