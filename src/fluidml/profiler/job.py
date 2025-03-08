import multiprocessing
import multiprocessing.connection
import multiprocessing.context
import multiprocessing.managers
import multiprocessing.sharedctypes
import multiprocessing.synchronize

from typing import List, Tuple


class Job(object):
    def __init__(self, *args, **kwargs) -> "Job":
        super().__init__(*args, **kwargs)


class RunnableJob(Job):
    def __init__(self, *args, **kwargs) -> "RunnableJob":
        super().__init__(*args, **kwargs)


class UnRunnableJob(Job):
    def __init__(self, *args, **kwargs) -> "UnRunnableJob":
        super().__init__(*args, **kwargs)


class CreateSubModJob(RunnableJob):
    def __init__(self, mod: str, *args, **kwargs) -> "CreateSubModJob":
        super().__init__(*args, **kwargs)
        self._mod: str = mod

    @property
    def mod(self) -> str:
        return self._mod


class BenchSubModJob(RunnableJob):
    def __init__(self, mod: str, *args, **kwargs) -> "BenchSubModJob":
        super().__init__(*args, **kwargs)
        self._mod: str = mod

    @property
    def mod(self) -> str:
        return self._mod


class ResultJob(UnRunnableJob):
    def __init__(
        self, kernel: str, axes: List[Tuple[int, ...]], time: float, *args, **kwargs
    ) -> "ResultJob":
        super().__init__(*args, **kwargs)
        self._kernel: str = kernel
        self._axes: List[Tuple[int, ...]] = axes
        self._time: float = time

    @property
    def kernel(self) -> str:
        return self._kernel

    @property
    def axes(self) -> List[Tuple[int, ...]]:
        return self._axes

    @property
    def time(self) -> float:
        return self._time


class JobPool(object):
    def __init__(
        self, mp_context: multiprocessing.context.SpawnContext, *args, **kwargs
    ) -> "JobPool":
        super().__init__(*args, **kwargs)
        self._mp_context: multiprocessing.context.SpawnContext = mp_context
        self._create_queue: multiprocessing.Queue = self._mp_context.Queue()
        self._bench_queue: multiprocessing.Queue = self._mp_context.Queue()
        self._result_queue: multiprocessing.Queue = self._mp_context.Queue()
        self._exception_queue: multiprocessing.Queue = self._mp_context.Queue()
        self._lock: multiprocessing.synchronize.RLock = self._mp_context.RLock()
        self._sem: multiprocessing.synchronize.Semaphore = self._mp_context.Semaphore(0)
        self._cond: multiprocessing.synchronize.Condition = self._mp_context.Condition(
            self._lock
        )
        self._create_jobs: multiprocessing.sharedctypes.Synchronized[
            int
        ] = self._mp_context.Value("i", 0)
        self._bench_jobs: multiprocessing.sharedctypes.Synchronized[
            int
        ] = self._mp_context.Value("i", 0)
        self._working_jobs: multiprocessing.sharedctypes.Synchronized[
            int
        ] = self._mp_context.Value("i", 0)

    @property
    def _job_num(self) -> int:
        with self._lock:
            return self._create_jobs.value + self._bench_jobs.value

    @property
    def empty(self) -> bool:
        with self._lock:
            return self._job_num == 0

    @property
    def done(self) -> bool:
        with self._lock:
            return self._working_jobs.value == 0 and self.empty

    def put(self, job: Job) -> None:
        with self._lock:
            if isinstance(job, CreateSubModJob):
                self._create_queue.put(job)
                self._create_jobs.value += 1
                self._sem.release()
            elif isinstance(job, BenchSubModJob):
                self._bench_queue.put(job)
                self._bench_jobs.value += 1
                self._sem.release()
            elif isinstance(job, ResultJob):
                self._result_queue.put(job)
            else:
                raise NotImplementedError(f"Unsupported job {job} of type {type(job)}.")

    def get(self) -> Job:
        while True:
            self._sem.acquire()
            with self._lock:
                if self._bench_jobs.value > 0:
                    self._bench_jobs.value -= 1
                    self._working_jobs.value += 1
                    job: Job = self._bench_queue.get()
                elif self._create_jobs.value > 0:
                    self._create_jobs.value -= 1
                    self._working_jobs.value += 1
                    job: Job = self._create_queue.get()
                else:
                    raise RuntimeError("No job available.")
                return job

    def free(self) -> None:
        with self._lock:
            self._working_jobs.value -= 1
            if self.done:
                self._cond.notify_all()

    def throw(self, exception: Exception) -> None:
        self._exception_queue.put(exception)

    def wait(
        self, check_period: float
    ) -> List[Tuple[str, Tuple[Tuple[int, ...]], float]]:
        while not self.done:
            with self._cond:
                while not self._cond.wait(timeout=check_period):
                    if not self._exception_queue.empty():
                        execption: Exception = self._exception_queue.get()
                        raise execption
                results: List[Tuple[str, Tuple[Tuple[int, ...]], float]] = []
                while not self._result_queue.empty():
                    job: ResultJob = self._result_queue.get()
                    results += [(job.kernel, job.axes, job.time)]
                return results
