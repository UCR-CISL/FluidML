import pickle

from collections import Counter, defaultdict
from typing import BinaryIO, Dict, Iterator, List, Tuple


class Schedule(object):
    def __init__(
        self, schedule: Dict[str, Tuple[int, ...]], *args, **kwargs
    ) -> "Schedule":
        super().__init__(*args, **kwargs)
        self._schedule: Dict[str, Tuple[int, ...]] = schedule

    def __contains__(self, key: str) -> bool:
        return key in self._schedule

    def __getitem__(self, key: str) -> Tuple[int, ...]:
        return self._schedule[key]

    def __str__(self) -> str:
        return f"Schedule(\n{self._schedule}\n)"

    @staticmethod
    def build(f: BinaryIO) -> "Schedule":
        schedule: Dict[str, Tuple[int, ...]] = pickle.load(f)
        return Schedule(schedule)

    def dump(self, f: BinaryIO) -> None:
        pickle.dump(self._schedule, f)

    @staticmethod
    def merge(schedules: Iterator["Schedule"]) -> "Schedule":
        table: Dict[str, List[Tuple[int, ...]]] = defaultdict(list)
        for schedule in schedules:
            for key, value in schedule._schedule.items():
                table[key] += [value]
        schedule: Dict[str, Tuple[int, ...]] = {}
        for key, values in table.items():
            counter: Counter = Counter(values)
            [(selected, _)] = counter.most_common(1)
            schedule[key] = selected
        return Schedule(schedule)


class ScheduleGroup(object):
    def __init__(
        self, schedule_group: Iterator[Schedule] = [], *args, **kwargs
    ) -> "ScheduleGroup":
        super().__init__(*args, **kwargs)
        self._schedule_group: List[Schedule] = [*schedule_group]

    def __iadd__(self, schedule: Schedule) -> "ScheduleGroup":
        self._schedule_group += [schedule]
        return self

    def __iter__(self) -> Iterator[Schedule]:
        for schedule in self._schedule_group:
            yield schedule

    def __or__(self, value: "ScheduleGroup") -> "ScheduleGroup":
        return ScheduleGroup(self._schedule_group + value._schedule_group)

    def __str__(self) -> str:
        return f"ScheduleGroup(\n{self._schedule_group}\n)"

    def merge(self) -> Schedule:
        return Schedule.merge(self._schedule_group)
