import iree.compiler.dialects.arith
import iree.compiler.ir
import sys

from collections import defaultdict
from itertools import product
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import iree.compiler.dialects

from ...utils.kstat import KStat
from ...utils.utils import permute_shape
from ..schedule import Schedule, ScheduleGroup
from ..wrapper import DummyValue, OpWrapper
from .scope import Scope


class Sequence(Scope):
    def __init__(
        self, wrappers: Iterator[OpWrapper] = [], *args, **kwargs
    ) -> "Sequence":
        super().__init__(*args, **kwargs)
        self._wrappers: List[OpWrapper] = [
            OpWrapper(wrapper._op, self) for wrapper in wrappers
        ]

    def __len__(self) -> int:
        return self.len

    def append(
        self, op: Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView, OpWrapper]
    ) -> "Sequence":
        self._wrappers += [OpWrapper(op, self)]
        return self

    def get_prevs(self, op) -> List[OpWrapper]:
        assert op in self, f"{op} not in {self}."
        idx: int = self._wrappers.index(op)
        if idx == 0:
            return []
        prev: OpWrapper = self._wrappers[idx - 1]
        assert prev in Scope.get_prevs(self, op)
        return [prev]

    def get_nexts(self, op) -> List[OpWrapper]:
        assert op in self, f"{op} not in {self}."
        idx: int = self._wrappers.index(op)
        if idx == len(self._wrappers) - 1:
            return []
        output: OpWrapper = self._wrappers[idx + 1]
        assert output in Scope.get_nexts(self, op)
        return [output]

    def iter(self) -> Iterator[OpWrapper]:
        for wrapper in self._wrappers:
            yield wrapper

    @property
    def len(self) -> int:
        return len(self._wrappers)

    def prepend(
        self,
        wrapper: Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView, OpWrapper],
    ) -> "Sequence":
        self._wrappers = [OpWrapper(wrapper._op, self)] + self._wrappers
        return self

    def put(
        self, op: Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView, OpWrapper]
    ) -> "Sequence":
        return self.append(op)

    def schedule(self, kstat: KStat) -> ScheduleGroup:
        wind: List[
            Tuple[
                Union[str, DummyValue],
                Dict[Tuple[int, ...], Tuple[float, Optional[Tuple[int, ...]]]],
            ]
        ] = []
        for idx, wrapper in enumerate(self):
            input: Union[iree.compiler.ir.Value, DummyValue] = wrapper.scope_input
            output: Union[iree.compiler.ir.Value, DummyValue] = wrapper.scope_output
            input_idx: Optional[int] = (
                wrapper.arg_index(input) if not isinstance(input, DummyValue) else None
            )
            output_idx: Optional[int] = (
                wrapper.arg_index(output)
                if not isinstance(output, DummyValue)
                else None
            )
            if wrapper.schedule_layout:
                assert (
                    input_idx is not None
                ), f"Input {input} not found for {wrapper} in {self}."
                assert (
                    output_idx is not None
                ), f"Output {output} not found for {wrapper} in {self}."
                name: str = wrapper.entry
                assert name in kstat
                table: Dict[Tuple[int, ...], float] = kstat[name]
                input_choices: Set[Tuple[int, ...]] = {k[input_idx] for k in table}
                output_choices: Set[Tuple[int, ...]] = {k[output_idx] for k in table}
                available_choices: Dict[
                    Tuple[Tuple[int, ...], ...], List[float]
                ] = defaultdict(list)
                for k, v in table.items():
                    input_layout: Tuple[int, ...] = k[input_idx]
                    output_layout: Tuple[int, ...] = k[output_idx]
                    if (
                        input_layout in input_choices
                        and output_layout in output_choices
                    ):
                        available_choices[(input_layout, output_layout)] += [v]
                choices: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {
                    k: min(v) for k, v in available_choices.items()
                }
            elif wrapper.force_layout:
                input_shape: List[int] = (
                    wrapper.arg_types[input_idx].shape if input_idx is not None else []
                )
                output_shape: List[int] = (
                    wrapper.arg_types[output_idx].shape
                    if output_idx is not None
                    else []
                )
                choices: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {
                    (
                        tuple(range(len(input_shape))),
                        tuple(range(len(output_shape))),
                    ): 0.0
                }
            elif wrapper.any_layout:
                input_shape: List[int] = (
                    wrapper.arg_types[input_idx].shape if input_idx is not None else []
                )
                output_shape: List[int] = (
                    wrapper.arg_types[output_idx].shape
                    if output_idx is not None
                    else []
                )
                choices: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {
                    combination: 0.0
                    for combination in product(
                        permute_shape(tuple(range(len(input_shape)))),
                        permute_shape(tuple(range(len(output_shape)))),
                    )
                }
            else:
                raise NotImplementedError(
                    f"Unsupported OpWrapper: {wrapper} in {self.__class__.__name__}.schedule."
                )
            input_key: Union[str, DummyValue] = (
                input.get_name() if isinstance(input, iree.compiler.ir.Value) else input
            )
            output_key: Union[str, DummyValue] = (
                output.get_name()
                if isinstance(output, iree.compiler.ir.Value)
                else output
            )
            if idx == 0:
                wind += [(input_key, {k: (v, None) for (k, _), v in choices.items()})]
            key, ktable = wind[-1]
            assert input_key == key
            input_layouts: Set[Tuple[int, ...]] = set(ktable) & set(
                k for k, _ in choices
            )
            assert (
                input_layouts
            ), f"Input layout on {input} is not found for {wrapper} in {self}."
            exec_time_table: Dict[
                Tuple[int, ...], Tuple[float, Tuple[int, ...]]
            ] = defaultdict(lambda: (sys.float_info.max, None))
            for input_layout in input_layouts:
                prev_exec_time, _ = ktable[input_layout]
                for output_layout in map(
                    lambda k: k[1],
                    filter(lambda k: k[0] == input_layout, choices),
                ):
                    kernel_exec_time: float = choices[(input_layout, output_layout)]
                    total_exec_time: float = prev_exec_time + kernel_exec_time
                    curr_exec_time, _ = exec_time_table[output_layout]
                    if curr_exec_time > total_exec_time:
                        exec_time_table[output_layout] = (total_exec_time, input_layout)
            wind += [(output_key, {**exec_time_table})]
        group: ScheduleGroup = ScheduleGroup()
        if wind:
            lk, ltable = wind[-1]
            min_time: float = min(time for _, (time, _) in ltable.items())
            for layout, (time, prev) in ltable.items():
                if time == min_time:
                    rewind: Dict[Union[str, DummyValue], Tuple[int, ...]] = {lk: layout}
                    for (ck, ctable) in wind[-2::-1]:
                        cur: Tuple[int, ...] = prev
                        _, prev = ctable[cur]
                        rewind = {ck: cur, **rewind}
                    rewind = {
                        k: v for k, v in rewind.items() if not isinstance(k, DummyValue)
                    }
                    schedule: Schedule = Schedule(rewind)
                    group += schedule
        return group
