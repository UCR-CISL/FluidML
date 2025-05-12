from __future__ import annotations

import iree.compiler.ir

from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

from ...utils import KStat, is_default_layout
from ..wrapper import OpWrapper
from .scope import Scope
from .sequence import Sequence


class Graph(Scope):
    def __init__(
        self,
        ops: Iterator[
            Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView, OpWrapper]
        ] = [],
        *args,
        **kwargs,
    ) -> Graph:
        super().__init__(*args, **kwargs)
        self._wrappers: Set[OpWrapper] = {OpWrapper(op, self) for op in ops}

    def contains(
        self, op: Union[OpWrapper, iree.compiler.ir.Operation, iree.compiler.ir.OpView]
    ) -> bool:
        if isinstance(op, OpWrapper):
            return op in self._wrappers
        else:
            return super().contains(op)

    def iter(self) -> Iterator[OpWrapper]:
        for wrapper in self._wrappers:
            yield wrapper

    def partitioned(self) -> List["Graph"]:
        graphs: List[Graph] = self._partitioned()
        for graph in graphs:
            assert graph.is_connected, f"Graph\n{graph}\nis not connected."
        return graphs

    def pathify(self, kstat: Optional[KStat] = None) -> List[Sequence]:
        assert self.is_connected, f"Graph\n{self}\nis not connected."
        dtable: Dict[OpWrapper, Tuple[Optional[OpWrapper], float]] = {}
        queue: List[OpWrapper] = [
            wrapper for wrapper in self._wrappers if wrapper.is_source
        ]
        while queue:
            wrapper: OpWrapper = queue.pop()
            if wrapper.is_source:
                dtable[wrapper] = (None, 0.0)
            else:
                deps: List[OpWrapper] = wrapper.scope_prevs
                if any(dep not in dtable for dep in deps):
                    continue
                prev, dist = max(
                    [(dep, dtable[dep][1]) for dep in deps], key=lambda x: x[1]
                )
                if kstat and wrapper.schedule_layout:
                    ktable: Dict[Tuple[Tuple[int, ...], ...], float] = kstat[
                        wrapper.entry
                    ]
                    [(_, timecost)] = [
                        (k, v) for k, v in ktable.items() if is_default_layout(k)
                    ]
                    dtable[wrapper] = (prev, dist + timecost)
                else:
                    dtable[wrapper] = (prev, dist + 1.0)
            for output in wrapper.scope_nexts:
                if output not in dtable:
                    queue += [output]
        destination, (prev, dist) = max(dtable.items(), key=lambda x: x[1][1])
        seq: Sequence = Sequence([destination])
        while prev:
            seq = seq.prepend(prev)
            prev, _ = dtable[prev]
        seq_set: Set[OpWrapper] = set(wrapper for wrapper in seq)
        graph: Graph = Graph()
        for wrapper in self._wrappers:
            if wrapper not in seq_set:
                graph += wrapper
        subgraphs: List[Graph] = graph.partitioned()
        seqs: List[Sequence] = [seq]
        for subgraph in subgraphs:
            seqs += subgraph.pathify()
        return seqs

    def put(
        self,
        op: Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView, OpWrapper],
    ) -> Graph:
        self._wrappers = {OpWrapper(op, self)} | self._wrappers
        return self

    @property
    def is_connected(self) -> bool:
        graphs: List[Graph] = self._partitioned()
        return len(graphs) == 1

    def _partitioned(self) -> List["Graph"]:
        wrappers: Set[OpWrapper] = {*self._wrappers}
        visited: Set[OpWrapper] = set()
        graphs: List[Graph] = []
        while wrappers:
            wrapper: OpWrapper = wrappers.pop()
            queue: List[OpWrapper] = [wrapper]
            graph: Graph = Graph()
            while queue:
                wrapper: OpWrapper = queue.pop()
                if wrapper in visited:
                    continue
                wrappers -= {wrapper}
                visited |= {wrapper}
                graph += wrapper
                queue += wrapper.scope_neighbors
            graphs += [graph]
        return graphs
