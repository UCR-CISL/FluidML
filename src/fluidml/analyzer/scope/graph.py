import iree.compiler.ir

from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

from ..wrapper import OpWrapper
from .scope import Scope
from .sequence import Sequence


class Graph(Scope):
    def __init__(self, wrappers: Iterator[OpWrapper] = [], *args, **kwargs) -> "Graph":
        super().__init__(*args, **kwargs)
        self._wrappers: Set[OpWrapper] = {
            OpWrapper(wrapper._op, self) for wrapper in wrappers
        }

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

    def pathify(self) -> List[Sequence]:
        assert self.is_connected, f"Graph\n{self}\nis not connected."
        dtable: Dict[OpWrapper, Tuple[Optional[OpWrapper], int]] = {}
        queue: List[OpWrapper] = [
            wrapper for wrapper in self._wrappers if wrapper.is_source
        ]
        while queue:
            wrapper: OpWrapper = queue.pop()
            if wrapper.is_source:
                dtable[wrapper] = (None, 0)
            else:
                deps: List[OpWrapper] = wrapper.scope_inputs
                if any(dep not in dtable for dep in deps):
                    continue
                prev, dist = max(
                    [(dep, dtable[dep][1]) for dep in deps], key=lambda x: x[1]
                )
                dtable[wrapper] = (prev, dist + 1)
            for output in wrapper.scope_outputs:
                if output not in dtable:
                    queue += [output]
        destination, (prev, dist) = max(dtable.items(), key=lambda x: x[1][1])
        seq: Sequence = Sequence([destination])
        while prev:
            seq = seq.prepend(prev)
            prev, _ = dtable[prev]
        seq_set: Set[OpWrapper] = set(wrapper for wrapper in seq)
        remains_set: Set[OpWrapper] = {
            wrapper for wrapper in self._wrappers if wrapper not in seq_set
        }
        graph: Graph = Graph()
        for remain in remains_set:
            has_inputs: bool = any(
                input in remains_set for input in remain.scope_inputs
            )
            has_outputs: bool = any(
                output in remains_set for output in remain.scope_outputs
            )
            if not has_inputs and not has_outputs:
                graph += remain._op
            elif not has_inputs and has_outputs:
                graph += remain._op
            elif has_inputs and not has_outputs:
                graph += remain._op
            else:
                graph += remain
        subgraphs: List[Graph] = graph.partitioned()
        seqs: List[Sequence] = [seq]
        for subgraph in subgraphs:
            seqs += subgraph.pathify()
        return seqs

    def put(
        self,
        op: Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView, OpWrapper],
    ) -> "Graph":
        if isinstance(op, iree.compiler.ir.Operation):
            wrapper: OpWrapper = OpWrapper(op.op_view, self)
        elif isinstance(op, iree.compiler.ir.OpView):
            wrapper: OpWrapper = OpWrapper(op, self)
        elif isinstance(op, OpWrapper):
            wrapper: OpWrapper = OpWrapper(op._op, self)
        else:
            raise TypeError(
                f"Unsupported type {type(op)} for `{self.__class__.__name__}.put`."
            )
        self._wrappers = {wrapper} | self._wrappers
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
