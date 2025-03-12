import iree.compiler.ir

from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from .wrapper import OpWrapper


class Graph(object):
    def __init__(self, wrappers: Sequence[OpWrapper] = [], *args, **kwargs) -> "Graph":
        super().__init__(*args, **kwargs)
        self._wrappers: Set[OpWrapper] = {
            OpWrapper(wrapper._op, self) for wrapper in wrappers
        }

    def __contains__(
        self, op: Union[OpWrapper, iree.compiler.ir.Operation, iree.compiler.ir.OpView]
    ) -> bool:
        return self.contains(op)

    def __iadd__(
        self,
        op: Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView, OpWrapper],
    ) -> "Graph":
        return self.put(op)

    def __repr__(self) -> str:
        return "\n".join(map(lambda wrapper: wrapper._op.get_asm(), self._wrappers))

    def contains(
        self,
        op: Union[OpWrapper, iree.compiler.ir.Operation, iree.compiler.ir.OpView],
    ) -> bool:
        if isinstance(op, iree.compiler.ir.Operation):
            return any(
                map(
                    lambda wrapper: wrapper._op == op.opview,
                    self._wrappers,
                )
            )
        elif isinstance(op, iree.compiler.ir.OpView):
            return any(
                map(
                    lambda wrapper: wrapper._op == op,
                    self._wrappers,
                )
            )
        elif isinstance(op, OpWrapper):
            return op in self._wrappers
        else:
            raise TypeError(f"Unsupported type {type(op)} for `Graph.contains.`")

    def get(
        self,
        op: Union[iree.compiler.ir.Operation, iree.compiler.ir.OpView],
    ) -> OpWrapper:
        if isinstance(op, iree.compiler.ir.Operation):
            [wrapper] = list(
                filter(
                    lambda wrapper: wrapper._op == op.opview,
                    self._wrappers,
                )
            )
        elif isinstance(op, iree.compiler.ir.OpView):
            [wrapper] = list(
                filter(
                    lambda wrapper: wrapper._op == op,
                    self._wrappers,
                )
            )
        elif isinstance(op, OpWrapper):
            [wrapper] = list(
                filter(
                    lambda wrapper: wrapper == op,
                    self._wrappers,
                )
            )
        else:
            raise TypeError(f"Unsupported type {type(op)} for Graph.get")
        return wrapper

    def get_inputs(self, op: OpWrapper) -> List[OpWrapper]:
        assert op in self, f"Op {op._op} is not in the graph."
        return [
            self.get(input.owner.opview)
            for input in op.inputs
            if input.owner.opview in self
        ]

    def get_outputs(self, op: OpWrapper) -> List[OpWrapper]:
        assert op in self, f"Op {op._op} is not in the graph."
        return [
            self.get(use.owner)
            for output in op.outputs
            for use in output.uses
            if use.owner in self
        ]

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
            raise TypeError(f"Unsupported type {type(op)} for `Graph.put`.")
        self._wrappers = {wrapper} | self._wrappers
        return self

    @property
    def is_connected(self) -> bool:
        graphs: List[Graph] = self._partitioned()
        return len(graphs) == 1

    def partitioned(self) -> List["Graph"]:
        graphs: List[Graph] = self._partitioned()
        for graph in graphs:
            assert graph.is_connected, f"Graph\n{graph}\nis not connected."
        return graphs

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

    def pathify(self) -> List[OpWrapper]:
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
        seq: List[OpWrapper] = [destination]
        while prev:
            seq = [prev] + seq
            prev, _ = dtable[prev]
        seq_set: Set[OpWrapper] = set(seq)
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
        seqs: List[List[OpWrapper]] = [seq]
        for subgraph in subgraphs:
            seqs += subgraph.pathify()
        return seqs
