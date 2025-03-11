import iree.compiler.ir

from typing import Dict, List, Optional, Set, Tuple, Type, TypeVar, TYPE_CHECKING, Union

from .wrapper import (
    DestinationOpWrapper,
    InputOpWrapper,
    OpWrapper,
    OutputOpWrapper,
    InterfaceOpWrapper,
    SourceOpWrapper,
)

OpWrapperSubclassType = TypeVar("OpWrapperSubclassType", bound=OpWrapper)


class Graph(object):
    def __init__(self, wrappers: List[OpWrapper], *args, **kwargs) -> "Graph":
        super().__init__(*args, **kwargs)
        self._wrappers: List[OpWrapper] = wrappers
        for wrapper in self._wrappers:
            wrapper._scope = self
        if len(wrappers) == 1:
            [wrapper] = wrappers
            assert isinstance(
                wrapper, InterfaceOpWrapper
            ), f"Single op {wrapper._op} must be `InterfaceOpWrapper`, while it's {type(wrapper)}."

    def __contains__(
        self, op: Union[OpWrapper, iree.compiler.ir.Operation, iree.compiler.ir.OpView]
    ) -> bool:
        return self.contains(op)

    def __repr__(self) -> str:
        return "\n".join(map(lambda wrapper: wrapper._op.get_asm(), self._wrappers))

    def contains(
        self,
        op: Union[OpWrapper, iree.compiler.ir.Operation, iree.compiler.ir.OpView],
        cls: Type[OpWrapperSubclassType] = OpWrapper,
    ) -> bool:
        assert issubclass(cls, OpWrapper), f"{cls} is not a subclass of `OpWrapper`."
        if isinstance(op, iree.compiler.ir.Operation):
            return any(
                map(
                    lambda wrapper: wrapper._op == op.opview
                    and isinstance(wrapper, cls),
                    self._wrappers,
                )
            )
        elif isinstance(op, iree.compiler.ir.OpView):
            return any(
                map(
                    lambda wrapper: wrapper._op == op and isinstance(wrapper, cls),
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
        cls: Type[OpWrapperSubclassType] = OpWrapper,
    ) -> OpWrapper:
        assert issubclass(cls, OpWrapper), f"{cls} is not a subclass of `OpWrapper`."
        if isinstance(op, iree.compiler.ir.Operation):
            [wrapper] = list(
                filter(
                    lambda wrapper: wrapper._op == op.opview
                    and isinstance(wrapper, cls),
                    self._wrappers,
                )
            )
        elif isinstance(op, iree.compiler.ir.OpView):
            [wrapper] = list(
                filter(
                    lambda wrapper: wrapper._op == op and isinstance(wrapper, cls),
                    self._wrappers,
                )
            )
        elif isinstance(op, OpWrapper):
            [wrapper] = list(
                filter(
                    lambda wrapper: wrapper == op and isinstance(wrapper, cls),
                    self._wrappers,
                )
            )
        else:
            raise TypeError(f"Unsupported type {type(op)} for Graph.get")
        return wrapper

    def get_inputs(self, op: OpWrapper) -> List[OpWrapper]:
        assert op in self, f"Op {op._op} is not in the graph."
        return [
            self.get(input.owner.opview, OutputOpWrapper)
            for input in op.inputs
            if self.contains(input.owner.opview, OutputOpWrapper)
        ]

    def get_outputs(self, op: OpWrapper) -> List[OpWrapper]:
        assert op in self, f"Op {op._op} is not in the graph."
        return [
            self.get(use.owner, InputOpWrapper)
            for output in op.outputs
            for use in output.uses
            if self.contains(use.owner, InputOpWrapper)
        ]

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
        wrappers: Set[OpWrapper] = self._wrappers_set
        visited: Set[OpWrapper] = set()
        graphs: List[Graph] = []
        while wrappers:
            wrapper: OpWrapper = wrappers.pop()
            queue: List[OpWrapper] = [wrapper]
            ops: List[OpWrapper] = []
            while queue:
                wrapper: OpWrapper = queue.pop()
                if wrapper in visited:
                    continue
                wrappers -= {wrapper}
                visited |= {wrapper}
                ops += [wrapper]
                queue += wrapper.scope_neighbors
            assert ops, "Ops cannot be empty."
            graph: Graph = Graph(ops)
            graphs += [graph]
        return graphs

    def pathify(self) -> List[OpWrapper]:
        assert self.is_connected, f"Graph\n{self}\nis not connected."
        dtable: Dict[OpWrapper, Tuple[Optional[OpWrapper], int]] = {}
        queue: List[OpWrapper] = [
            wrapper
            for wrapper in self._wrappers
            if isinstance(wrapper, SourceOpWrapper)
        ]
        while queue:
            wrapper: OpWrapper = queue.pop()
            if isinstance(wrapper, SourceOpWrapper):
                dtable[wrapper] = (None, 0)
            else:
                deps: List[OpWrapper] = self.get_inputs(wrapper)
                if any(dep not in dtable for dep in deps):
                    continue
                prev, dist = max(
                    [(dep, dtable[dep][1]) for dep in deps], key=lambda x: x[1]
                )
                dtable[wrapper] = (prev, dist + 1)
            for output in self.get_outputs(wrapper):
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
        remains: List[OpWrapper] = []
        for remain in remains_set:
            has_inputs: bool = any(
                input in remains_set for input in self.get_inputs(remain)
            )
            has_outputs: bool = any(
                output in remains_set for output in self.get_outputs(remain)
            )
            if not has_inputs and not has_outputs:
                remains += [InterfaceOpWrapper(remain._op)]
            elif not has_inputs and has_outputs:
                remains += [SourceOpWrapper(remain._op)]
            elif has_inputs and not has_outputs:
                remains += [DestinationOpWrapper(remain._op)]
            else:
                remains += [remain]
        remained: Graph = Graph(remains)
        subgraphs: List[Graph] = remained.partitioned()
        seqs: List[List[OpWrapper]] = [seq]
        for subgraph in subgraphs:
            seqs += subgraph.pathify()
        return seqs

    @property
    def _wrappers_set(self) -> Set[OpWrapper]:
        return set(self._wrappers)

    @property
    def tensors(self) -> Set[iree.compiler.ir.Value]:
        return set(tensor for wrapper in self._wrappers for tensor in wrapper.neighbors)

    @property
    def tensor_names(self) -> Set[str]:
        return set(
            name for wrapper in self._wrappers for name in wrapper.neighbor_names
        )
