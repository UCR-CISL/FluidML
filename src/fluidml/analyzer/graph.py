import iree.compiler.ir

from typing import List, Set, Type, TypeVar, Union

from .op_wrapper import (
    DestinationOpWrapper,
    InputOpWrapper,
    OpWrapper,
    OutputOpWrapper,
    SourceOpWrapper,
)

OpWrapperSubclassType = TypeVar("OpWrapperSubclassType", bound=OpWrapper)


class Graph(object):
    def __init__(self, wrappers: List[OpWrapper], *args, **kwargs) -> "Graph":
        super().__init__(*args, **kwargs)
        self._wrappers: List[OpWrapper] = wrappers

    def __contains__(
        self, op: Union[OpWrapper, iree.compiler.ir.Operation, iree.compiler.ir.OpView]
    ) -> bool:
        if isinstance(op, iree.compiler.ir.Operation):
            return any(map(lambda wrapper: wrapper._op == op.opview, self._wrappers))
        elif isinstance(op, iree.compiler.ir.OpView):
            return any(map(lambda wrapper: wrapper._op == op, self._wrappers))
        elif isinstance(op, OpWrapper):
            return op in self._wrappers
        else:
            raise TypeError(f"Unsupported type {type(op)} for Graph.__contains__")

    def __repr__(self) -> str:
        return "\n".join(map(lambda wrapper: wrapper._op.get_asm(), self._wrappers))

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
            if input.owner in self
        ]

    def get_outputs(self, op: OpWrapper) -> List[OpWrapper]:
        assert op in self, f"Op {op._op} is not in the graph."
        return [
            self.get(use.owner, InputOpWrapper)
            for output in op.outputs
            for use in output.uses
            if use.owner in self
        ]

    def get_neighbors(self, op: OpWrapper) -> List[OpWrapper]:
        return self.get_inputs(op) + self.get_outputs(op)

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
                for op_wrapper in self.get_neighbors(wrapper):
                    queue += [op_wrapper]
            assert ops, "Ops cannot be empty."
            graph: Graph = Graph(ops)
            graphs += [graph]
        return graphs

    @property
    def _wrappers_set(self) -> Set[OpWrapper]:
        return set(self._wrappers)

    @property
    def tensors(self) -> Set[iree.compiler.ir.Value]:
        return set(tensor for wrapper in self._wrappers for tensor in wrapper.tensors)

    @property
    def tensor_names(self) -> Set[str]:
        return set(name for wrapper in self._wrappers for name in wrapper.tensor_names)
