import iree.compiler.ir

from functools import cached_property
from typing import List, Set, Union

from .op_wrapper import OpWrapper


class Graph(object):
    def __init__(
        self, ops: List[Union[OpWrapper, iree.compiler.ir.OpView]], *args, **kwargs
    ) -> "Graph":
        super().__init__(*args, **kwargs)
        self.wrappers: List[OpWrapper] = []
        for op in ops:
            if isinstance(op, OpWrapper):
                self.wrappers += [op]
            elif isinstance(op, iree.compiler.ir.OpView):
                self.wrappers += [OpWrapper.from_op(op)]
            else:
                raise NotImplementedError(
                    f"The type of op {op} is not supported as a member of `Graph` yet, which is {type(op)}."
                )

    def __repr__(self) -> str:
        return "\n".join(map(lambda wrapper: wrapper.op.get_asm(), self.wrappers))

    def get_inputs(
        self, op: Union[OpWrapper, iree.compiler.ir.OpView]
    ) -> List[iree.compiler.ir.Value]:
        if isinstance(op, iree.compiler.ir.OpView):
            op: OpWrapper = OpWrapper.from_op(op)
        assert op in self.wrappers, f"Op {op} is not in the graph."
        inputs: List[iree.compiler.ir.Value] = [input for input in op.inputs]
        return inputs

    def get_outputs(
        self, op: Union[OpWrapper, iree.compiler.ir.OpView]
    ) -> List[iree.compiler.ir.Value]:
        if isinstance(op, iree.compiler.ir.OpView):
            op: OpWrapper = OpWrapper.from_op(op)
        assert op in self.wrappers, f"Op {op} is not in the graph."
        outputs: List[iree.compiler.ir.Value] = [output for output in op.outputs]
        return outputs

    def get_prevs(
        self, op: Union[OpWrapper, iree.compiler.ir.OpView]
    ) -> List[iree.compiler.ir.OpView]:
        return [
            input.owner.opview
            for input in self.get_inputs(op)
            if input.owner in self.wrappers_set
        ]

    def get_nexts(
        self, op: Union[OpWrapper, iree.compiler.ir.OpView]
    ) -> List[iree.compiler.ir.OpView]:
        return [
            use.owner
            for output in self.get_outputs(op)
            for use in output.uses
            if use.owner in self.wrappers_set
        ]

    def is_connected(self) -> bool:
        graphs: List[Graph] = self._partitioned()
        return len(graphs) == 1

    def partitioned(self) -> List["Graph"]:
        graphs: List[Graph] = self._partitioned()
        for graph in graphs:
            assert graph.is_connected(), f"Graph\n{graph}\nis not connected."
        return graphs

    def _partitioned(self) -> List["Graph"]:
        wrappers: Set[OpWrapper] = set(self.wrappers)
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
                for op_view in self.get_prevs(wrapper) + self.get_nexts(wrapper):
                    op_wrapper: OpWrapper = OpWrapper.from_op(op_view)
                    queue += [op_wrapper]
            assert ops, "Ops cannot be empty."
            graph: Graph = Graph(ops)
            graphs += [graph]
        return graphs

    @cached_property
    def wrappers_set(self) -> Set[OpWrapper]:
        return set(self.wrappers)

    @cached_property
    def tensors(self) -> Set[iree.compiler.ir.Value]:
        return set(tensor for wrapper in self.wrappers for tensor in wrapper.tensors)

    @cached_property
    def tensor_names(self) -> Set[str]:
        return set(name for wrapper in self.wrappers for name in wrapper.tensor_names)
