import ast

from typing import List
from collections.abc import Iterable
from onnxscript import ir
from onnxscript.rewriter._pattern_ir import (
    GraphPattern,
    NodeOutputPattern,
    ValuePattern,
)
from onnxscript.rewriter._rewrite_rule import (
    ReplacementPatternFunction,
    ReplacementSubgraph,
)
from onnxscript.rewriter.pattern import (
    MatchResult,
    OpsetPatternBuilder,
    RewriterContext,
    pattern_builder,
)


def has_internal_usage(usage):
    return "INTERNAL" in usage


def has_external_usage(usage):
    return "EXTERNAL" in usage


def classify_usage(value, nodes):
    usage = set()
    if value in nodes[0].graph.outputs:
        usage.add("EXTERNAL")
    for use in value.uses():
        user_node = use[0]
        if user_node in nodes:
            usage.add("INTERNAL")
        else:
            usage.add("EXTERNAL")
    return usage


def find_subgraph_inputs(nodes):
    inputs = set()
    initializers = set()
    for node in nodes:
        for ninput in node.inputs:
            if ninput in node.graph.inputs:
                inputs.add(ninput)
            elif any(ninput is init for init in node.graph.initializers):
                initializers.add(ninput)
            elif ninput.producer() == None:
                inputs.add(ninput)
            elif ninput.producer() not in nodes:
                inputs.add(ninput)

    return inputs, initializers


def find_subgraph_outputs(nodes):
    output = set()
    used_output = set()
    for node in nodes:
        for noutput in node.outputs:
            usage = classify_usage(noutput, nodes)
            if has_external_usage(usage):
                if has_internal_usage(usage):
                    used_output.add(noutput)
                else:
                    output.add(noutput)
    return [output, used_output]


def bGraphView(name, nodes):
    # check that all nodes belong to the same graph
    for node in nodes:
        if node.graph != nodes[0].graph:
            raise ValueError("All nodes must belong to the same graph")

    graph = nodes[0].graph

    print(f"Sorting nodes for GraphView: {name}")
    # sort the nodes in topological order
    graph.sort()
    # add the nodes to the new list in the order they appear in the graph
    sorted_nodes = []
    for node in graph._nodes:
        if node in nodes:
            sorted_nodes.append(node)
    nodes = sorted_nodes

    [view_inputs, view_initializers] = find_subgraph_inputs(nodes)
    [view_outputs, used_outputs] = find_subgraph_outputs(nodes)

    for used_output in used_outputs:
        producer_node = used_output.producer()
        nodes.remove(producer_node)
        for output in producer_node.outputs:
            usage = classify_usage(output, nodes)
            if has_internal_usage(usage):
                view_inputs.add(output)
            if has_external_usage(usage):
                if output in view_outputs:
                    view_outputs.remove(output)

    return ir.GraphView(
        name=name,
        inputs=view_inputs,
        outputs=view_outputs,
        nodes=nodes,
        initializers=view_initializers,
    )


class PytorchMetadataNode:
    def __init__(self, node):
        self._node = node

        if self.check_node_metadata_exists():
            self.instance_metadata = ast.literal_eval(
                self._node.metadata_props["pkg.torch.onnx.name_scopes"]
            )
            self.class_metadata = ast.literal_eval(
                self._node.metadata_props["pkg.torch.onnx.class_hierarchy"]
            )

    def check_node_metadata_exists(self):
        if (
            "pkg.torch.onnx.name_scopes" in self._node.metadata_props
            and "pkg.torch.onnx.class_hierarchy" in self._node.metadata_props
        ):
            return True
        else:
            return False

    def is_last_level(self, level):
        if len(self.instance_metadata) - 1 == level:
            return True
        else:
            return False

    def get_instance_name(self, depth=0):
        if depth >= len(self.instance_metadata):
            return None
        else:
            return self.instance_metadata[depth]

    def get_class_name(self, depth=0):
        if depth >= len(self.instance_metadata):
            return None
        else:
            return self.class_metadata[depth]


class PytorchHierarchyNode:
    def __init__(self):
        self.instance_name = None
        self.module_type = None
        self.children = []
        self.nodes = []

    def print_hierarchy(self, instance_hierarchy: List[str] = None):
        if instance_hierarchy is None:
            instance_hierarchy = []
        instance_hierarchy.append(self.instance_name)

        for child in self.children:
            child.print_hierarchy(list(instance_hierarchy))

        for node in self.nodes:
            print(
                f"Node: {node._node.name}, Instance: {'/'.join(instance_hierarchy)}, Module: {self.module_type}"
            )

    def get_unwrapped_nodes(self):
        # Return _node from self._nodes
        return [node._node for node in self.nodes]

    # Checks if the search hierarchy matches the instance hierarchy
    def hierarchy_matches(self, search_hierarchy: List[str], instance_hierarchy: List[str] = []):
        search_length = min(len(search_hierarchy), len(instance_hierarchy))
        for i in range(search_length):
            if search_hierarchy[i] != instance_hierarchy[i]:
                return False
        return True

    # Return all nodes from the given name hierarchy on down
    def get_nodes(self, search_hierarchy: List[str], instance_hierarchy: List[str] = None):
        if instance_hierarchy is None:
            instance_hierarchy = []

        nodes_to_return = []
        # base case for recursion
        # 1 - search_hierarchy does not match instance_hierarchy
        instance_hierarchy.append(self.instance_name)

        if not self.hierarchy_matches(search_hierarchy, instance_hierarchy):
            return []

        for child in self.children:
            child_nodes = child.get_nodes(search_hierarchy, list(instance_hierarchy))
            nodes_to_return.extend(child_nodes)

        if len(instance_hierarchy) >= len(search_hierarchy):
            nodes_to_return.extend(self.get_unwrapped_nodes())

        return nodes_to_return

    def add_node(self, node, level=0):
        if not isinstance(node, PytorchMetadataNode):
            node = PytorchMetadataNode(node)
            if node.check_node_metadata_exists() is False:
                return False

        if self.instance_name is None:
            self.instance_name = node.get_instance_name(level)
        if self.module_type is None:
            self.module_type = node.get_class_name(level)

        # check that instance name and module type match
        if self.instance_name != node.get_instance_name(level):
            return False
        if self.module_type != node.get_class_name(level):
            return False
        # if this is the last level of the hierarchy, add the node to this node
        # otherwise find the child node that matches the next level of the hierarchy
        # and add the node to that child
        if node.is_last_level(level):
            self.nodes.append(node)
            return True
        else:
            for child in self.children:
                if child.instance_name == node.get_instance_name(level + 1):
                    return child.add_node(node, level + 1)

            # if no child matches the next level of the hierarchy, create a new child node
            new_child = PytorchHierarchyNode()
            new_child.instance_name = node.get_instance_name(level + 1)
            new_child.module_type = node.get_class_name(level + 1)
            self.children.append(new_child)
            return new_child.add_node(node, level + 1)


def direct_convert_ir_graph_to_pattern(graph):
    # Transform IR values to ValuePatterns
    vmap = {}
    for input in graph.inputs:
        vmap[input] = ValuePattern(input.name)

    for init in graph.initializers:
        vmap[init] = ValuePattern(init.name)

    for node in graph._nodes:
        if node.op_type == "Constant":
            vmap[node.outputs[0]] = ValuePattern(node.outputs[0].name)

    builder = OpsetPatternBuilder("", record=True)

    with pattern_builder(builder):
        for node in graph._nodes:
            ninputs = []
            for ninput in node.inputs:
                ninputs.append(vmap[ninput])

            vp_outputs = builder.__getattr__(node.op_type)(
                *ninputs, _domain=node.domain, _outputs=len(node.outputs)
            )

            if isinstance(vp_outputs, NodeOutputPattern):
                vp_outputs = [vp_outputs]

            for vp_output in iter(vp_outputs):
                vmap[node.outputs[vp_output.output_index]] = vp_output

    pinputs = []
    for input in graph.inputs:
        pinputs.append(vmap[input])

    # build graph outputs
    poutputs = []
    for output in graph.outputs:
        poutputs.append(vmap[output])

    return GraphPattern(inputs=pinputs, outputs=poutputs, nodes=builder.nodes())


def remove_input_from_node(node, inp):
    node._inputs = [x for x in node._inputs if x is not inp]
    inp._remove_usage(node)


def same(input_list):
    return len(set(input_list)) == 1


def vdisconnect(value):
    value._uses = {}
    value._producer = None
    value._index = None
    value._graph = None
    return value


class ReplacementPatternGraph(ReplacementPatternFunction):
    def __init__(self, ir_graph):
        self._graph = ir_graph

    def get_replacement(self, match: MatchResult) -> ReplacementSubgraph | None:
        context = RewriterContext()
        # match.bindings is dictionary of value_name (str) in replacement subgraph pattern (i.e. ir_graph -> IR Value in actual graph)
        vvmap = (
            {}
        )  # Dictionary mapping values in replacement subgraph pattern -> values in the replacement subgraph

        for value in self._graph.inputs:
            if value.name in match.bindings:
                vvmap[value] = match.bindings[value.name]
            else:
                vvmap[value] = value

        for node in self._graph._nodes:
            ninputs = []
            for ninput in node.inputs:
                ninputs.append(vvmap[ninput])

            coutput = context.__getattr__(node.op_type)(
                *ninputs,
                **node.attributes,
                _outputs=len(node.outputs),
                _domain=node.domain,
                _version=node.version,
            )
            if not isinstance(coutput, Iterable):
                coutput = [coutput]

            for i, cout in enumerate(coutput):
                cout._type = node.outputs[i].type
                cout._shape = node.outputs[i].shape
                vvmap[node.outputs[cout.index()]] = cout

        new_outputs = [vvmap[x] for x in self._graph.outputs]
        return ReplacementSubgraph(
            match, new_outputs, context.nodes, context.initializers, context.used_opsets
        )


def find_nodes_of_optype(graph, layername):
    nodes = []
    for node in ir.traversal.RecursiveGraphIterator(graph):
        if node.op_type == layername:
            nodes.append(node)
    return nodes


def build_constant_from_tensor(name, tensor):
    value_attribute = ir.Attr(name="value", type=ir.AttributeType.TENSOR, value=tensor)
    ir_value_out = ir.Value(name=name + "_out", type=ir.TensorType(tensor.dtype))
    return ir.Node(
        "", "Constant", name=name, inputs=[], outputs=[ir_value_out], attributes=[value_attribute]
    )


def build_concat_node_from_inputs(inputs):
    axis = ir.Attr(name="axis", type=ir.AttributeType.INT, value=0)

    ndim = len(inputs) * inputs[0].shape.dims[0]
    output_shape = ir.Shape([ndim, *inputs[0].shape.dims[1:]])
    output = ir.Value(name=f"{inputs[0].name}_concat", shape=output_shape, type=inputs[0].type)
    return ir.Node("", "Concat", inputs=inputs, attributes=[axis], outputs=[output])


def build_reshape_node(inp, reshape_shape):
    reshape_out = ir.Value(name=f"{inp.name}_reshape", type=inp.type)
    return ir.Node("", "Reshape", inputs=[inp, reshape_shape], outputs=[reshape_out])
