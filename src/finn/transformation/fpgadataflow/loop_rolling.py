import copy

from typing import List, Tuple
from onnxscript import ir

import numpy as np

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.fold_constants import FoldConstants

from onnxscript.utils import graph_view_utils as gvu
from onnxscript.rewriter import pattern
from onnxscript.rewriter import pattern_builder_jsm as pb
from onnxscript.rewriter import rewrite



import onnxscript
import onnx
import qonnx

def get_constant_from_value(value):
    """
    Get the constant value of a tensor.
    """
    # Handle input and/or inititalizer values
    if value.producer() is None:
        return value.const_value.numpy()
    elif value.producer().op_type == "Constant":
        return value.producer().attributes['value'].value.numpy()


def same_values(inputs):
    """
    Check if all inputs have the same constant value.
    """
    if not inputs:
        return False

    first_value = get_constant_from_value(inputs[0])

    for inp in inputs[1:]:
        if not np.array_equal(first_value, get_constant_from_value(inp)):
            return False

    return True


def build_loop_replace_pattern(graph, LoopBody):

    nodes = pb.find_nodes_of_optype(graph, LoopBody.function.name)
    iterations = len(nodes)

    graph_nodes = []
    loop_inputs = []

    graph_inputs  = []
    const_indexes = []
    for i, LoopInputType in enumerate(LoopBody.signature):

        if LoopInputType == pb.LoopBodyInputType.PARAMETER:
            # Build Concat Node
            concat_inputs  = []
            for node in nodes:
                nvalue = pb.vdisconnect(copy.copy(node.inputs[i]))
                graph_inputs.append(nvalue)
                concat_inputs.append(nvalue)

            # if inputs are scalars then we need to manually perform a concat
            if len(concat_inputs[0].shape.dims) == 0:
                const_values_as_numpy = np.array([x.const_value.numpy() for x in concat_inputs])
                const_values_as_tensor = ir.Tensor(const_values_as_numpy)
                const_values_as_const_node = pb.build_constant_from_tensor(f'concat_{i}', const_values_as_tensor)
                graph_nodes.append(const_values_as_const_node)

                reshape_shape_const = pb.build_constant_from_tensor(f'reshape_shape_const_{i}',
                                                             ir.Tensor(np.array([len(concat_inputs), 1])))
                reshape_node  = pb.build_reshape_node(const_values_as_const_node.outputs[0], reshape_shape_const.outputs[0])
            else:
                concat_node = pb.build_concat_node_from_inputs(concat_inputs)
                graph_nodes.append(concat_node)
                # Build Reshape Node
                reshape_shape_const = pb.build_constant_from_tensor(f'reshape_shape_const_{i}', ir.Tensor(np.array([len(nodes),*concat_inputs[0].shape.dims])))

                reshape_node = pb.build_reshape_node(concat_node.outputs[0], reshape_shape_const.outputs[0])
            graph_nodes.append(reshape_shape_const)
            graph_nodes.append(reshape_node)
            loop_inputs.append(reshape_node.outputs[0])
        elif LoopInputType == pb.LoopBodyInputType.CONSTANT:
            const_indexes.append(i)

            # if input is constant push down into loop body graph
            constant_producer       = nodes[0].inputs[i].producer()
            constant_producer_value = constant_producer.outputs[0]

            # build new node and value for the loop body graph
            new_const_prod_value    = ir.Value(name=constant_producer_value.name + "_push_down",
                                                type=constant_producer_value.type,
                                                shape=constant_producer_value.shape)
            new_const_prod_node     = ir.Node(name=constant_producer.name + "_push_down",
                                               domain=constant_producer.domain,
                                               inputs=constant_producer.inputs,
                                               op_type="Constant",
                                               attributes=constant_producer.attributes.values(),
                                               outputs=[new_const_prod_value])
            # add new nodes to loop body
            LoopBody.function.append(new_const_prod_node)
            LoopBody.function.sort()

            for usage in LoopBody.function.inputs[i].uses():
                usage.node.replace_input_with(usage.idx, new_const_prod_value)

            LoopBody.function.sort()

        elif LoopInputType == pb.LoopBodyInputType.ACTIVATION:
            cinp = pb.vdisconnect(copy.copy(LoopBody.function.inputs[i]))
            graph_inputs.append(cinp)
            loop_inputs.append(cinp)

    for i in reversed(const_indexes):
        del LoopBody.function.inputs[i]
        del LoopBody.signature[i]

    loop_outputs = []
    graph_outputs = []
    for out in LoopBody.function.outputs:
        output = pb.vdisconnect(copy.copy(out))
        loop_outputs.append(output)
        graph_outputs.append(output)

    g_loop_body = LoopBody.function._graph
    odt = g_loop_body.outputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"]
    idt = odt
    body_attr = ir.Attr(name='body', type=ir.AttributeType.GRAPH, value=LoopBody.function._graph)
    backend_attr = ir.Attr(name='backend', type=ir.AttributeType.STRING, value='fpgadataflow')
    iteration = ir.Attr(name='iteration', type=ir.AttributeType.INT, value=iterations)
    inputdatatype_attr = ir.Attr(name='inputDataType', type=ir.AttributeType.STRING, value=idt)
    outputdatatype_attr = ir.Attr(name='outputDataType', type=ir.AttributeType.STRING, value=odt)

    finn_loop_node = ir.Node("finn.custom_op.fpgadataflow.rtl",
                             'FINNLoop',
                             inputs=loop_inputs,
                             attributes=[body_attr, backend_attr, iteration, inputdatatype_attr, outputdatatype_attr],
                             outputs=loop_outputs,
                             graph=None)

    graph_nodes.append(finn_loop_node)

    graph = ir.Graph(name='loop_replace',nodes=graph_nodes, inputs=graph_inputs, outputs=graph_outputs)

    graph.sort()

    model = ir.serde.serialize_model(ir.Model(graph, ir_version=10))
    onnx.save(model, 'replacementgraph.onnx')

    return pb.ReplacementPatternGraph(graph)




class LoopExtraction(Transformation):
    def __init__(self, hierarchy_list : List[str]):
        super().__init__()

        assert isinstance(hierarchy_list, list), "Hierarchy list must be a list of strings"
        assert all(isinstance(item, str) for item in hierarchy_list), "All items in hierarchy list must be strings"
        self.hierarchy_list = hierarchy_list
        self.loop_body_template = None

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        # Apply the loop extraction transformation
        # Extract the Loop Body from ONNX metadata
        model_ir    = onnxscript.ir.serde.deserialize_model(model.model)
        graph       = model_ir.graph

        P = gvu.PytorchHierarchyNode()
        unadded_nodes = []
        for node in graph._nodes:
            added = P.add_node(node)
            if not added:
                unadded_nodes.append(node)
        P.print_hierarchy()
        print(f"Total nodes: {len(graph._nodes)}")
        print(f"Unadded nodes: {len(unadded_nodes)}")
        # Handle the nodes that have no metadata from PyTorch
        # by finding a neighboring node and using its metadata
        # use predecessor first than successor nodes
        # probably should provide a more robust selection method
        # in the future
        for node in unadded_nodes:
            print(f"adding metadata for node {node.name}...")
            preds = node.predecessors()
            succs = node.successors()
            if len(preds) > 0:
                mnode = preds[0]
            elif len(succs) > 0:
                mnode = succs[0]
            else:
                print("error: could not find metadata for node")
                exit(1)
            node.metadata_props['pkg.torch.onnx.name_scopes'] = mnode.metadata_props['pkg.torch.onnx.name_scopes']
            node.metadata_props['pkg.torch.onnx.class_hierarchy'] = mnode.metadata_props['pkg.torch.onnx.class_hierarchy']
            print(f"added metadata for node {node.name}")

            assert(P.add_node(node))

        nodes = P.get_nodes(self.hierarchy_list)
        print(f"Nodes in layer 0: {len(nodes)}")
        loop_body_graph_view = gvu.bGraphView(f'loop-body', nodes)
        print(f"Layer 0 graph view: {len(loop_body_graph_view._nodes)}")
        loop_body_model = onnxscript.ir.Model(loop_body_graph_view, ir_version=10)
        proto = onnxscript.ir.serde.serialize_model(loop_body_model)
        onnx.save(proto, 'loop-body-template.onnx')
        print("Load Loop Body Template")
        self.loop_body_template = pb.LoopBodyTemplate('loop-body-template.onnx')

        # Replace instances of the loop body with a function call to the loop body
        change_layers_to_function_calls = pattern.RewriteRule(
        self.loop_body_template.pattern,
        self.loop_body_template.function_replace
        )
        print("Replacing layers with function calls")

        model_layers_replaced = rewrite(
            model_ir,
            pattern_rewrite_rules = [change_layers_to_function_calls]
        )

        model_layers_replaced.functions[self.loop_body_template.function.identifier()] = self.loop_body_template.function
        model_layers_replaced.graph.opset_imports['loop']=0

        model_proto = onnxscript.ir.serde.serialize_model(model_layers_replaced)
        model.model = model_proto
        onnx.save(model_proto, 'simple_module_layers_replaced.onnx')


        return (model, False)


class LoopRolling(Transformation):
    """Boilerplate Transformation for loop rolling in fpgadataflow."""

    def __init__(self, loop_body_template):
        super().__init__()
        self.loop_body_template = loop_body_template

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:

        model_ir = onnxscript.ir.serde.deserialize_model(model.model)
        graph = model_ir.graph
        LoopBody = self.loop_body_template

        #################################
        ## I/O Normalization for Loop Body
        #################################
        graph.sort()

        # get the consecutive node layers
        # TODO: write a check to ensure that there is only one
        #       set of consecutive nodes.
        nodes = pb.find_nodes_of_optype(graph, LoopBody.function.name)

        # Loop through all the nodes (execept the last one) and
        # identify the input to output pairs
        input_swaps = []
        for i in range(len(nodes)-1):
            a_node = nodes[i]
            b_node = nodes[i+1]

            for a_out in a_node.outputs:
                # Require that outputs of a have a single use of b_node
                assert(len(a_out.uses()) == 1)
                assert(a_out.uses()[0][0] is b_node)

                a_use_index = a_out.uses()[0][1]
                input_swap = (a_out.index(), a_use_index)
                if i == 0:
                    # add swaps from the first node
                    input_swaps.append(input_swap)
                else:
                    # check that they are the same in the rest
                    assert(input_swap in input_swaps)

        # apply the input swaps to each nodes
        for node in nodes:
            for swap in input_swaps:
                a = node.inputs[swap[0]]
                b = node.inputs[swap[1]]
                node.replace_input_with(swap[0], b)
                node.replace_input_with(swap[1], a)

        # apply the input swaps to the function graph
        # mark swapped nodes as activations
        activations = 0
        for swap in input_swaps:
            a = LoopBody.function.inputs[swap[0]]
            b = LoopBody.function.inputs[swap[1]]
            LoopBody.function.inputs[swap[0]] = b
            LoopBody.function.inputs[swap[1]] = a
            LoopBody.signature[swap[0]] = pb.LoopBodyInputType.ACTIVATION
            activations+=1

        # Next Label Inputs according to how they are produced.
        # Indexable inputs will have different constant or none producers
        # Constant values broadcast to all nodes will have the same producer
        # Skip the (all) Activation inputs (have been swapped to beginning of the list)
        for index in range(activations, len(nodes[0].inputs)):
            inputs    = []
            for node in nodes:
                print(f"Node {node.name} {node.inputs[index]}")
                cinput = node.inputs[index]
                inputs.append(cinput)

            if pb.same(inputs) or same_values(inputs):
                # Constant with Respect to Loop
                LoopBody.signature[index] = pb.LoopBodyInputType.CONSTANT
            else:
                # Must be Indexed
                LoopBody.signature[index] = pb.LoopBodyInputType.PARAMETER

        ###################################################
        ## End I/O Normalization for Loop Body
        ###################################################

        LoopMatchPattern,nodes = LoopBody.build_function_match_pattern(model_ir.graph, use_iteration_ext=False)

        loop_replace_pattern = build_loop_replace_pattern(model_ir.graph, LoopBody)

        change_function_calls_to_loop = pattern.RewriteRule(
            LoopMatchPattern,
            loop_replace_pattern
        )
        rewrite_set = pattern.RewriteRuleSet([change_function_calls_to_loop])
        count = rewrite_set.apply_to_model(model_ir, verbose=None)
        print(f"Rolled {count} function calls into a loop operator")

        model = onnxscript.ir.serde.serialize_model(model_ir)

        model_wrapper = qonnx.core.modelwrapper.ModelWrapper(model)

        model = model_wrapper.transform(FoldConstants())

        model.save('simple_module_layers_replaced_loop.onnx')

        return (model, False)
