#!/usr/bin/env python3
"""
Decompose LayerNormalization nodes in ONNX models for TRT 8.5.x compatibility.

TensorRT 8.5.2 (JetPack 5.x) does not natively support the LayerNormalization
op (added in TRT 8.6+). This script replaces each LayerNormalization node with
its equivalent basic operations: ReduceMean, Sub, Pow, Add, Sqrt, Div, Mul, Add.

Usage:
  python3 onnx_layernorm_decompose.py input.onnx output.onnx
"""

import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def decompose_layernorm(model: onnx.ModelProto) -> onnx.ModelProto:
  """Replace all LayerNormalization nodes with basic ops."""
  graph = model.graph
  new_nodes = []
  new_initializers = []
  counter = 0

  for node in graph.node:
    if node.op_type != 'LayerNormalization':
      new_nodes.append(node)
      continue

    counter += 1
    prefix = f"layernorm_decomp_{counter}"

    # Parse attributes
    epsilon = 1e-5
    axis = -1
    for attr in node.attribute:
      if attr.name == 'epsilon':
        epsilon = attr.f
      elif attr.name == 'axis':
        axis = attr.i

    input_name = node.input[0]
    scale_name = node.input[1]
    bias_name = node.input[2] if len(node.input) > 2 else None
    output_name = node.output[0]

    # Create constant for epsilon
    eps_name = f"{prefix}_epsilon"
    eps_tensor = numpy_helper.from_array(np.array([epsilon], dtype=np.float32), name=eps_name)
    new_initializers.append(eps_tensor)

    # Create constant for exponent (2.0)
    two_name = f"{prefix}_two"
    two_tensor = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name=two_name)
    new_initializers.append(two_tensor)

    # mean = ReduceMean(input, axis=axis, keepdims=1)
    mean_name = f"{prefix}_mean"
    mean_node = helper.make_node('ReduceMean', inputs=[input_name], outputs=[mean_name],
                                  axes=[axis], keepdims=1, name=f"{prefix}_reducemean")
    new_nodes.append(mean_node)

    # diff = Sub(input, mean)
    diff_name = f"{prefix}_diff"
    diff_node = helper.make_node('Sub', inputs=[input_name, mean_name], outputs=[diff_name],
                                  name=f"{prefix}_sub")
    new_nodes.append(diff_node)

    # diff_sq = Pow(diff, 2)
    diff_sq_name = f"{prefix}_diff_sq"
    diff_sq_node = helper.make_node('Pow', inputs=[diff_name, two_name], outputs=[diff_sq_name],
                                     name=f"{prefix}_pow")
    new_nodes.append(diff_sq_node)

    # var = ReduceMean(diff_sq, axis=axis, keepdims=1)
    var_name = f"{prefix}_var"
    var_node = helper.make_node('ReduceMean', inputs=[diff_sq_name], outputs=[var_name],
                                 axes=[axis], keepdims=1, name=f"{prefix}_var_reducemean")
    new_nodes.append(var_node)

    # var_eps = Add(var, epsilon)
    var_eps_name = f"{prefix}_var_eps"
    var_eps_node = helper.make_node('Add', inputs=[var_name, eps_name], outputs=[var_eps_name],
                                     name=f"{prefix}_add_eps")
    new_nodes.append(var_eps_node)

    # std = Sqrt(var_eps)
    std_name = f"{prefix}_std"
    std_node = helper.make_node('Sqrt', inputs=[var_eps_name], outputs=[std_name],
                                 name=f"{prefix}_sqrt")
    new_nodes.append(std_node)

    # normalized = Div(diff, std)
    norm_name = f"{prefix}_normalized"
    norm_node = helper.make_node('Div', inputs=[diff_name, std_name], outputs=[norm_name],
                                  name=f"{prefix}_div")
    new_nodes.append(norm_node)

    # scaled = Mul(normalized, scale)
    scaled_name = f"{prefix}_scaled"
    scaled_node = helper.make_node('Mul', inputs=[norm_name, scale_name], outputs=[scaled_name],
                                    name=f"{prefix}_mul_scale")
    new_nodes.append(scaled_node)

    # output = Add(scaled, bias) or just scaled if no bias
    if bias_name:
      out_node = helper.make_node('Add', inputs=[scaled_name, bias_name], outputs=[output_name],
                                   name=f"{prefix}_add_bias")
      new_nodes.append(out_node)
    else:
      # Rename scaled output to match expected output
      scaled_node.output[0] = output_name

  if counter == 0:
    return model

  # Rebuild graph
  new_graph = helper.make_graph(
    new_nodes,
    graph.name,
    graph.input,
    graph.output,
    initializer=list(graph.initializer) + new_initializers,
  )
  # Preserve value_info
  new_graph.value_info.extend(graph.value_info)

  new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
  new_model.ir_version = model.ir_version
  new_model.producer_name = model.producer_name
  new_model.producer_version = model.producer_version
  new_model.domain = model.domain
  new_model.model_version = model.model_version
  new_model.doc_string = model.doc_string

  print(f"Decomposed {counter} LayerNormalization nodes into basic ops")
  return new_model


if __name__ == '__main__':
  if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} input.onnx output.onnx")
    sys.exit(1)

  input_path = sys.argv[1]
  output_path = sys.argv[2]

  model = onnx.load(input_path)
  model = decompose_layernorm(model)

  # Validate
  try:
    onnx.checker.check_model(model)
    print("Model validation passed")
  except Exception as e:
    print(f"Warning: model validation: {e}")

  onnx.save(model, output_path)
  print(f"Saved to {output_path}")
