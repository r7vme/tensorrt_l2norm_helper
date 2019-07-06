#!/usr/bin/env python3
import graphsurgeon as gs
import tensorflow as tf
import uff

if __name__ == "__main__":
  # USER DEFINED VALUES
  output_nodes = ["dimension/LeakyRelu","orientation/l2_normalize","confidence/Softmax"]
  input_node   = "input_1"
  pb_file      = "sample.pb"
  uff_file     = "sample.uff"
  # END USER DEFINED VALUES

  # read tensorflow graph
  # NOTE: Make sure to freeze and optimize (remove training nodes, etc.)
  dynamic_graph = gs.DynamicGraph(pb_file)
  nodes=[n.name for n in dynamic_graph.as_graph_def().node]
  ns={}
  for node in nodes:
    # replace LeakyRelu with default TRT plugin LReLU_TRT
    if "LeakyRelu" in node:
      ns[node]=gs.create_plugin_node(name=node,op="LReLU_TRT", negSlope=0.1)
    # replace Maximum with L2Norm_Helper_TRT max operation (CUDA's fmaxf)
    if node == "orientation/l2_normalize/Maximum":
      ns[node]=gs.create_plugin_node(name=node,op="L2Norm_Helper_TRT",op_type=0,eps=1e-12)
    # replace Rsqrt with L2Norm_Helper_TRT max operation (CUDA's rsqrtf)
    if node == "orientation/l2_normalize/Rsqrt":
      ns[node]=gs.create_plugin_node(name=node,op="L2Norm_Helper_TRT",op_type=1)
  dynamic_graph.collapse_namespaces(ns)
  # write UFF to file
  uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), output_nodes=output_nodes,
                                  output_filename=uff_file, text=False)
