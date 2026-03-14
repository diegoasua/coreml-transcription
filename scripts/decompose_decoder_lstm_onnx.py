#!/usr/bin/env python3
"""Rewrite single-step decoder LSTM nodes into explicit ONNX gate ops.

This targets the Parakeet TDT decoder graph, where each LSTM invocation runs
with sequence length 1 and explicit recurrent state tensors. Replacing the ONNX
`LSTM` op with matmul + elementwise gates gives Core ML a better chance to keep
the graph on ANE-friendly primitive ops.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


def _const_node(name: str, value: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_const",
        value=numpy_helper.from_array(value, name=f"{name}_value"),
    )


def _shape_dims(value_info: Any) -> list[Any]:
    dims: list[Any] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def _make_random_feed(model: onnx.ModelProto, decoder_frames: int) -> dict[str, np.ndarray]:
    dtype_map = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
    }
    feed: dict[str, np.ndarray] = {}
    for value_info in model.graph.input:
        dims: list[int] = []
        raw_dims = _shape_dims(value_info)
        for idx, dim in enumerate(raw_dims):
            if isinstance(dim, int) and dim > 0:
                dims.append(dim)
                continue
            if value_info.name == "encoder_outputs" and idx == 2:
                dims.append(decoder_frames)
            else:
                dims.append(1)
        dtype = dtype_map.get(value_info.type.tensor_type.elem_type, np.float32)
        if np.issubdtype(dtype, np.floating):
            values = np.random.uniform(-0.25, 0.25, size=dims).astype(dtype)
        else:
            values = np.zeros(dims, dtype=dtype)
        if value_info.name == "target_length":
            values = np.ones(dims, dtype=dtype)
        feed[value_info.name] = values
    return feed


def _replace_lstm_node(
    node: onnx.NodeProto,
    *,
    node_index: int,
    graph_inputs: set[str],
) -> list[onnx.NodeProto]:
    if len(node.input) < 7:
        raise ValueError(f"LSTM node '{node.name or node_index}' missing required inputs.")

    hidden_size = None
    for attr in node.attribute:
        if attr.name == "hidden_size":
            hidden_size = int(attr.i)
            break
    if hidden_size is None or hidden_size <= 0:
        raise ValueError(f"LSTM node '{node.name or node_index}' missing hidden_size.")

    if any(attr.name == "direction" and helper.get_attribute_value(attr) != b"forward" for attr in node.attribute):
        raise ValueError("Only forward single-direction LSTM nodes are supported.")

    x_name, w_name, r_name, b_name, seq_lens_name, h0_name, c0_name = list(node.input[:7])
    if seq_lens_name:
        raise ValueError("Decoder LSTM rewrite expects empty sequence_lens input.")
    if h0_name not in graph_inputs or c0_name not in graph_inputs:
        # In this decoder graph they are graph values produced by Slice nodes, not
        # top-level inputs. We still support them, but we require them to exist.
        pass

    prefix = f"{node.output[0]}__decomp_{node_index}"
    nodes: list[onnx.NodeProto] = []

    axis0 = f"{prefix}_axis0"
    axis1 = f"{prefix}_axis1"
    axes01 = f"{prefix}_axes01"
    split4 = f"{prefix}_split4"
    split2 = f"{prefix}_split2"
    for name, value in (
        (axis0, np.asarray([0], dtype=np.int64)),
        (axis1, np.asarray([1], dtype=np.int64)),
        (axes01, np.asarray([0, 1], dtype=np.int64)),
        (split4, np.asarray([hidden_size] * 4, dtype=np.int64)),
        (split2, np.asarray([hidden_size * 4, hidden_size * 4], dtype=np.int64)),
    ):
        nodes.append(_const_node(name, value))

    x_step = f"{prefix}_x_step"
    h_prev = f"{prefix}_h_prev"
    c_prev = f"{prefix}_c_prev"
    nodes.extend(
        [
            helper.make_node("Squeeze", inputs=[x_name, axis0], outputs=[x_step], name=f"{prefix}_squeeze_x"),
            helper.make_node("Squeeze", inputs=[h0_name, axis0], outputs=[h_prev], name=f"{prefix}_squeeze_h0"),
            helper.make_node("Squeeze", inputs=[c0_name, axis0], outputs=[c_prev], name=f"{prefix}_squeeze_c0"),
        ]
    )

    w_squeezed = f"{prefix}_w_squeezed"
    r_squeezed = f"{prefix}_r_squeezed"
    b_squeezed = f"{prefix}_b_squeezed"
    nodes.extend(
        [
            helper.make_node("Squeeze", inputs=[w_name, axis0], outputs=[w_squeezed], name=f"{prefix}_squeeze_w"),
            helper.make_node("Squeeze", inputs=[r_name, axis0], outputs=[r_squeezed], name=f"{prefix}_squeeze_r"),
            helper.make_node("Squeeze", inputs=[b_name, axis0], outputs=[b_squeezed], name=f"{prefix}_squeeze_b"),
        ]
    )

    w_t = f"{prefix}_w_t"
    r_t = f"{prefix}_r_t"
    nodes.extend(
        [
            helper.make_node("Transpose", inputs=[w_squeezed], outputs=[w_t], name=f"{prefix}_transpose_w", perm=[1, 0]),
            helper.make_node("Transpose", inputs=[r_squeezed], outputs=[r_t], name=f"{prefix}_transpose_r", perm=[1, 0]),
        ]
    )

    xb = f"{prefix}_xb"
    hb = f"{prefix}_hb"
    nodes.extend(
        [
            helper.make_node("MatMul", inputs=[x_step, w_t], outputs=[xb], name=f"{prefix}_x_mm"),
            helper.make_node("MatMul", inputs=[h_prev, r_t], outputs=[hb], name=f"{prefix}_h_mm"),
        ]
    )

    wb, rb = [f"{prefix}_{name}" for name in ("wb", "rb")]
    nodes.append(helper.make_node("Split", inputs=[b_squeezed, split2], outputs=[wb, rb], name=f"{prefix}_split_bias", axis=0))

    bias = f"{prefix}_bias"
    gates_pre = f"{prefix}_gates_pre"
    gates = f"{prefix}_gates"
    nodes.extend(
        [
            helper.make_node("Add", inputs=[wb, rb], outputs=[bias], name=f"{prefix}_bias_add"),
            helper.make_node("Add", inputs=[xb, hb], outputs=[gates_pre], name=f"{prefix}_xh_add"),
            helper.make_node("Add", inputs=[gates_pre, bias], outputs=[gates], name=f"{prefix}_gates_add"),
        ]
    )

    gate_i, gate_o, gate_f, gate_c = [f"{prefix}_{name}" for name in ("i", "o", "f", "c")]
    nodes.append(
        helper.make_node(
            "Split",
            inputs=[gates, split4],
            outputs=[gate_i, gate_o, gate_f, gate_c],
            name=f"{prefix}_split_gates",
            axis=1,
        )
    )

    i_sig = f"{prefix}_i_sig"
    o_sig = f"{prefix}_o_sig"
    f_sig = f"{prefix}_f_sig"
    c_tanh = f"{prefix}_c_tanh"
    nodes.extend(
        [
            helper.make_node("Sigmoid", inputs=[gate_i], outputs=[i_sig], name=f"{prefix}_sigmoid_i"),
            helper.make_node("Sigmoid", inputs=[gate_o], outputs=[o_sig], name=f"{prefix}_sigmoid_o"),
            helper.make_node("Sigmoid", inputs=[gate_f], outputs=[f_sig], name=f"{prefix}_sigmoid_f"),
            helper.make_node("Tanh", inputs=[gate_c], outputs=[c_tanh], name=f"{prefix}_tanh_c"),
        ]
    )

    forget_term = f"{prefix}_forget_term"
    input_term = f"{prefix}_input_term"
    c_next_2d = f"{prefix}_c_next_2d"
    c_next_tanh = f"{prefix}_c_next_tanh"
    h_next_2d = f"{prefix}_h_next_2d"
    nodes.extend(
        [
            helper.make_node("Mul", inputs=[f_sig, c_prev], outputs=[forget_term], name=f"{prefix}_forget_mul"),
            helper.make_node("Mul", inputs=[i_sig, c_tanh], outputs=[input_term], name=f"{prefix}_input_mul"),
            helper.make_node("Add", inputs=[forget_term, input_term], outputs=[c_next_2d], name=f"{prefix}_c_add"),
            helper.make_node("Tanh", inputs=[c_next_2d], outputs=[c_next_tanh], name=f"{prefix}_c_next_tanh"),
            helper.make_node("Mul", inputs=[o_sig, c_next_tanh], outputs=[h_next_2d], name=f"{prefix}_h_mul"),
        ]
    )

    y_h = node.output[1]
    y_c = node.output[2]
    y = node.output[0]
    h_exp = f"{prefix}_h_exp"
    nodes.extend(
        [
            helper.make_node("Unsqueeze", inputs=[h_next_2d, axis0], outputs=[y_h], name=f"{prefix}_unsqueeze_h"),
            helper.make_node("Unsqueeze", inputs=[c_next_2d, axis0], outputs=[y_c], name=f"{prefix}_unsqueeze_c"),
            helper.make_node("Unsqueeze", inputs=[y_h, axis1], outputs=[y], name=f"{prefix}_unsqueeze_y"),
        ]
    )
    return nodes


def decompose_model(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    graph_inputs = {item.name for item in graph.input}
    graph_inputs.update(item.name for item in graph.value_info)
    graph_inputs.update(item.name for item in graph.output)

    replaced = 0
    new_nodes: list[onnx.NodeProto] = []
    for idx, node in enumerate(graph.node):
        if node.op_type == "LSTM":
            new_nodes.extend(_replace_lstm_node(node, node_index=idx, graph_inputs=graph_inputs))
            replaced += 1
        else:
            new_nodes.append(node)

    if replaced == 0:
        raise ValueError("No LSTM nodes found.")

    del graph.node[:]
    graph.node.extend(new_nodes)
    model = shape_inference.infer_shapes(model)
    checker.check_model(model)
    return model


def verify_models(
    original_path: Path,
    rewritten_path: Path,
    *,
    decoder_frames: int,
    runs: int,
    atol: float,
    rtol: float,
) -> None:
    original = onnx.load(str(original_path))
    feed = _make_random_feed(original, decoder_frames=decoder_frames)
    so = ort.SessionOptions()
    ref = ort.InferenceSession(str(original_path), sess_options=so, providers=["CPUExecutionProvider"])
    test = ort.InferenceSession(str(rewritten_path), sess_options=so, providers=["CPUExecutionProvider"])

    output_names = [out.name for out in original.graph.output]
    worst_abs = 0.0
    worst_rel = 0.0
    for _ in range(runs):
        outputs_ref = ref.run(None, feed)
        outputs_test = test.run(None, feed)
        for name, a, b in zip(output_names, outputs_ref, outputs_test):
            arr_a = np.asarray(a)
            arr_b = np.asarray(b)
            if arr_a.dtype.kind in {"i", "u"}:
                if not np.array_equal(arr_a, arr_b):
                    raise AssertionError(f"Mismatch in integer output '{name}'.")
                continue
            abs_err = float(np.max(np.abs(arr_a - arr_b)))
            denom = np.maximum(np.abs(arr_a), 1e-8)
            rel_err = float(np.max(np.abs(arr_a - arr_b) / denom))
            worst_abs = max(worst_abs, abs_err)
            worst_rel = max(worst_rel, rel_err)
            if not np.allclose(arr_a, arr_b, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"Verification failed for output '{name}': max_abs={abs_err:.6e} max_rel={rel_err:.6e}"
                )
    print(
        f"Verification passed: runs={runs} max_abs={worst_abs:.6e} max_rel={worst_rel:.6e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose decoder ONNX LSTM nodes into primitive ops.")
    parser.add_argument("--input-onnx", type=Path, required=True)
    parser.add_argument("--output-onnx", type=Path, required=True)
    parser.add_argument("--decoder-frames", type=int, default=300)
    parser.add_argument("--verify-runs", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(str(args.input_onnx))
    rewritten = decompose_model(model)
    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewritten, str(args.output_onnx))
    print(f"Rewritten ONNX saved: {args.output_onnx}")
    verify_models(
        original_path=args.input_onnx,
        rewritten_path=args.output_onnx,
        decoder_frames=max(1, args.decoder_frames),
        runs=max(1, args.verify_runs),
        atol=args.atol,
        rtol=args.rtol,
    )


if __name__ == "__main__":
    main()
