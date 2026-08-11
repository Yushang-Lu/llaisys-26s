"""Python wrapper for the native Qwen2 inference implementation.

This module deliberately does not use a Python tensor framework. The model math, including greedy decoding, lives in ``libllaisys``; Python only parses metadata, copies safetensors BF16 byte ranges into native tensors, and drives autoregressive decoding.
"""

from __future__ import annotations

import ctypes
import json
import math
import mmap
import operator
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..libllaisys import DataType, DeviceType, LIB_LLAISYS
from ..libllaisys.qwen2 import LlaisysQwen2Meta


_SAFETENSORS_HEADER_PREFIX_BYTES = 8
_BF16_BYTES = 2


@dataclass(frozen=True)
class _TensorRecord:
    """One tensor's validated byte range in a safetensors file."""

    path: Path
    offset: int
    nbytes: int
    dtype: str
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class _WeightSpec:
    """The safetensors name, native weights field, and expected shape."""

    name: str
    field: str
    shape: Tuple[int, ...]
    layer: Optional[int] = None


def _json_object_without_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    """Reject malformed headers whose duplicate JSON keys would be ambiguous."""

    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r} in safetensors header")
        result[key] = value
    return result


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"config field {name!r} must be a positive integer")
    return int(value)


def _as_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"config field {name!r} must be a positive number")
    return float(value)


class Qwen2:
    """A native CPU, NVIDIA, or MetaX Qwen2 causal language model.

    ``generate`` always uses greedy (argmax) decoding. ``top_k``, ``top_p``, and ``temperature`` remain in the signature for compatibility with the assignment test harness, but are intentionally not used by this model.
    """

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        try:
            self._device = DeviceType(device)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported device {device!r}") from exc
        self._lock = threading.RLock()
        self._model = None
        self._weights = None
        self._closed = False
        self._device_ids = (ctypes.c_int * 1)(0)

        self._model_path = Path(model_path).expanduser()
        if not self._model_path.is_dir():
            raise FileNotFoundError(f"model directory does not exist: {self._model_path}")

        config = self._read_config(self._model_path)
        self._meta = self._make_meta(config)
        self._vocab_size = int(config["vocab_size"])
        self._max_sequence_length = int(config["max_position_embeddings"])
        self._eos_token_id = int(self._meta.end_token)
        specs = self._weight_specs(config)

        try:
            records = self._scan_safetensors(self._model_path)
            self._validate_weight_records(records, specs)
            self._create_model()
            self._load_weights(records, specs)
        except BaseException:
            # The native model owns every tensor handle written into its weight
            # table, including the subset loaded before a failure.
            self.close()
            raise

    @staticmethod
    def _read_config(model_path: Path) -> Mapping[str, Any]:
        config_path = model_path / "config.json"
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"missing model configuration: {config_path}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {config_path}: {exc}") from exc

        if not isinstance(config, dict):
            raise ValueError(f"{config_path} must contain a JSON object")
        return config

    @staticmethod
    def _make_meta(config: Mapping[str, Any]) -> LlaisysQwen2Meta:
        required = (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "max_position_embeddings",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
            "eos_token_id",
        )
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(f"model config is missing required fields: {', '.join(missing)}")

        nlayer = _as_positive_int(config["num_hidden_layers"], "num_hidden_layers")
        hidden_size = _as_positive_int(config["hidden_size"], "hidden_size")
        nhead = _as_positive_int(config["num_attention_heads"], "num_attention_heads")
        nkvhead = _as_positive_int(config["num_key_value_heads"], "num_key_value_heads")
        intermediate_size = _as_positive_int(config["intermediate_size"], "intermediate_size")
        max_sequence_length = _as_positive_int(
            config["max_position_embeddings"], "max_position_embeddings"
        )
        vocab_size = _as_positive_int(config["vocab_size"], "vocab_size")
        epsilon = _as_positive_float(config["rms_norm_eps"], "rms_norm_eps")
        theta = _as_positive_float(config["rope_theta"], "rope_theta")

        if hidden_size % nhead:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if nhead % nkvhead:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        eos_token_id = config["eos_token_id"]
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
            raise ValueError("Qwen2 requires config.eos_token_id to be an integer")
        if not 0 <= eos_token_id < vocab_size:
            raise ValueError("config.eos_token_id is outside the vocabulary")

        torch_dtype = config.get("torch_dtype")
        if torch_dtype is not None and str(torch_dtype).lower() not in {"bf16", "bfloat16"}:
            raise ValueError(
                f"unsupported model dtype {torch_dtype!r}; native Qwen2 supports BF16 only"
            )

        return LlaisysQwen2Meta(
            int(DataType.BF16),
            nlayer,
            hidden_size,
            nhead,
            nkvhead,
            hidden_size // nhead,
            intermediate_size,
            max_sequence_length,
            vocab_size,
            epsilon,
            theta,
            eos_token_id,
        )

    @staticmethod
    def _weight_specs(config: Mapping[str, Any]) -> Tuple[_WeightSpec, ...]:
        """Build the complete, exact DeepSeek-R1-Distill-Qwen weight map."""

        nlayer = int(config["num_hidden_layers"])
        hidden_size = int(config["hidden_size"])
        nhead = int(config["num_attention_heads"])
        nkvhead = int(config["num_key_value_heads"])
        intermediate_size = int(config["intermediate_size"])
        vocab_size = int(config["vocab_size"])
        head_dim = hidden_size // nhead
        kv_hidden_size = nkvhead * head_dim

        specs = [
            _WeightSpec("model.embed_tokens.weight", "in_embed", (vocab_size, hidden_size)),
            _WeightSpec("lm_head.weight", "out_embed", (vocab_size, hidden_size)),
            _WeightSpec("model.norm.weight", "out_norm_w", (hidden_size,)),
        ]

        for layer in range(nlayer):
            prefix = f"model.layers.{layer}"
            specs.extend(
                (
                    _WeightSpec(
                        f"{prefix}.input_layernorm.weight",
                        "attn_norm_w",
                        (hidden_size,),
                        layer,
                    ),
                    _WeightSpec(
                        f"{prefix}.self_attn.q_proj.weight",
                        "attn_q_w",
                        (hidden_size, hidden_size),
                        layer,
                    ),
                    _WeightSpec(
                        f"{prefix}.self_attn.q_proj.bias",
                        "attn_q_b",
                        (hidden_size,),
                        layer,
                    ),
                    _WeightSpec(
                        f"{prefix}.self_attn.k_proj.weight",
                        "attn_k_w",
                        (kv_hidden_size, hidden_size),
                        layer,
                    ),
                    _WeightSpec(
                        f"{prefix}.self_attn.k_proj.bias",
                        "attn_k_b",
                        (kv_hidden_size,),
                        layer,
                    ),
                    _WeightSpec(
                        f"{prefix}.self_attn.v_proj.weight",
                        "attn_v_w",
                        (kv_hidden_size, hidden_size),
                        layer,
                    ),
                    _WeightSpec(
                        f"{prefix}.self_attn.v_proj.bias",
                        "attn_v_b",
                        (kv_hidden_size,),
                        layer,
                    ),
                    _WeightSpec(
                        f"{prefix}.self_attn.o_proj.weight",
                        "attn_o_w",
                        (hidden_size, hidden_size),
                        layer,
                    ),
                    _WeightSpec(
                        f"{prefix}.post_attention_layernorm.weight",
                        "mlp_norm_w",
                        (hidden_size,),
                        layer,
                    ),
                    _WeightSpec(
                        f"{prefix}.mlp.gate_proj.weight",
                        "mlp_gate_w",
                        (intermediate_size, hidden_size),
                        layer,
                    ),
                    _WeightSpec(
                        f"{prefix}.mlp.up_proj.weight",
                        "mlp_up_w",
                        (intermediate_size, hidden_size),
                        layer,
                    ),
                    _WeightSpec(
                        f"{prefix}.mlp.down_proj.weight",
                        "mlp_down_w",
                        (hidden_size, intermediate_size),
                        layer,
                    ),
                )
            )

        expected_count = 3 + nlayer * 12
        if len(specs) != expected_count:
            raise AssertionError("internal Qwen2 weight-map construction error")
        return tuple(specs)

    @staticmethod
    def _scan_safetensors(model_path: Path) -> Dict[str, _TensorRecord]:
        """Read only safetensors headers and return their validated records."""

        files = sorted(path for path in model_path.glob("*.safetensors") if path.is_file())
        if not files:
            raise FileNotFoundError(f"no .safetensors files found in {model_path}")

        records: Dict[str, _TensorRecord] = {}
        for path in files:
            try:
                file_size = path.stat().st_size
                with path.open("rb") as tensor_file:
                    prefix = tensor_file.read(_SAFETENSORS_HEADER_PREFIX_BYTES)
                    if len(prefix) != _SAFETENSORS_HEADER_PREFIX_BYTES:
                        raise ValueError("file is smaller than the safetensors header prefix")
                    header_size = struct.unpack("<Q", prefix)[0]
                    if header_size > file_size - _SAFETENSORS_HEADER_PREFIX_BYTES:
                        raise ValueError("safetensors header extends beyond end of file")
                    header_bytes = tensor_file.read(header_size)
            except OSError as exc:
                raise OSError(f"cannot read safetensors file {path}: {exc}") from exc

            if len(header_bytes) != header_size:
                raise ValueError(f"truncated safetensors header in {path}")
            try:
                header = json.loads(
                    header_bytes.decode("utf-8"),
                    object_pairs_hook=_json_object_without_duplicate_keys,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid safetensors header in {path}: {exc}") from exc
            if not isinstance(header, dict):
                raise ValueError(f"safetensors header in {path} must be a JSON object")

            data_start = _SAFETENSORS_HEADER_PREFIX_BYTES + header_size
            data_size = file_size - data_start
            for name, descriptor in header.items():
                if name == "__metadata__":
                    if not isinstance(descriptor, dict):
                        raise ValueError(f"invalid __metadata__ value in {path}")
                    continue
                if name in records:
                    raise ValueError(f"tensor {name!r} occurs in more than one safetensors file")
                if not isinstance(descriptor, dict):
                    raise ValueError(f"invalid descriptor for tensor {name!r} in {path}")

                dtype = descriptor.get("dtype")
                shape = descriptor.get("shape")
                offsets = descriptor.get("data_offsets")
                if not isinstance(dtype, str):
                    raise ValueError(f"tensor {name!r} in {path} has no valid dtype")
                if not isinstance(shape, list) or not all(
                    isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
                    for dim in shape
                ):
                    raise ValueError(f"tensor {name!r} in {path} has an invalid shape")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
                ):
                    raise ValueError(f"tensor {name!r} in {path} has invalid data_offsets")

                begin, end = offsets
                if begin < 0 or end < begin or end > data_size:
                    raise ValueError(f"tensor {name!r} in {path} has out-of-range data_offsets")
                records[name] = _TensorRecord(
                    path=path,
                    offset=data_start + begin,
                    nbytes=end - begin,
                    dtype=dtype,
                    shape=tuple(shape),
                )
        return records

    @staticmethod
    def _validate_weight_records(
        records: Mapping[str, _TensorRecord], specs: Sequence[_WeightSpec]
    ) -> None:
        expected = {spec.name: spec for spec in specs}
        actual_names = set(records)
        expected_names = set(expected)
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {len(missing)} tensor(s): {', '.join(missing[:3])}")
            if extra:
                details.append(f"unexpected {len(extra)} tensor(s): {', '.join(extra[:3])}")
            raise ValueError("invalid Qwen2 safetensors weight set (" + "; ".join(details) + ")")
        if len(records) != len(specs):
            raise ValueError(
                f"invalid Qwen2 tensor count: expected {len(specs)}, found {len(records)}"
            )

        for spec in specs:
            record = records[spec.name]
            if record.dtype != "BF16":
                raise ValueError(
                    f"tensor {spec.name!r} has dtype {record.dtype!r}; expected 'BF16'"
                )
            if record.shape != spec.shape:
                raise ValueError(
                    f"tensor {spec.name!r} has shape {record.shape}; expected {spec.shape}"
                )
            expected_bytes = math.prod(spec.shape) * _BF16_BYTES
            if record.nbytes != expected_bytes:
                raise ValueError(
                    f"tensor {spec.name!r} has {record.nbytes} bytes; expected {expected_bytes}"
                )

    def _create_model(self) -> None:
        model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta),
            ctypes.c_int(int(self._device)),
            self._device_ids,
            ctypes.c_int(1),
        )
        if not model:
            self._raise_native_error("llaisysQwen2ModelCreate")
        self._model = model

        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(model)
        if not weights:
            self._raise_native_error("llaisysQwen2ModelWeights")
        self._weights = weights

    @staticmethod
    def _raise_native_error(operation: str) -> None:
        error_ptr = LIB_LLAISYS.llaisysQwen2GetLastError()
        if error_ptr:
            message = ctypes.cast(error_ptr, ctypes.c_char_p).value
            if message:
                raise RuntimeError(
                    f"{operation} failed: {message.decode('utf-8', errors='replace')}"
                )
        raise RuntimeError(f"{operation} failed")

    def _weight_handle(self, spec: _WeightSpec):
        """Return the model-owned tensor handle allocated by ModelCreate."""

        weights = self._weights.contents
        tensor = (
            getattr(weights, spec.field)
            if spec.layer is None
            else getattr(weights, spec.field)[spec.layer]
        )
        if not tensor:
            raise RuntimeError(f"native model did not allocate weight {spec.name!r}")
        return tensor

    def _load_weights(
        self, records: Mapping[str, _TensorRecord], specs: Sequence[_WeightSpec]
    ) -> None:
        """Load each raw BF16 safetensors span directly into its native tensor."""

        records_by_path: Dict[Path, list[Tuple[_WeightSpec, _TensorRecord]]] = {}
        for spec in specs:
            record = records[spec.name]
            records_by_path.setdefault(record.path, []).append((spec, record))

        for path, file_records in records_by_path.items():
            try:
                with path.open("rb") as tensor_file:
                    # ACCESS_COPY makes the mapping writable enough for
                    # ctypes.from_buffer without changing the source file.
                    with mmap.mmap(tensor_file.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
                        for spec, record in file_records:
                            # ModelCreate allocates and owns all weight tensors.
                            # Keep the returned handle intact; Python only loads
                            # safetensors bytes into it and never destroys it.
                            tensor = self._weight_handle(spec)
                            source = ctypes.c_char.from_buffer(mapping, record.offset)
                            try:
                                source_ptr = ctypes.c_void_p(ctypes.addressof(source))
                                LIB_LLAISYS.tensorLoad(tensor, source_ptr)
                            finally:
                                # An exported ctypes buffer prevents mmap.close().
                                del source
            except OSError as exc:
                raise OSError(f"cannot memory-map safetensors file {path}: {exc}") from exc

    @staticmethod
    def _normalize_tokens(tokens: Sequence[int], vocab_size: int) -> list[int]:
        result = []
        for index, token in enumerate(tokens):
            if isinstance(token, bool):
                raise TypeError(f"token {index} must be an integer, not bool")
            try:
                token_id = operator.index(token)
            except TypeError as exc:
                raise TypeError(f"token {index} must be an integer") from exc
            if not 0 <= token_id < vocab_size:
                raise ValueError(f"token {index} ({token_id}) is outside the vocabulary")
            result.append(int(token_id))
        return result

    @staticmethod
    def _normalize_max_new_tokens(max_new_tokens: Optional[int]) -> int:
        # Keep a bounded default rather than accidentally asking the native
        # model to fill its complete 131072-token context window.
        if max_new_tokens is None:
            return 128
        if isinstance(max_new_tokens, bool):
            raise TypeError("max_new_tokens must be an integer")
        try:
            value = operator.index(max_new_tokens)
        except TypeError as exc:
            raise TypeError("max_new_tokens must be an integer") from exc
        if value < 0:
            raise ValueError("max_new_tokens must not be negative")
        return int(value)

    def _require_open(self):
        if self._closed or not self._model:
            raise RuntimeError("Qwen2 model has been closed")
        return self._model

    def reset(self) -> None:
        """Clear the native per-layer KV cache."""

        with self._lock:
            LIB_LLAISYS.llaisysQwen2ModelReset(self._require_open())
            if LIB_LLAISYS.llaisysQwen2GetLastError():
                self._raise_native_error("llaisysQwen2ModelReset")

    def _infer(self, token_ids: Sequence[int]) -> int:
        if not token_ids:
            raise ValueError("native Qwen2 inference requires at least one token")
        native_tokens = (ctypes.c_int64 * len(token_ids))(*token_ids)
        next_token = int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._require_open(), native_tokens, ctypes.c_size_t(len(token_ids))
            )
        )
        if next_token < 0:
            self._raise_native_error("llaisysQwen2ModelInfer")
        return next_token

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> list[int]:
        """Return ``inputs`` followed by up to ``max_new_tokens`` argmax tokens."""

        # These parameters are accepted to match the test's Hugging Face call
        # shape. Assignment 3 deliberately exercises greedy decoding only.
        del top_k, top_p, temperature

        input_tokens = self._normalize_tokens(inputs, self._vocab_size)
        new_token_count = self._normalize_max_new_tokens(max_new_tokens)
        if not input_tokens and new_token_count:
            raise ValueError("cannot generate from an empty prompt")
        if len(input_tokens) > self._max_sequence_length:
            raise ValueError("prompt exceeds max_position_embeddings")
        if new_token_count > self._max_sequence_length - len(input_tokens):
            raise ValueError("prompt plus requested output exceeds max_position_embeddings")

        outputs = list(input_tokens)
        with self._lock:
            # A Qwen2 instance may be reused for independent prompts. The
            # native cache persists across Infer calls, not across generate.
            self.reset()
            if new_token_count == 0:
                return outputs

            next_token = self._infer(input_tokens)
            outputs.append(next_token)
            if next_token == self._eos_token_id:
                return outputs

            for _ in range(1, new_token_count):
                next_token = self._infer((next_token,))
                outputs.append(next_token)
                if next_token == self._eos_token_id:
                    break
        return outputs

    def close(self) -> None:
        """Release the native model, including its weight tensors and KV cache."""

        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            model = self._model
            self._model = None
            self._weights = None
            self._closed = True
            if model:
                LIB_LLAISYS.llaisysQwen2ModelDestroy(model)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        # Destructors run during interpreter teardown as well; never mask the
        # original exception just because the shared library is already gone.
        try:
            self.close()
        except Exception:
            pass
