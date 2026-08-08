import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from llaisys.models.qwen2 import Qwen2
from llaisys.models.qwen2 import _TensorRecord
from llaisys.models.qwen2 import _WeightSpec


class _FakeNativeLibrary:
    def __init__(self):
        self.reset_calls = []
        self.last_error = None

    def llaisysQwen2GetLastError(self):
        return self.last_error

    def llaisysQwen2ModelReset(self, model):
        self.reset_calls.append(model)


def _make_wrapper(responses, *, eos_token=99, max_sequence_length=32):
    model = object.__new__(Qwen2)
    model._lock = threading.RLock()
    model._model = 1
    model._weights = None
    model._closed = False
    model._vocab_size = 128
    model._max_sequence_length = max_sequence_length
    model._eos_token_id = eos_token

    remaining = iter(responses)
    infer_calls = []

    def fake_infer(token_ids):
        infer_calls.append(tuple(token_ids))
        return next(remaining)

    def fake_close():
        model._model = None
        model._closed = True

    model._infer = fake_infer
    model.close = fake_close
    return model, infer_calls


class Qwen2GenerateTests(unittest.TestCase):
    def test_repeated_generate_resets_cache_and_uses_incremental_tokens(self):
        native = _FakeNativeLibrary()
        model, infer_calls = _make_wrapper([10, 11, 12, 13])

        with patch("llaisys.models.qwen2.LIB_LLAISYS", native):
            self.assertEqual(model.generate([1, 2], max_new_tokens=2),
                             [1, 2, 10, 11])
            self.assertEqual(model.generate([3], max_new_tokens=2),
                             [3, 12, 13])

        self.assertEqual(native.reset_calls, [1, 1])
        self.assertEqual(infer_calls,
                         [(1, 2), (10,), (3,), (12,)])
        model.close()

    def test_eos_stops_generation_immediately(self):
        native = _FakeNativeLibrary()
        model, infer_calls = _make_wrapper([99, 7], eos_token=99)

        with patch("llaisys.models.qwen2.LIB_LLAISYS", native):
            self.assertEqual(model.generate([4], max_new_tokens=8),
                             [4, 99])

        self.assertEqual(native.reset_calls, [1])
        self.assertEqual(infer_calls, [(4,)])
        model.close()

    def test_context_limit_is_checked_before_native_inference(self):
        native = _FakeNativeLibrary()
        model, infer_calls = _make_wrapper(
            [5], max_sequence_length=4)

        with patch("llaisys.models.qwen2.LIB_LLAISYS", native):
            with self.assertRaisesRegex(
                ValueError,
                "prompt plus requested output exceeds",
            ):
                model.generate([1, 2, 3], max_new_tokens=2)

        self.assertEqual(native.reset_calls, [])
        self.assertEqual(infer_calls, [])
        model.close()


class Qwen2WeightValidationTests(unittest.TestCase):
    def setUp(self):
        self.specs = (
            _WeightSpec("weight", "field", (2, 2)),
        )
        self.valid_record = _TensorRecord(
            path=Path("model.safetensors"),
            offset=64,
            nbytes=8,
            dtype="BF16",
            shape=(2, 2),
        )

    def test_rejects_missing_or_unexpected_weights(self):
        with self.assertRaisesRegex(ValueError, "missing 1 tensor"):
            Qwen2._validate_weight_records({}, self.specs)

        records = {
            "weight": self.valid_record,
            "unexpected": self.valid_record,
        }
        with self.assertRaisesRegex(ValueError, "unexpected 1 tensor"):
            Qwen2._validate_weight_records(records, self.specs)

    def test_rejects_wrong_dtype_shape_and_byte_count(self):
        invalid_cases = (
            (
                self.valid_record.__class__(
                    self.valid_record.path,
                    self.valid_record.offset,
                    self.valid_record.nbytes,
                    "F32",
                    self.valid_record.shape,
                ),
                "expected 'BF16'",
            ),
            (
                self.valid_record.__class__(
                    self.valid_record.path,
                    self.valid_record.offset,
                    self.valid_record.nbytes,
                    self.valid_record.dtype,
                    (4, 1),
                ),
                r"expected \(2, 2\)",
            ),
            (
                self.valid_record.__class__(
                    self.valid_record.path,
                    self.valid_record.offset,
                    6,
                    self.valid_record.dtype,
                    self.valid_record.shape,
                ),
                "6 bytes; expected 8",
            ),
        )

        for record, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    Qwen2._validate_weight_records(
                        {"weight": record},
                        self.specs,
                    )


if __name__ == "__main__":
    unittest.main()
