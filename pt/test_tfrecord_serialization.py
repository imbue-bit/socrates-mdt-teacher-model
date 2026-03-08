"""Tests for TFRecord serialization pipeline.

Verifies that the protobuf-based TFRecord writer correctly serializes
feature dictionaries without TypeError, and that serialized data can
be deserialized back to the original values.
"""

import collections
import io
import struct
import tempfile
import os

import numpy as np

from tfrecord import example_pb2
from tfrecord.writer import TFRecordWriter


def test_serialize_tf_example_int_features():
    """Verify int features serialize and round-trip correctly."""
    datum = {
        "input_ids": ([101, 2003, 1010, 102], "int"),
        "attention_mask": ([1, 1, 1, 1], "int"),
        "token_type_ids": ([0, 0, 0, 0], "int"),
        "labels": ([-100, 2003, -100, -100], "int"),
        "next_sentence_label": ([0], "int"),
    }
    serialized = TFRecordWriter.serialize_tf_example(datum)
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0

    example = example_pb2.Example()
    example.ParseFromString(serialized)

    features = example.features.feature
    assert set(features.keys()) == set(datum.keys())
    assert list(features["input_ids"].int64_list.value) == [101, 2003, 1010, 102]
    assert list(features["attention_mask"].int64_list.value) == [1, 1, 1, 1]
    assert list(features["labels"].int64_list.value) == [-100, 2003, -100, -100]
    assert list(features["next_sentence_label"].int64_list.value) == [0]


def test_serialize_tf_example_float_features():
    """Verify float features serialize correctly."""
    datum = {
        "scores": ([0.1, 0.9, 0.5], "float"),
    }
    serialized = TFRecordWriter.serialize_tf_example(datum)
    example = example_pb2.Example()
    example.ParseFromString(serialized)
    values = list(example.features.feature["scores"].float_list.value)
    assert len(values) == 3
    assert abs(values[0] - 0.1) < 1e-6


def test_serialize_tf_example_byte_features():
    """Verify byte features serialize correctly."""
    datum = {
        "raw_data": ([b"hello world"], "byte"),
    }
    serialized = TFRecordWriter.serialize_tf_example(datum)
    example = example_pb2.Example()
    example.ParseFromString(serialized)
    values = list(example.features.feature["raw_data"].bytes_list.value)
    assert values == [b"hello world"]


def test_serialize_tf_example_scalar_value():
    """Verify scalar (non-list) values are auto-wrapped in a list."""
    datum = {
        "label": (42, "int"),
    }
    serialized = TFRecordWriter.serialize_tf_example(datum)
    example = example_pb2.Example()
    example.ParseFromString(serialized)
    assert list(example.features.feature["label"].int64_list.value) == [42]


def test_serialize_tf_example_ordered_dict():
    """Verify OrderedDict (as used in create_pretraining_data.py) works."""
    datum = collections.OrderedDict()
    datum["input_ids"] = ([101, 2003, 102], "int")
    datum["attention_mask"] = ([1, 1, 1], "int")
    datum["token_type_ids"] = ([0, 0, 0], "int")
    datum["labels"] = ([-100, -100, -100], "int")

    serialized = TFRecordWriter.serialize_tf_example(datum)
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0

    example = example_pb2.Example()
    example.ParseFromString(serialized)
    assert set(example.features.feature.keys()) == {"input_ids", "attention_mask", "token_type_ids", "labels"}


def test_writer_write_to_file():
    """Verify TFRecordWriter writes valid TFRecord files."""
    datum = {
        "input_ids": ([101, 2003, 1010, 102], "int"),
        "labels": ([-100, 2003, -100, -100], "int"),
    }
    with tempfile.NamedTemporaryFile(suffix=".tfrecord", delete=False) as f:
        tmp_path = f.name

    try:
        writer = TFRecordWriter(tmp_path)
        writer.write(datum)
        writer.write(datum)
        writer.close()

        assert os.path.getsize(tmp_path) > 0

        # Read back and verify the TFRecord format manually
        with open(tmp_path, "rb") as f:
            # Read first record
            length_bytes = f.read(8)
            assert len(length_bytes) == 8
            length = struct.unpack("<Q", length_bytes)[0]
            _crc = f.read(4)  # length CRC
            record_bytes = f.read(length)
            _crc = f.read(4)  # data CRC

            example = example_pb2.Example()
            example.ParseFromString(record_bytes)
            assert list(example.features.feature["input_ids"].int64_list.value) == [101, 2003, 1010, 102]
    finally:
        os.unlink(tmp_path)


def test_serialize_tf_example_with_numpy_array():
    """Verify numpy arrays work as feature values."""
    datum = {
        "embeddings": (np.array([1.0, 2.0, 3.0], dtype=np.float32), "float"),
        "ids": (np.array([1, 2, 3], dtype=np.int64), "int"),
    }
    serialized = TFRecordWriter.serialize_tf_example(datum)
    example = example_pb2.Example()
    example.ParseFromString(serialized)
    assert list(example.features.feature["ids"].int64_list.value) == [1, 2, 3]


def test_serialize_sequence_example():
    """Verify SequenceExample serialization works."""
    context_datum = {
        "length": (5, "int"),
    }
    features_datum = {
        "tokens": ([[1, 2], [3, 4], [5, 6]], "int"),
    }
    serialized = TFRecordWriter.serialize_tf_sequence_example(context_datum, features_datum)
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0

    seq_example = example_pb2.SequenceExample()
    seq_example.ParseFromString(serialized)
    assert list(seq_example.context.feature["length"].int64_list.value) == [5]


if __name__ == "__main__":
    test_serialize_tf_example_int_features()
    print("PASS: test_serialize_tf_example_int_features")

    test_serialize_tf_example_float_features()
    print("PASS: test_serialize_tf_example_float_features")

    test_serialize_tf_example_byte_features()
    print("PASS: test_serialize_tf_example_byte_features")

    test_serialize_tf_example_scalar_value()
    print("PASS: test_serialize_tf_example_scalar_value")

    test_serialize_tf_example_ordered_dict()
    print("PASS: test_serialize_tf_example_ordered_dict")

    test_writer_write_to_file()
    print("PASS: test_writer_write_to_file")

    test_serialize_tf_example_with_numpy_array()
    print("PASS: test_serialize_tf_example_with_numpy_array")

    test_serialize_sequence_example()
    print("PASS: test_serialize_sequence_example")

    print("\nAll tests passed!")
