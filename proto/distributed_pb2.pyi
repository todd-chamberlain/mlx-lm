from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CompressionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NONE: _ClassVar[CompressionType]
    ZSTD: _ClassVar[CompressionType]
    LZ4: _ClassVar[CompressionType]

class PartitionStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BALANCED: _ClassVar[PartitionStrategy]
    MEMORY_AWARE: _ClassVar[PartitionStrategy]
    CUSTOM: _ClassVar[PartitionStrategy]

class PipelineStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INITIALIZING: _ClassVar[PipelineStatus]
    READY: _ClassVar[PipelineStatus]
    RUNNING: _ClassVar[PipelineStatus]
    DEGRADED: _ClassVar[PipelineStatus]
    FAILED: _ClassVar[PipelineStatus]
NONE: CompressionType
ZSTD: CompressionType
LZ4: CompressionType
BALANCED: PartitionStrategy
MEMORY_AWARE: PartitionStrategy
CUSTOM: PartitionStrategy
INITIALIZING: PipelineStatus
READY: PipelineStatus
RUNNING: PipelineStatus
DEGRADED: PipelineStatus
FAILED: PipelineStatus

class ActivationTransfer(_message.Message):
    __slots__ = ("request_id", "layer_index", "tensor_data", "metadata", "compression")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_INDEX_FIELD_NUMBER: _ClassVar[int]
    TENSOR_DATA_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    layer_index: int
    tensor_data: bytes
    metadata: TensorMetadata
    compression: CompressionType
    def __init__(self, request_id: _Optional[str] = ..., layer_index: _Optional[int] = ..., tensor_data: _Optional[bytes] = ..., metadata: _Optional[_Union[TensorMetadata, _Mapping]] = ..., compression: _Optional[_Union[CompressionType, str]] = ...) -> None: ...

class TensorMetadata(_message.Message):
    __slots__ = ("shape", "dtype", "size_bytes")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    size_bytes: int
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., size_bytes: _Optional[int] = ...) -> None: ...

class LayerAssignment(_message.Message):
    __slots__ = ("node_id", "start_layer", "end_layer", "next_node_id", "next_node_address")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    START_LAYER_FIELD_NUMBER: _ClassVar[int]
    END_LAYER_FIELD_NUMBER: _ClassVar[int]
    NEXT_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NEXT_NODE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    start_layer: int
    end_layer: int
    next_node_id: str
    next_node_address: str
    def __init__(self, node_id: _Optional[str] = ..., start_layer: _Optional[int] = ..., end_layer: _Optional[int] = ..., next_node_id: _Optional[str] = ..., next_node_address: _Optional[str] = ...) -> None: ...

class PipelineConfig(_message.Message):
    __slots__ = ("pipeline_id", "model_name", "total_layers", "assignments", "strategy")
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    TOTAL_LAYERS_FIELD_NUMBER: _ClassVar[int]
    ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: str
    model_name: str
    total_layers: int
    assignments: _containers.RepeatedCompositeFieldContainer[LayerAssignment]
    strategy: PartitionStrategy
    def __init__(self, pipeline_id: _Optional[str] = ..., model_name: _Optional[str] = ..., total_layers: _Optional[int] = ..., assignments: _Optional[_Iterable[_Union[LayerAssignment, _Mapping]]] = ..., strategy: _Optional[_Union[PartitionStrategy, str]] = ...) -> None: ...

class DistributedInferenceRequest(_message.Message):
    __slots__ = ("request_id", "pipeline_id", "input_tokens", "activation", "max_tokens", "temperature", "top_p")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    pipeline_id: str
    input_tokens: _containers.RepeatedScalarFieldContainer[int]
    activation: ActivationTransfer
    max_tokens: int
    temperature: float
    top_p: float
    def __init__(self, request_id: _Optional[str] = ..., pipeline_id: _Optional[str] = ..., input_tokens: _Optional[_Iterable[int]] = ..., activation: _Optional[_Union[ActivationTransfer, _Mapping]] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., top_p: _Optional[float] = ...) -> None: ...

class DistributedInferenceResponse(_message.Message):
    __slots__ = ("request_id", "activation", "output_tokens", "generated_text", "metrics")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    GENERATED_TEXT_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    activation: ActivationTransfer
    output_tokens: _containers.RepeatedScalarFieldContainer[int]
    generated_text: str
    metrics: PipelineMetrics
    def __init__(self, request_id: _Optional[str] = ..., activation: _Optional[_Union[ActivationTransfer, _Mapping]] = ..., output_tokens: _Optional[_Iterable[int]] = ..., generated_text: _Optional[str] = ..., metrics: _Optional[_Union[PipelineMetrics, _Mapping]] = ...) -> None: ...

class PipelineMetrics(_message.Message):
    __slots__ = ("stage_latency_ms", "transfer_latency_ms", "compute_latency_ms", "activation_size_bytes", "layers_processed")
    STAGE_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    ACTIVATION_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    LAYERS_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    stage_latency_ms: int
    transfer_latency_ms: int
    compute_latency_ms: int
    activation_size_bytes: int
    layers_processed: int
    def __init__(self, stage_latency_ms: _Optional[int] = ..., transfer_latency_ms: _Optional[int] = ..., compute_latency_ms: _Optional[int] = ..., activation_size_bytes: _Optional[int] = ..., layers_processed: _Optional[int] = ...) -> None: ...

class PipelineHealthRequest(_message.Message):
    __slots__ = ("pipeline_id",)
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: str
    def __init__(self, pipeline_id: _Optional[str] = ...) -> None: ...

class PipelineHealthResponse(_message.Message):
    __slots__ = ("pipeline_id", "status", "nodes")
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: str
    status: PipelineStatus
    nodes: _containers.RepeatedCompositeFieldContainer[NodeHealth]
    def __init__(self, pipeline_id: _Optional[str] = ..., status: _Optional[_Union[PipelineStatus, str]] = ..., nodes: _Optional[_Iterable[_Union[NodeHealth, _Mapping]]] = ...) -> None: ...

class NodeHealth(_message.Message):
    __slots__ = ("node_id", "is_healthy", "assigned_layers", "memory_used_bytes")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    IS_HEALTHY_FIELD_NUMBER: _ClassVar[int]
    ASSIGNED_LAYERS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    is_healthy: bool
    assigned_layers: int
    memory_used_bytes: int
    def __init__(self, node_id: _Optional[str] = ..., is_healthy: bool = ..., assigned_layers: _Optional[int] = ..., memory_used_bytes: _Optional[int] = ...) -> None: ...

class TransferAck(_message.Message):
    __slots__ = ("success", "message", "received_bytes")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RECEIVED_BYTES_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    received_bytes: int
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., received_bytes: _Optional[int] = ...) -> None: ...
