from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InferenceRequest(_message.Message):
    __slots__ = ("request_id", "prompt", "max_tokens", "temperature", "top_p", "seed", "model", "use_cache")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    USE_CACHE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    model: str
    use_cache: bool
    def __init__(self, request_id: _Optional[str] = ..., prompt: _Optional[str] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., seed: _Optional[int] = ..., model: _Optional[str] = ..., use_cache: bool = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("request_id", "text", "tokens", "prompt_tokens", "completion_tokens", "tokens_per_second", "worker_id", "latency_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOKENS_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    text: str
    tokens: _containers.RepeatedScalarFieldContainer[int]
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    worker_id: str
    latency_ms: float
    def __init__(self, request_id: _Optional[str] = ..., text: _Optional[str] = ..., tokens: _Optional[_Iterable[int]] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., tokens_per_second: _Optional[float] = ..., worker_id: _Optional[str] = ..., latency_ms: _Optional[float] = ...) -> None: ...

class StreamChunk(_message.Message):
    __slots__ = ("request_id", "text", "token", "is_final", "worker_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    text: str
    token: int
    is_final: bool
    worker_id: str
    def __init__(self, request_id: _Optional[str] = ..., text: _Optional[str] = ..., token: _Optional[int] = ..., is_final: bool = ..., worker_id: _Optional[str] = ...) -> None: ...

class WorkerRegistration(_message.Message):
    __slots__ = ("worker_id", "address", "device_info", "max_concurrent", "supported_models")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_INFO_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENT_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_MODELS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    address: str
    device_info: DeviceInfo
    max_concurrent: int
    supported_models: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., device_info: _Optional[_Union[DeviceInfo, _Mapping]] = ..., max_concurrent: _Optional[int] = ..., supported_models: _Optional[_Iterable[str]] = ...) -> None: ...

class RegistrationResponse(_message.Message):
    __slots__ = ("success", "message", "coordinator_id")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    COORDINATOR_ID_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    coordinator_id: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., coordinator_id: _Optional[str] = ...) -> None: ...

class DeviceInfo(_message.Message):
    __slots__ = ("model", "memory_gb", "gpu_cores", "os_version", "mlx_version")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    MEMORY_GB_FIELD_NUMBER: _ClassVar[int]
    GPU_CORES_FIELD_NUMBER: _ClassVar[int]
    OS_VERSION_FIELD_NUMBER: _ClassVar[int]
    MLX_VERSION_FIELD_NUMBER: _ClassVar[int]
    model: str
    memory_gb: int
    gpu_cores: int
    os_version: str
    mlx_version: str
    def __init__(self, model: _Optional[str] = ..., memory_gb: _Optional[int] = ..., gpu_cores: _Optional[int] = ..., os_version: _Optional[str] = ..., mlx_version: _Optional[str] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("worker_id", "status", "metrics", "supported_models")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_MODELS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    status: WorkerStatus
    metrics: WorkerMetrics
    supported_models: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, worker_id: _Optional[str] = ..., status: _Optional[_Union[WorkerStatus, _Mapping]] = ..., metrics: _Optional[_Union[WorkerMetrics, _Mapping]] = ..., supported_models: _Optional[_Iterable[str]] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("acknowledged", "message", "heartbeat_interval")
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    message: str
    heartbeat_interval: int
    def __init__(self, acknowledged: bool = ..., message: _Optional[str] = ..., heartbeat_interval: _Optional[int] = ...) -> None: ...

class WorkerStatus(_message.Message):
    __slots__ = ("status", "current_load", "memory_usage_percent", "uptime_seconds")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_LOAD_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USAGE_PERCENT_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    status: str
    current_load: int
    memory_usage_percent: float
    uptime_seconds: int
    def __init__(self, status: _Optional[str] = ..., current_load: _Optional[int] = ..., memory_usage_percent: _Optional[float] = ..., uptime_seconds: _Optional[int] = ...) -> None: ...

class WorkerMetrics(_message.Message):
    __slots__ = ("avg_latency_ms", "tokens_per_second", "total_requests", "failed_requests", "active_requests", "memory_usage_gb")
    AVG_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    TOKENS_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    TOTAL_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    FAILED_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USAGE_GB_FIELD_NUMBER: _ClassVar[int]
    avg_latency_ms: float
    tokens_per_second: float
    total_requests: int
    failed_requests: int
    active_requests: int
    memory_usage_gb: float
    def __init__(self, avg_latency_ms: _Optional[float] = ..., tokens_per_second: _Optional[float] = ..., total_requests: _Optional[int] = ..., failed_requests: _Optional[int] = ..., active_requests: _Optional[int] = ..., memory_usage_gb: _Optional[float] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ("include_workers",)
    INCLUDE_WORKERS_FIELD_NUMBER: _ClassVar[int]
    include_workers: bool
    def __init__(self, include_workers: bool = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status", "total_workers", "healthy_workers", "active_requests", "worker_health")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_WORKERS_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_WORKERS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    WORKER_HEALTH_FIELD_NUMBER: _ClassVar[int]
    status: str
    total_workers: int
    healthy_workers: int
    active_requests: int
    worker_health: _containers.RepeatedCompositeFieldContainer[WorkerHealthInfo]
    def __init__(self, status: _Optional[str] = ..., total_workers: _Optional[int] = ..., healthy_workers: _Optional[int] = ..., active_requests: _Optional[int] = ..., worker_health: _Optional[_Iterable[_Union[WorkerHealthInfo, _Mapping]]] = ...) -> None: ...

class WorkerHealthInfo(_message.Message):
    __slots__ = ("worker_id", "status", "last_heartbeat_seconds", "current_load")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LAST_HEARTBEAT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_LOAD_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    status: str
    last_heartbeat_seconds: float
    current_load: int
    def __init__(self, worker_id: _Optional[str] = ..., status: _Optional[str] = ..., last_heartbeat_seconds: _Optional[float] = ..., current_load: _Optional[int] = ...) -> None: ...

class HealthStatus(_message.Message):
    __slots__ = ("status", "message", "metrics")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    status: str
    message: str
    metrics: WorkerMetrics
    def __init__(self, status: _Optional[str] = ..., message: _Optional[str] = ..., metrics: _Optional[_Union[WorkerMetrics, _Mapping]] = ...) -> None: ...

class ModelDownloadRequest(_message.Message):
    __slots__ = ("model_name", "force_redownload")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    FORCE_REDOWNLOAD_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    force_redownload: bool
    def __init__(self, model_name: _Optional[str] = ..., force_redownload: bool = ...) -> None: ...

class ModelDownloadResponse(_message.Message):
    __slots__ = ("success", "message", "download_id", "workers_notified")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    WORKERS_NOTIFIED_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    download_id: str
    workers_notified: int
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., download_id: _Optional[str] = ..., workers_notified: _Optional[int] = ...) -> None: ...

class ModelStatusRequest(_message.Message):
    __slots__ = ("model_name", "download_id")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    download_id: str
    def __init__(self, model_name: _Optional[str] = ..., download_id: _Optional[str] = ...) -> None: ...

class ModelStatusResponse(_message.Message):
    __slots__ = ("model_name", "status", "progress_percent", "size_bytes", "worker_status")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_PERCENT_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    WORKER_STATUS_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    status: str
    progress_percent: float
    size_bytes: int
    worker_status: _containers.RepeatedCompositeFieldContainer[WorkerModelStatus]
    def __init__(self, model_name: _Optional[str] = ..., status: _Optional[str] = ..., progress_percent: _Optional[float] = ..., size_bytes: _Optional[int] = ..., worker_status: _Optional[_Iterable[_Union[WorkerModelStatus, _Mapping]]] = ...) -> None: ...

class WorkerModelStatus(_message.Message):
    __slots__ = ("worker_id", "status", "progress_percent", "error")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_PERCENT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    status: str
    progress_percent: float
    error: str
    def __init__(self, worker_id: _Optional[str] = ..., status: _Optional[str] = ..., progress_percent: _Optional[float] = ..., error: _Optional[str] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
