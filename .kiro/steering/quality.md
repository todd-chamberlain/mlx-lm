# Code Quality & Maintenance Standards

## Core Principles

This is a complex distributed system that must remain maintainable long-term. Every change should leave the codebase cleaner than you found it.

### Work Cleanly

- **Delete before adding** - Remove unused code, imports, and dependencies proactively
- **Refactor as you go** - Don't let technical debt accumulate
- **One purpose per file** - If a file does too much, split it
- **Clear boundaries** - Services should have well-defined interfaces

### Active Cleanup

When working in any part of the codebase:

1. **Remove dead code immediately**
   - Unused functions, classes, or imports
   - Commented-out code blocks (use git history instead)
   - Deprecated features that are no longer called
   - Debug print statements and temporary logging

2. **Fix issues you encounter**
   - Misleading variable names → rename them
   - Missing type hints → add them
   - Unclear logic → add comments or refactor
   - Inconsistent patterns → align with conventions

3. **Update documentation alongside code**
   - Outdated docstrings → update or remove
   - Incorrect README sections → fix immediately
   - Missing API documentation → add it
   - Stale configuration examples → update them

### Weed Out Bad Code

Actively identify and eliminate:

- **Duplicated logic** - Extract to shared utilities in `common/`
- **God classes** - Break down classes with too many responsibilities
- **Deep nesting** - Refactor complex conditionals (max 3 levels)
- **Long functions** - Split functions over 50 lines into smaller pieces
- **Magic numbers** - Replace with named constants
- **Unclear names** - Use descriptive names even if longer
- **Missing error handling** - Add proper try/except with specific exceptions
- **Synchronous blocking** - Convert to async where appropriate

### Code Smells to Eliminate

```python
# BAD: Magic numbers, unclear names
def process(x, y):
    if x > 100:
        return y * 0.85
    return y

# GOOD: Named constants, clear intent
MAX_BATCH_SIZE = 100
CACHE_HIT_DISCOUNT = 0.85

def calculate_processing_cost(batch_size: int, base_cost: float) -> float:
    """Calculate cost with cache hit discount for large batches."""
    if batch_size > MAX_BATCH_SIZE:
        return base_cost * CACHE_HIT_DISCOUNT
    return base_cost
```

```python
# BAD: Deep nesting, unclear flow
def route_request(req):
    if req:
        if req.model:
            if req.model in models:
                if workers_available():
                    return route_to_worker(req)
    return None

# GOOD: Early returns, clear flow
def route_request(request: InferenceRequest) -> Optional[WorkerInfo]:
    """Route inference request to available worker."""
    if not request or not request.model:
        return None
    
    if request.model not in available_models:
        raise ModelNotFoundError(f"Model {request.model} not available")
    
    if not workers_available():
        raise NoWorkersAvailableError("All workers busy")
    
    return route_to_worker(request)
```

## Maintainability Checklist

Before committing any change, verify:

- [ ] No unused imports or variables
- [ ] All functions have type hints
- [ ] Complex logic has explanatory comments
- [ ] Error cases are handled explicitly
- [ ] Tests exist and pass
- [ ] Documentation is updated
- [ ] No TODO comments without tracking issues
- [ ] Logging includes relevant context
- [ ] Resource cleanup (connections, files, memory)
- [ ] Consistent with existing patterns

## Documentation Standards

### Code Documentation

```python
def distribute_model_layers(
    model: nn.Module,
    num_workers: int,
    memory_per_worker: float
) -> List[LayerPartition]:
    """
    Distribute model layers across workers based on memory constraints.
    
    Args:
        model: The model to partition
        num_workers: Number of available workers
        memory_per_worker: Available memory per worker in GB
    
    Returns:
        List of layer partitions, one per worker
    
    Raises:
        InsufficientMemoryError: If model cannot fit across workers
        InvalidPartitionError: If layer distribution is invalid
    
    Example:
        >>> partitions = distribute_model_layers(llama_70b, 2, 32.0)
        >>> len(partitions)
        2
    """
```

### README and Docs

Keep documentation:
- **Accurate** - Test examples before committing
- **Current** - Update when behavior changes
- **Concise** - Remove outdated sections
- **Practical** - Focus on common use cases

## Complexity Management

### Service Boundaries

Each service should:
- Have a single clear responsibility
- Communicate via well-defined interfaces (gRPC, REST)
- Be independently testable
- Handle its own errors gracefully

### Dependency Management

- **Minimize dependencies** - Only add libraries when necessary
- **Pin versions** - Use specific versions in requirements.txt
- **Audit regularly** - Remove unused dependencies
- **Update deliberately** - Test thoroughly after updates

### Configuration Management

- **Environment variables** for deployment-specific values
- **YAML files** for complex structured configuration
- **Validation** using Pydantic models
- **Defaults** that work for development
- **Documentation** for every configuration option

## Testing Standards

### Test Coverage

Maintain 100% coverage for:
- Core inference logic (`worker/inference_engine.py`)
- Request routing (`coordinator/request_router.py`)
- Model partitioning (`worker/model_partitioner.py`)
- Error recovery (`common/error_recovery.py`)

### Test Quality

```python
# BAD: Unclear test, no assertions
def test_inference():
    result = engine.infer("test")
    assert result

# GOOD: Clear intent, specific assertions
def test_inference_returns_valid_tokens():
    """Inference should return token IDs within vocabulary range."""
    prompt = "Hello, world!"
    result = engine.infer(prompt, max_tokens=10)
    
    assert len(result.tokens) <= 10
    assert all(0 <= token < engine.vocab_size for token in result.tokens)
    assert result.latency_ms > 0
```

## Performance Considerations

- **Profile before optimizing** - Use actual metrics
- **Async for I/O** - Network, disk, database operations
- **Batch when possible** - Reduce per-request overhead
- **Cache intelligently** - Balance memory vs computation
- **Monitor in production** - Track latency, throughput, memory

## Error Handling

```python
# BAD: Bare except, no context
try:
    result = worker.infer(request)
except:
    return None

# GOOD: Specific exceptions, context, recovery
try:
    result = await worker.infer(request)
except WorkerTimeoutError as e:
    logger.warning(
        "Worker timeout, retrying with different worker",
        extra={"worker_id": worker.id, "request_id": request.id}
    )
    return await self._retry_with_fallback(request)
except ModelNotLoadedError as e:
    logger.error(
        "Model not loaded on worker",
        extra={"worker_id": worker.id, "model": request.model}
    )
    raise InferenceError(f"Model {request.model} unavailable") from e
```

## Long-Term Health

### Regular Maintenance Tasks

- **Weekly**: Review and close stale issues/PRs
- **Monthly**: Audit dependencies for updates and vulnerabilities
- **Quarterly**: Review and refactor high-complexity modules
- **Annually**: Evaluate architecture for scaling needs

### Technical Debt

- **Track it** - Create issues for known problems
- **Prioritize it** - Schedule cleanup alongside features
- **Prevent it** - Don't merge code that adds debt
- **Pay it down** - Dedicate time to refactoring

### Code Review Focus

When reviewing code, check for:
- Unnecessary complexity
- Missing error handling
- Inadequate testing
- Unclear naming
- Undocumented behavior
- Performance implications
- Security concerns

## Remember

> "Always leave the code better than you found it." - The Boy Scout Rule

This is a production system handling real inference workloads. Code quality directly impacts reliability, performance, and the ability to add features quickly. Invest in cleanliness now to save time later.
