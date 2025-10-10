# Test Performance Notes

## Issue: Slow Test Execution

### Problem
The integration tests in `test_sharding_backward_compatibility.py` and `test_sharding_performance.py` are marked as `@pytest.mark.asyncio` and defined as `async def`, but most don't actually perform any async operations (no `await` statements).

### Impact
- **Slower execution**: Creating event loops for synchronous code adds overhead
- **Perceived hangs**: When running many tests together, the cumulative overhead makes it seem like tests are hanging
- **Resource usage**: Unnecessary event loop creation and cleanup

### Root Cause
```python
# Current (slow):
@pytest.mark.asyncio
async def test_something(self):
    # No await statements - this is synchronous code
    assert something == True

# Should be (fast):
def test_something(self):
    # Synchronous test for synchronous code
    assert something == True
```

### Why Tests Still Work
- pytest-asyncio handles async functions correctly
- Tests pass and assertions work
- Just slower than necessary

### Recommendation for Future Optimization

Convert synchronous tests to regular (non-async) tests:

**Tests that should remain async:**
- Tests with `await` statements
- Tests calling async functions
- Tests using `asyncio.gather()`, `asyncio.sleep()`, etc.

**Tests that should be converted to sync:**
- Tests only setting up mocks
- Tests only making assertions
- Tests with no `await` statements

### Example Conversion

```python
# BEFORE (unnecessarily async):
@pytest.mark.asyncio
async def test_load_non_sharded_model(self, model_manager):
    model_info = ModelInfo(...)
    model_manager.loaded_models[model_name] = model_info
    assert model_info is not None

# AFTER (properly sync):
def test_load_non_sharded_model(self, model_manager):
    model_info = ModelInfo(...)
    model_manager.loaded_models[model_name] = model_info
    assert model_info is not None
```

### Estimated Impact
- **Current**: ~3-5 seconds for 17 tests
- **After optimization**: ~0.5-1 second for 17 tests
- **Improvement**: 3-5x faster

### Files Affected
1. `tests/test_sharding_backward_compatibility.py` - Most tests can be sync
2. `tests/test_sharding_performance.py` - Tests with `asyncio.sleep()` should stay async, others can be sync

### Priority
**Low** - Tests work correctly, just slower than optimal. This is a performance optimization, not a bug fix.

### Action Items
- [ ] Audit all tests for actual async operations
- [ ] Remove `@pytest.mark.asyncio` and `async def` from synchronous tests
- [ ] Keep async only for tests that truly need it
- [ ] Re-run test suite to verify 3-5x speedup

## Workaround for Now

Run tests individually or in smaller groups to avoid cumulative overhead:

```bash
# Instead of running all at once:
pytest tests/test_sharding_*.py  # Slow

# Run separately:
pytest tests/test_sharded_model_download.py  # Fast
pytest tests/test_sharding_backward_compatibility.py  # Fast
pytest tests/test_sharding_performance.py  # Fast
```

Or use pytest-xdist for parallel execution:
```bash
pip install pytest-xdist
pytest tests/test_sharding_*.py -n auto  # Run in parallel
```
