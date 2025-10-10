#!/bin/bash
# Run core tests to validate the unified system

echo "=========================================="
echo "RUNNING CORE SYSTEM TESTS"
echo "=========================================="
echo ""

# Track results
TOTAL=0
PASSED=0
FAILED=0

run_test() {
    local test_file=$1
    local test_name=$2
    
    echo "Testing: $test_name"
    TOTAL=$((TOTAL + 1))
    
    if pytest "$test_file" -q --tb=no 2>&1 | grep -q "passed"; then
        echo "✓ PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "✗ FAILED"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# Core functionality tests
run_test "tests/test_model_partitioner.py" "Model Partitioner (21 tests)"
run_test "tests/test_coordinator_service.py" "Coordinator Service (10 tests)"
run_test "tests/test_worker_pool.py" "Worker Pool Management"
run_test "tests/test_request_router.py" "Request Routing"
run_test "tests/test_model_manager.py" "Model Manager"
run_test "tests/test_activation_cache.py" "Activation Cache (34 tests)"
run_test "tests/test_kv_cache.py" "KV Cache"
run_test "tests/test_memory_pool.py" "Memory Pool (79 tests)"

# Sharding tests
run_test "tests/test_sharded_model_download.py" "Sharded Model Download"
run_test "tests/test_api_sharding_endpoints.py" "API Sharding Endpoints"

echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo "Total test suites: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 ALL CORE TESTS PASSED!"
    echo "The unified system is working correctly."
    exit 0
else
    echo "⚠️  Some tests failed. Check output above."
    exit 1
fi
