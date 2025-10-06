#!/bin/bash
# Run all test suites locally before pushing to CI

set -e

echo "================================"
echo "Running MLX Inference Test Suite"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
    fi
}

# Track overall status
OVERALL_STATUS=0

# 1. Run unit tests
echo ""
echo "Running unit tests..."
if pytest tests/ -v -m "not integration and not performance" --cov=. --cov-report=term; then
    print_status 0 "Unit tests passed"
else
    print_status 1 "Unit tests failed"
    OVERALL_STATUS=1
fi

# 2. Run integration tests
echo ""
echo "Running integration tests..."
if pytest tests/ -v -m "integration"; then
    print_status 0 "Integration tests passed"
else
    print_status 1 "Integration tests failed"
    OVERALL_STATUS=1
fi

# 3. Run performance tests (optional, can be slow)
if [ "$RUN_PERFORMANCE" = "true" ]; then
    echo ""
    echo "Running performance tests..."
    if pytest tests/test_performance_benchmarking.py -v -s; then
        print_status 0 "Performance tests passed"
    else
        print_status 1 "Performance tests failed"
        OVERALL_STATUS=1
    fi
fi

# 4. Run linting
echo ""
echo "Running code quality checks..."

# Black
if black --check .; then
    print_status 0 "Black formatting check passed"
else
    print_status 1 "Black formatting check failed"
    OVERALL_STATUS=1
fi

# isort
if isort --check-only .; then
    print_status 0 "isort check passed"
else
    print_status 1 "isort check failed"
    OVERALL_STATUS=1
fi

# Flake8
if flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics; then
    print_status 0 "Flake8 check passed"
else
    print_status 1 "Flake8 check failed"
    OVERALL_STATUS=1
fi

# 5. Summary
echo ""
echo "================================"
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed!${NC}"
fi
echo "================================"

exit $OVERALL_STATUS
