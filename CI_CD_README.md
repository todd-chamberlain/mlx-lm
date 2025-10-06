# CI/CD Pipeline Documentation

## Overview

Comprehensive CI/CD pipeline for the MLX distributed inference system with automated testing, Docker image building, deployment validation, and performance monitoring.

## Pipeline Architecture

### 1. Continuous Integration (CI)

**Workflow**: `.github/workflows/ci.yml`

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Jobs**:

1. **Unit Tests** (15 min timeout)
   - Python 3.11 environment
   - Install dependencies with pip cache
   - Run pytest with coverage
   - Upload coverage to Codecov
   - Generate HTML coverage report
   - Exclude integration and performance tests

2. **Integration Tests** (30 min timeout)
   - Redis service container
   - Run integration tests with markers
   - Test service communication
   - Upload test results

3. **Performance Tests** (45 min timeout)
   - Load performance baseline
   - Run performance benchmarks
   - Check for regressions
   - Generate performance reports
   - Comment on PR with results
   - Upload artifacts

4. **Lint and Code Quality** (10 min timeout)
   - Black code formatting check
   - isort import sorting check
   - Flake8 syntax checking
   - Pylint code analysis

5. **Security Scanning** (10 min timeout)
   - Safety dependency vulnerability check
   - Bandit security issue detection
   - Upload security reports

6. **Build Summary**
   - Aggregate all job results
   - Fail if critical jobs fail
   - Report overall build status

### 2. Docker Build and Publish

**Workflow**: `.github/workflows/docker-build.yml`

**Triggers**:
- Push to `main` branch
- Version tags (`v*`)
- Pull requests to `main`
- Manual workflow dispatch

**Services Built**:
- API Server
- Coordinator
- Worker

**Steps**:
1. Checkout code
2. Set up Docker Buildx
3. Log in to GitHub Container Registry
4. Extract metadata and tags
5. Build and push Docker images
6. Run Trivy vulnerability scanner
7. Upload security results

**Image Tags**:
- Branch name (e.g., `main`)
- PR number (e.g., `pr-123`)
- Semantic version (e.g., `v1.2.3`, `1.2`)
- Git SHA (e.g., `sha-abc123`)

### 3. Deployment Pipeline

**Workflow**: `.github/workflows/deploy.yml`

**Triggers**:
- Successful CI and Docker build workflows
- Manual workflow dispatch with environment selection

**Environments**:
- **Staging**: Automatic deployment after successful builds
- **Production**: Manual approval required

**Deployment Steps**:

1. **Deploy to Staging**
   - Checkout code
   - Set up kubectl
   - Deploy to staging environment
   - Wait for deployment completion
   - Run health checks
   - Run smoke tests
   - Rollback on failure

2. **Deploy to Production**
   - Requires staging deployment success
   - Manual approval gate
   - Deploy to production environment
   - Extended wait time
   - Run health checks
   - Run smoke tests
   - Rollback on failure
   - Notify on success

## Test Suites

### Unit Tests

**Location**: `tests/`  
**Marker**: No marker (default)  
**Exclusions**: `-m "not integration and not performance"`

**Coverage**:
- Individual component tests
- Mock external dependencies
- Fast execution (< 15 minutes)

### Integration Tests

**Location**: `tests/`  
**Marker**: `-m "integration"`

**Coverage**:
- Service-to-service communication
- Database interactions
- External service integration
- Moderate execution time (< 30 minutes)

### Performance Tests

**Location**: `tests/test_performance_benchmarking.py`  
**Marker**: `-m "performance"` or explicit file

**Coverage**:
- Latency benchmarks
- Throughput measurement
- Concurrent request handling
- Stress testing
- Longer execution time (< 45 minutes)

### Smoke Tests

**Location**: `tests/smoke/`

**Coverage**:
- Basic functionality validation
- Health endpoint checks
- Critical path testing
- Fast execution (< 5 minutes)

## Performance Regression Detection

### Baseline Management

**Baseline File**: `performance_baseline.json`

**Creation**:
```bash
# Run performance tests to create baseline
pytest tests/test_performance_benchmarking.py -v -s

# Baseline is automatically created if it doesn't exist
```

**Update**:
```bash
# Update baseline after intentional performance improvements
python tests/performance_regression.py --update-baseline
```

### Regression Thresholds

- **RPS Degradation**: 10% decrease
- **Latency Increase**: 20% increase (p95, p99)
- **Error Rate Increase**: 5% increase
- **CPU Usage Increase**: 30% increase
- **Memory Usage Increase**: 30% increase

### Regression Handling

1. **Detection**: Automatic comparison against baseline
2. **Reporting**: Comment on PR with results
3. **Artifacts**: Upload detailed reports
4. **Gating**: Optional - fail build on regression

## Docker Image Management

### Registry

**Registry**: GitHub Container Registry (ghcr.io)  
**Authentication**: GitHub token (automatic)

### Image Naming

```
ghcr.io/<owner>/<repo>-<service>:<tag>
```

**Examples**:
- `ghcr.io/myorg/mlx-inference-api:main`
- `ghcr.io/myorg/mlx-inference-coordinator:v1.2.3`
- `ghcr.io/myorg/mlx-inference-worker:sha-abc123`

### Build Optimization

- **Multi-stage builds**: Reduce image size
- **Layer caching**: GitHub Actions cache
- **BuildKit**: Advanced build features
- **Cache modes**: `type=gha,mode=max`

### Security Scanning

**Tool**: Trivy  
**Format**: SARIF  
**Upload**: GitHub Security tab

**Scan Coverage**:
- OS vulnerabilities
- Application dependencies
- Configuration issues
- Secrets detection

## Deployment Validation

### Health Checks

**Script**: `scripts/health_check.sh`

**Checks**:
- HTTP endpoint availability
- Response time validation
- Service status verification
- Dependency health

**Usage**:
```bash
./scripts/health_check.sh <host> <port>
```

### Smoke Tests

**Purpose**: Quick validation of critical functionality

**Tests**:
- API server health endpoint
- Metrics endpoint
- Models endpoint
- Basic completion request
- Coordinator connectivity
- End-to-end latency
- Concurrent request handling

**Execution**:
```bash
pytest tests/smoke/ -v --base-url=https://staging.example.com
```

### Rollback Procedures

**Automatic Rollback**:
- Triggered on health check failure
- Triggered on smoke test failure
- Reverts to previous deployment

**Manual Rollback**:
```bash
# Kubernetes
kubectl rollout undo deployment/api-server -n production

# Docker Compose
docker-compose down
docker-compose up -d --force-recreate
```

## Environment Configuration

### Staging Environment

**URL**: `https://staging.example.com`  
**Purpose**: Pre-production testing  
**Deployment**: Automatic on successful builds

**Configuration**:
- Reduced resource limits
- Test data only
- Debug logging enabled
- Performance monitoring

### Production Environment

**URL**: `https://api.example.com`  
**Purpose**: Live production traffic  
**Deployment**: Manual approval required

**Configuration**:
- Full resource allocation
- Production data
- Info/Warning logging
- Full monitoring and alerting

## Secrets Management

### Required Secrets

**GitHub Secrets**:
- `GITHUB_TOKEN`: Automatic (for registry access)
- `CODECOV_TOKEN`: Optional (for coverage upload)
- `SLACK_WEBHOOK`: Optional (for notifications)

**Kubernetes Secrets**:
- API keys
- Database credentials
- Service tokens
- TLS certificates

### Secret Rotation

1. Update secret in secret manager
2. Update Kubernetes secret
3. Restart affected services
4. Verify functionality

## Monitoring and Alerting

### CI/CD Metrics

**Tracked Metrics**:
- Build success rate
- Test pass rate
- Build duration
- Deployment frequency
- Mean time to recovery (MTTR)

### Performance Metrics

**Tracked Metrics**:
- Request latency (p50, p95, p99)
- Requests per second
- Error rate
- Resource usage

### Alerts

**Alert Conditions**:
- Build failures
- Test failures
- Performance regressions
- Deployment failures
- Security vulnerabilities

## Best Practices

### 1. Branch Strategy

- **main**: Production-ready code
- **develop**: Integration branch
- **feature/***: Feature branches
- **hotfix/***: Emergency fixes

### 2. Commit Messages

Follow conventional commits:
```
feat: add new feature
fix: bug fix
docs: documentation update
test: add tests
ci: CI/CD changes
```

### 3. Pull Requests

- Run CI checks before merging
- Require code review
- Check performance impact
- Update documentation

### 4. Version Tagging

Use semantic versioning:
```
v1.0.0 - Major release
v1.1.0 - Minor release
v1.1.1 - Patch release
```

### 5. Deployment Schedule

- **Staging**: Continuous deployment
- **Production**: Scheduled windows
- **Hotfixes**: As needed with approval

## Troubleshooting

### Build Failures

**Unit Test Failures**:
1. Check test logs in artifacts
2. Run tests locally
3. Fix failing tests
4. Push fix

**Integration Test Failures**:
1. Check service logs
2. Verify service dependencies
3. Check network connectivity
4. Review configuration

**Performance Test Failures**:
1. Check performance report
2. Compare with baseline
3. Identify bottlenecks
4. Optimize or update baseline

### Deployment Failures

**Health Check Failures**:
1. Check service logs
2. Verify configuration
3. Check resource availability
4. Review recent changes

**Smoke Test Failures**:
1. Check test output
2. Verify endpoints
3. Check service status
4. Review deployment logs

### Rollback Issues

**Rollback Failures**:
1. Check rollback logs
2. Verify previous version
3. Manual intervention may be needed
4. Contact operations team

## Maintenance

### Regular Tasks

**Weekly**:
- Review build metrics
- Check security scan results
- Update dependencies
- Review performance trends

**Monthly**:
- Update baseline metrics
- Review and optimize pipelines
- Update documentation
- Security audit

**Quarterly**:
- Major dependency updates
- Infrastructure review
- Disaster recovery testing
- Performance optimization

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Deployment Strategies](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Performance Testing Guide](tests/PERFORMANCE_TESTING_README.md)

## Support

For CI/CD issues:
1. Check workflow logs in GitHub Actions
2. Review this documentation
3. Check service status
4. Contact DevOps team
