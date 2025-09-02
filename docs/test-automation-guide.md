# Test Automation Guide for AI Agents Repository

## Overview

This guide provides comprehensive strategies for automating testing in the AI Agents repository, covering local development, CI/CD integration, and production deployment workflows.

## Test Suite Architecture

### Test Organization

```
tests/
├── conftest.py                     # Shared fixtures and test configuration
├── run_comprehensive_tests.py     # Main test runner with reporting
└── python/
    ├── test_base_agent_comprehensive.py      # BaseAgent unit tests
    ├── test_orchestrator_comprehensive.py   # Multi-agent integration tests
    ├── test_utils_comprehensive.py          # Utility module tests
    ├── test_base_agent.py                   # Original contract tests
    └── test_orchestrator.py                 # Original contract tests
```

### Test Categories

1. **Unit Tests**: Individual component testing with mocking
2. **Integration Tests**: Multi-component interaction testing
3. **Performance Tests**: Resource usage and speed benchmarking
4. **Windows Compatibility Tests**: Platform-specific functionality
5. **Contract Tests**: API and interface compliance

## Test Execution

### Local Development

```bash
# Run all tests with comprehensive reporting
python tests/run_comprehensive_tests.py

# Run specific test suites
python -m pytest tests/python/test_base_agent_comprehensive.py -v
python -m pytest tests/python/test_orchestrator_comprehensive.py -v
python -m pytest tests/python/test_utils_comprehensive.py -v

# Run with coverage
python -m pytest --cov=templates --cov=orchestrator --cov=utils --cov-report=html

# Run performance tests only
python -m pytest -m "not slow" --tb=short

# Run specific test markers
python -m pytest -m integration  # Integration tests only
python -m pytest -m unit         # Unit tests only
python -m pytest -m performance  # Performance tests only
```

### Windows-Specific Commands

```cmd
# Activate virtual environment
.venv\Scripts\activate

# Install test dependencies
pip install -r requirements-dev.txt

# Run tests with Windows paths
python tests\run_comprehensive_tests.py
```

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/test.yml`:

```yaml
name: Comprehensive Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: [3.11, 3.12, 3.13]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run comprehensive tests
      run: |
        python tests/run_comprehensive_tests.py
    
    - name: Upload test reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-reports-${{ matrix.os }}-${{ matrix.python-version }}
        path: test_reports/
    
    - name: Upload coverage to Codecov
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'
      uses: codecov/codecov-action@v3
      with:
        file: ./test_reports/coverage.xml
        fail_ci_if_error: true

  performance-benchmark:
    runs-on: ubuntu-latest
    needs: test
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run performance benchmarks
      run: |
        python -m pytest tests/python/ -m performance --benchmark-json=benchmark.json
    
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        auto-push: true
        github-token: ${{ secrets.GITHUB_TOKEN }}
```

### Azure DevOps Pipeline

Create `azure-pipelines.yml`:

```yaml
trigger:
- main
- develop

pool:
  vmImage: 'windows-latest'

variables:
  pythonVersion: '3.12'

stages:
- stage: Test
  displayName: 'Comprehensive Testing'
  jobs:
  - job: UnitTests
    displayName: 'Unit and Integration Tests'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
      displayName: 'Use Python $(pythonVersion)'
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
      displayName: 'Install dependencies'
    
    - script: |
        python tests/run_comprehensive_tests.py
      displayName: 'Run comprehensive tests'
    
    - task: PublishTestResults@2
      condition: always()
      inputs:
        testResultsFiles: 'test_reports/*_results.xml'
        testRunTitle: 'Python $(pythonVersion)'
    
    - task: PublishCodeCoverageResults@1
      inputs:
        codeCoverageTool: Cobertura
        summaryFileLocation: 'test_reports/coverage.xml'
        reportDirectory: 'test_reports/htmlcov'

- stage: Deploy
  displayName: 'Deployment Validation'
  dependsOn: Test
  condition: succeeded()
  jobs:
  - job: DeploymentTests
    displayName: 'Deployment Smoke Tests'
    steps:
    - script: |
        python tests/run_comprehensive_tests.py --integration-only
      displayName: 'Run deployment smoke tests'
```

## Test Configuration

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=templates
    --cov=orchestrator
    --cov=utils
    --cov-report=html:test_reports/htmlcov
    --cov-report=json:test_reports/coverage.json
    --cov-report=term-missing
    --cov-fail-under=90
    --junitxml=test_reports/junit.xml
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow-running tests
    windows: Windows-specific tests
```

### Coverage Configuration

`.coveragerc`:

```ini
[run]
source = .
omit = 
    */tests/*
    */venv/*
    */__pycache__/*
    */conftest.py
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    @abstractmethod

[html]
directory = test_reports/htmlcov
title = AI Agents Test Coverage Report

[json]
output = test_reports/coverage.json
```

## Quality Gates

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8

-   repo: local
    hooks:
    -   id: pytest-check
        name: pytest-check
        entry: python -m pytest tests/python/test_base_agent.py tests/python/test_orchestrator.py
        language: system
        pass_filenames: false
        always_run: true
```

### Quality Thresholds

- **Code Coverage**: Minimum 90% line coverage
- **Test Success Rate**: 100% for contract tests, 95%+ for comprehensive tests
- **Performance Benchmarks**:
  - Agent task processing: < 100ms per task
  - Memory usage: < 100MB increase for 100 tasks
  - Parallel execution: 3x speedup for 3 agents

### Failure Handling

1. **Immediate Failures**: Stop pipeline on critical test failures
2. **Flaky Test Detection**: Retry failed tests up to 3 times
3. **Performance Degradation**: Alert if benchmarks exceed thresholds by >20%
4. **Coverage Regression**: Fail if coverage drops below minimum threshold

## Test Data Management

### Fixtures and Test Data

```python
# conftest.py - Shared test fixtures
@pytest.fixture(scope="session")
def test_database():
    """Session-scoped test database"""
    db_path = tempfile.mktemp(suffix='.db')
    yield db_path
    os.unlink(db_path)

@pytest.fixture
def sample_tasks():
    """Sample tasks for testing"""
    return [
        Task(id="test_1", description="Test task 1", requirements={}),
        Task(id="test_2", description="Test task 2", requirements={"priority": "high"})
    ]
```

### Mock Strategies

1. **External Dependencies**: Mock API calls, database connections
2. **Slow Operations**: Mock file I/O, network requests
3. **Non-deterministic Behavior**: Mock random number generation, timestamps

## Performance Testing

### Benchmarking Framework

```python
import pytest
import time
from utils.observability.metrics import MetricsCollector

@pytest.mark.performance
def test_agent_processing_speed():
    """Benchmark agent processing speed"""
    metrics = MetricsCollector("benchmark")
    
    with metrics.timer("agent_processing"):
        # Test implementation
        pass
    
    stats = metrics.get_stats("agent_processing")
    assert stats["mean"] < 0.1  # 100ms threshold
```

### Load Testing

```python
import asyncio
import pytest

@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_agent_load():
    """Test system under concurrent load"""
    agents = [create_test_agent(f"agent_{i}") for i in range(10)]
    tasks = [agent.process_task("load_test") for agent in agents]
    
    start_time = time.perf_counter()
    results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    
    assert len(results) == 10
    assert end_time - start_time < 5.0  # Complete within 5 seconds
```

## Monitoring and Alerting

### Test Result Monitoring

1. **Success Rate Tracking**: Monitor test pass rates over time
2. **Performance Regression**: Alert on benchmark degradation
3. **Flaky Test Detection**: Identify and report unstable tests
4. **Coverage Trends**: Track coverage changes across commits

### Integration with Monitoring Tools

- **Grafana Dashboard**: Visualize test metrics and trends
- **Slack Notifications**: Alert team on test failures
- **Email Reports**: Daily/weekly test summary reports

## Best Practices

### Test Writing Guidelines

1. **Isolation**: Each test should be independent and repeatable
2. **Clarity**: Test names should clearly describe what is being tested
3. **Speed**: Unit tests should complete in < 10ms, integration tests < 1s
4. **Determinism**: Tests should not rely on external state or timing
5. **Coverage**: Aim for >95% line coverage on critical paths

### Maintenance Strategies

1. **Regular Review**: Weekly review of failed/flaky tests
2. **Test Refactoring**: Consolidate duplicate tests, improve readability
3. **Performance Optimization**: Profile and optimize slow tests
4. **Documentation Updates**: Keep test documentation current

### Debugging Failed Tests

```bash
# Run failed test with detailed output
python -m pytest tests/python/test_base_agent_comprehensive.py::test_name -v -s --tb=long

# Run with debugger
python -m pytest tests/python/test_base_agent_comprehensive.py::test_name --pdb

# Capture logs for debugging
python -m pytest tests/python/test_base_agent_comprehensive.py::test_name --log-cli-level=DEBUG
```

## Deployment Integration

### Environment-Specific Testing

- **Development**: Run comprehensive tests on every commit
- **Staging**: Full test suite + smoke tests + performance benchmarks  
- **Production**: Smoke tests + health checks + monitoring validation

### Rollback Triggers

Automatic rollback conditions:
- Critical test failures in production
- Performance degradation > 50%
- Error rate increase > 10%
- Memory leak detection

## Conclusion

This comprehensive test automation strategy ensures:

- **High Code Quality**: Through extensive test coverage and quality gates
- **Reliable Deployments**: Via thorough integration and smoke testing
- **Performance Assurance**: Through continuous benchmarking and monitoring
- **Platform Compatibility**: With Windows-specific test validation
- **Developer Productivity**: Through fast feedback loops and clear reporting

The test infrastructure supports both current development needs and future scalability requirements, providing a solid foundation for the AI Agents repository's continued evolution.