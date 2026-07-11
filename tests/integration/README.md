# Integration Tests for Redubber v2.0

Comprehensive end-to-end integration tests verifying the complete system: API → database → file operations → async TTS → task queue.

## Test Structure

```
tests/integration/
├── __init__.py
├── conftest.py                        # Integration test fixtures
├── test_project_workflow.py          # Project management tests
├── test_redubbing_workflow.py        # Redubbing pipeline tests
├── test_task_queue.py                 # Task queue management tests
├── test_file_operations.py            # File operation tests
└── test_async_tts_performance.py      # Performance benchmarks
```

## Running Tests

### All Integration Tests

```bash
# Run all integration tests (slow)
poetry run pytest -m integration

# Run with verbose output
poetry run pytest -m integration -v

# Run with output capture disabled (see print statements)
poetry run pytest -m integration -s
```

### Unit Tests Only (Fast)

```bash
# Skip integration tests
poetry run pytest -m "not integration"
```

### Performance Benchmarks

```bash
# Run only performance tests
poetry run pytest -m performance

# Run performance tests with timing
poetry run pytest -m performance -v -s
```

### Specific Test Files

```bash
# Project workflow tests
poetry run pytest tests/integration/test_project_workflow.py -v

# Redubbing workflow tests
poetry run pytest tests/integration/test_redubbing_workflow.py -v

# Task queue tests
poetry run pytest tests/integration/test_task_queue.py -v

# File operations tests
poetry run pytest tests/integration/test_file_operations.py -v

# Performance tests
poetry run pytest tests/integration/test_async_tts_performance.py -v
```

### All Tests (Unit + Integration)

```bash
# Run everything
poetry run pytest

# With coverage report
poetry run pytest --cov=app --cov-report=html
```

## Test Categories

### 1. Project Workflow Tests (`test_project_workflow.py`)

**Tests:**
- `test_create_project_and_scan` - Create project → trigger scan → verify videos indexed
- `test_list_projects` - Multiple projects → list → verify ordering
- `test_update_voice_settings` - Update voice → retrieve → verify persistence
- `test_delete_project` - Delete → verify DB cleanup
- `test_scan_concurrency` - Concurrent scan requests → verify protection
- `test_project_metadata_persistence` - Timestamps and metadata consistency

**What it verifies:**
- Project CRUD operations
- File scanning and indexing
- Database persistence
- Concurrent request handling
- Metadata integrity

### 2. Redubbing Workflow Tests (`test_redubbing_workflow.py`)

**Tests:**
- `test_submit_redub_task` - Submit job → verify task_id returned
- `test_task_status_polling` - Submit → poll status → verify progress updates
- `test_task_completion` - Full pipeline → verify completed status (TODO: requires API mocking)
- `test_file_replacement` - Redub → verify backup created → verify original replaced
- `test_metadata_sync` - Redub → verify DB audio_streams updated
- `test_multiple_queued_tasks` - Submit 3 jobs → verify queuing works
- `test_redub_with_voice_settings` - Custom voice settings applied correctly

**What it verifies:**
- End-to-end redubbing pipeline
- Task submission and tracking
- Status updates and progress reporting
- File backup and replacement
- Voice settings application

### 3. Task Queue Tests (`test_task_queue.py`)

**Tests:**
- `test_task_cancellation` - Submit → cancel → verify cleanup
- `test_graceful_shutdown` - Submit → shutdown app → verify task completion
- `test_concurrent_limit` - Submit 10 jobs → verify max_concurrent respected
- `test_queue_full` - Fill queue → verify 503 or queue full error
- `test_failed_task_recovery` - Simulate failure → verify error status
- `test_task_progress_updates` - Verify progress tracking
- `test_task_timeout_handling` - Long-running task timeout

**What it verifies:**
- Task queue capacity limits
- Concurrent task processing
- Graceful shutdown behavior
- Error handling and recovery
- Task cancellation

### 4. File Operations Tests (`test_file_operations.py`)

**Tests:**
- `test_atomic_file_replacement` - Verify os.replace() used (no partial writes)
- `test_backup_creation` - Redub → verify timestamped backup exists
- `test_rollback_on_failure` - Simulate failure → verify backup restored
- `test_disk_space_check` - Insufficient space → verify pre-flight fails
- `test_cleanup_temp_files` - Redub → verify temp dirs deleted
- `test_concurrent_file_access` - Concurrent operations don't interfere
- `test_path_traversal_protection` - Security: path validation
- `test_large_file_handling` - Chunked I/O for large files
- `test_file_permissions` - Permission handling

**What it verifies:**
- Atomic file operations
- Backup and rollback mechanisms
- Temporary file cleanup
- Disk space management
- Security and permissions

### 5. Performance Benchmarks (`test_async_tts_performance.py`)

**Tests:**
- `test_async_tts_performance` - Verify 5x speedup with async TTS
- `test_concurrent_limit_performance` - Different concurrency levels
- `test_error_handling_doesnt_block_queue` - Errors don't block other tasks
- `test_memory_efficiency_large_batch` - Large batch memory efficiency
- `test_sequential_vs_async_comparison` - Direct comparison
- `test_progress_callback_performance` - Callback overhead minimal

**What it verifies:**
- 5x performance improvement from async TTS
- Optimal concurrency levels
- Error resilience
- Memory efficiency
- Progress tracking overhead

## Key Fixtures

### `integration_client`

FastAPI TestClient with full lifespan (task queue, database, etc.)

```python
def test_example(integration_client: TestClient):
    response = integration_client.get("/api/health")
    assert response.status_code == 200
```

### `integration_db`

DatabaseManager instance for direct database operations

```python
def test_example(integration_db: DatabaseManager):
    project_id = integration_db.add_project("/path", "Test Project")
    assert project_id > 0
```

### `test_video_dir`

Directory containing sample video files for testing

```python
def test_example(test_video_dir: Path):
    videos = list(test_video_dir.glob("*.mp4"))
    assert len(videos) > 0
```

### `sample_project_path`

Path to sample project directory (string)

```python
def test_example(integration_client: TestClient, sample_project_path: str):
    response = integration_client.post(
        "/api/projects/",
        json={"path": sample_project_path}
    )
```

### `sample_video_path`

Path to sample video file (string)

```python
def test_example(sample_video_path: str):
    assert Path(sample_video_path).exists()
```

## Test Markers

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.integration` - Slow integration tests
- `@pytest.mark.performance` - Performance benchmark tests
- `@pytest.mark.asyncio` - Async tests (auto-detected)

## Configuration

Integration tests use isolated temporary directories and test database:

```python
# From conftest.py
monkeypatch.setattr(settings, "tmp_dir", str(tmp_dir))
monkeypatch.setattr(settings, "mounted_storage", str(storage_dir))
monkeypatch.setattr(settings, "database_url", str(storage_dir / "test_redubber.db"))
monkeypatch.setattr(settings, "openai_api_key", "test-key-integration")
monkeypatch.setattr(settings, "max_concurrent_redubs", 2)
monkeypatch.setattr(settings, "task_queue_max_size", 10)
```

## Expected Results

### Successful Test Run

```
tests/integration/test_project_workflow.py::test_create_project_and_scan PASSED
tests/integration/test_project_workflow.py::test_list_projects PASSED
tests/integration/test_project_workflow.py::test_update_voice_settings PASSED
...
============================= 35 passed in 45.23s ==============================
```

### Performance Benchmark Example

```
✓ Async TTS Performance: 0.15s for 100 segments
  Estimated speedup vs sequential (~10s): 66.7x

=== Concurrency Performance Test ===
  max_concurrent= 1: 2.503s (speedup: 1.0x)
  max_concurrent= 5: 0.511s (speedup: 4.9x)
  max_concurrent=10: 0.256s (speedup: 9.8x)
  max_concurrent=25: 0.104s (speedup: 24.1x)
  max_concurrent=50: 0.052s (speedup: 48.1x)
```

## Troubleshooting

### Tests Fail with "OpenAI API Key" Error

Integration tests use a mock API key. If tests are actually calling OpenAI API:
- Check that mocks are properly configured
- Verify `settings.openai_api_key` is patched in fixtures

### Tests Timeout

Some tests have built-in timeouts (e.g., 10 seconds for scanning):
- Increase timeout values if running on slower hardware
- Check that task queue workers are starting correctly

### Database Errors

If tests fail with database errors:
- Verify SQLite is available
- Check that temporary directories are writable
- Ensure proper cleanup between tests

### Import Errors

If tests fail to import modules:
```bash
# Verify all dependencies installed
poetry install

# Activate virtual environment
poetry shell
```

## Development Workflow

1. **Write new test:**
   ```python
   @pytest.mark.integration
   def test_new_feature(integration_client: TestClient):
       # Test implementation
       pass
   ```

2. **Run just that test:**
   ```bash
   poetry run pytest tests/integration/test_file.py::test_new_feature -v
   ```

3. **Run all integration tests:**
   ```bash
   poetry run pytest -m integration
   ```

4. **Run full test suite:**
   ```bash
   poetry run pytest
   ```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Run unit tests
        run: poetry run pytest -m "not integration"
      - name: Run integration tests
        run: poetry run pytest -m integration
      - name: Run performance benchmarks
        run: poetry run pytest -m performance -s
```

## Test Coverage

To generate coverage report:

```bash
# Run with coverage
poetry run pytest --cov=app --cov=tests --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

## Notes

- Integration tests create isolated temporary directories for each test session
- Database is recreated for each test to ensure isolation
- File operations use temporary paths to avoid affecting production data
- Performance tests include assertions on timing to catch regressions
- Some tests are marked with `pytest.skip()` where full implementation requires external API mocking

## Future Improvements

1. **API Mocking:** Complete OpenAI API mocking for full end-to-end tests
2. **Load Testing:** Add tests for system behavior under sustained load
3. **Stress Testing:** Test with very large video files and long transcripts
4. **Security Testing:** Expand path traversal and injection attack tests
5. **Monitoring Integration:** Test integration with monitoring/alerting systems
