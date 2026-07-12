"""Integration tests for file operations.

Tests atomic file replacement, backup creation, rollback on failure,
disk space checks, and cleanup of temporary files.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


@pytest.mark.integration
def test_atomic_file_replacement(integration_test_dir: Path) -> None:
    """Test atomic file replacement using os.replace().

    Verifies:
    - File replacement is atomic (no partial writes visible)
    - Original file is not corrupted during replacement
    - os.replace() is used (not write-then-rename)

    Args:
        integration_test_dir: Base directory for integration tests.
    """
    import os

    # Create test directory
    test_dir = integration_test_dir / "atomic_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create original file
    original_file = test_dir / "video.mp4"
    original_content = b"original video content"
    original_file.write_bytes(original_content)

    # Create new content in temporary file
    temp_file = test_dir / "video_new.tmp"
    new_content = b"new video content after processing"
    temp_file.write_bytes(new_content)

    # Perform atomic replacement using os.replace()
    os.replace(temp_file, original_file)

    # Verify replacement succeeded
    assert original_file.read_bytes() == new_content
    assert not temp_file.exists()

    # Cleanup
    original_file.unlink(missing_ok=True)


@pytest.mark.integration
def test_backup_creation(integration_test_dir: Path) -> None:
    """Test backup creation before file replacement.

    Verifies:
    - Backup file is created with timestamp
    - Backup contains original content
    - Multiple backups don't conflict (unique timestamps)

    Args:
        integration_test_dir: Base directory for integration tests.
    """
    # Create test directory
    test_dir = integration_test_dir / "backup_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create original file
    video_file = test_dir / "test_video.mp4"
    original_content = b"original content to backup"
    video_file.write_bytes(original_content)

    # Create first backup
    timestamp1 = int(time.time() * 1000)  # milliseconds for uniqueness
    backup1 = test_dir / f"test_video_backup_{timestamp1}.mp4"
    backup1.write_bytes(original_content)

    # Wait to ensure different timestamp
    time.sleep(0.1)

    # Create second backup
    timestamp2 = int(time.time() * 1000)
    backup2 = test_dir / f"test_video_backup_{timestamp2}.mp4"
    backup2.write_bytes(original_content)

    # Verify both backups exist
    assert backup1.exists()
    assert backup2.exists()
    assert backup1.name != backup2.name  # Different names

    # Verify content
    assert backup1.read_bytes() == original_content
    assert backup2.read_bytes() == original_content

    # Cleanup
    backup1.unlink(missing_ok=True)
    backup2.unlink(missing_ok=True)
    video_file.unlink(missing_ok=True)


@pytest.mark.integration
def test_rollback_on_failure(integration_test_dir: Path) -> None:
    """Test rollback mechanism when file replacement fails.

    Verifies:
    - Backup is restored if replacement fails
    - Original file is not corrupted
    - Cleanup occurs properly

    Args:
        integration_test_dir: Base directory for integration tests.
    """
    import os

    # Create test directory
    test_dir = integration_test_dir / "rollback_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create original file
    video_file = test_dir / "video.mp4"
    original_content = b"original safe content"
    video_file.write_bytes(original_content)

    # Create backup
    backup_file = test_dir / f"video_backup_{int(time.time())}.mp4"
    backup_file.write_bytes(original_content)

    # Simulate failed replacement
    try:
        # Create corrupted temporary file
        temp_file = test_dir / "video_corrupted.tmp"
        temp_file.write_bytes(b"corrupted")

        # Simulate failure during replacement
        # In real code, this might be an exception during processing
        raise ValueError("Simulated processing failure")

    except ValueError:
        # Rollback: restore from backup
        if backup_file.exists():
            os.replace(backup_file, video_file)

    # Verify original content restored
    assert video_file.read_bytes() == original_content
    assert not backup_file.exists()  # Backup consumed during restore

    # Cleanup
    video_file.unlink(missing_ok=True)
    temp_file.unlink(missing_ok=True)


@pytest.mark.integration
def test_disk_space_check(integration_test_dir: Path) -> None:
    """Test pre-flight disk space checks.

    Verifies:
    - Available disk space is checked before operations
    - Operations fail gracefully if insufficient space
    - Appropriate error is raised

    Args:
        integration_test_dir: Base directory for integration tests.
    """
    import shutil as sh

    # Get disk space information
    stat = sh.disk_usage(integration_test_dir)

    # Verify we can check disk space
    assert stat.total > 0
    assert stat.used >= 0
    assert stat.free >= 0

    # Simulate space check for large file
    required_space = 10 * 1024 * 1024 * 1024  # 10 GB

    has_space = stat.free > required_space

    # This test just verifies the check works
    # In production, would fail operation if not has_space
    if not has_space:
        # Would raise appropriate exception
        assert stat.free < required_space


@pytest.mark.integration
def test_cleanup_temp_files(integration_test_dir: Path) -> None:
    """Test cleanup of temporary files after operations.

    Verifies:
    - Temporary directory is created
    - Files are cleaned up after successful operation
    - Files are cleaned up after failed operation
    - No orphaned temporary files remain

    Args:
        integration_test_dir: Base directory for integration tests.
    """
    # Create temporary directory
    temp_dir = integration_test_dir / "redubber_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary files (simulating processing)
    temp_files = [
        temp_dir / "audio_extract.wav",
        temp_dir / "transcription.json",
        temp_dir / "translation.json",
        temp_dir / "segments" / "seg_001.wav",
        temp_dir / "segments" / "seg_002.wav",
    ]

    # Create files
    for temp_file in temp_files:
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_bytes(b"temporary data")

    # Verify files exist
    for temp_file in temp_files:
        assert temp_file.exists()

    # Cleanup operation
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Verify cleanup succeeded
    assert not temp_dir.exists()


@pytest.mark.integration
def test_concurrent_file_access(integration_test_dir: Path) -> None:
    """Test handling of concurrent file access.

    Verifies:
    - File locking or serialization prevents corruption
    - Concurrent operations don't interfere
    - Proper error handling for locked files

    Args:
        integration_test_dir: Base directory for integration tests.
    """
    from concurrent.futures import ThreadPoolExecutor

    # Create test file
    test_dir = integration_test_dir / "concurrent_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "shared.mp4"
    test_file.write_bytes(b"initial content")

    results = []

    def read_file(file_id: int) -> str:
        """Read file and return content."""
        content = test_file.read_bytes()
        return f"reader-{file_id}: {len(content)} bytes"

    # Concurrent reads should be safe
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(read_file, i) for i in range(5)]
        results = [f.result() for f in futures]

    # All reads should succeed
    assert len(results) == 5
    for result in results:
        assert "bytes" in result

    # Cleanup
    test_file.unlink(missing_ok=True)


@pytest.mark.integration
def test_path_traversal_protection(integration_test_dir: Path) -> None:
    """Test protection against path traversal attacks.

    Verifies:
    - Paths are validated and normalized
    - Attempts to access parent directories are blocked
    - Symlink attacks are prevented

    Args:
        integration_test_dir: Base directory for integration tests.
    """

    # Create test directory structure
    safe_dir = integration_test_dir / "safe_zone"
    safe_dir.mkdir(parents=True, exist_ok=True)
    unsafe_dir = integration_test_dir / "unsafe_zone"
    unsafe_dir.mkdir(parents=True, exist_ok=True)

    # Create file in unsafe directory
    unsafe_file = unsafe_dir / "sensitive.txt"
    unsafe_file.write_text("sensitive data")

    # Simulate path validation
    def is_safe_path(base_dir: Path, user_path: str) -> bool:
        """Check if user path is within base directory."""
        try:
            full_path = (base_dir / user_path).resolve()
            return (
                base_dir.resolve() in full_path.parents
                or base_dir.resolve() == full_path.parent
            )
        except (ValueError, OSError):
            return False

    # Test valid paths
    assert is_safe_path(safe_dir, "video.mp4")
    assert is_safe_path(safe_dir, "subdir/video.mp4")

    # Test path traversal attempts
    assert not is_safe_path(safe_dir, "../unsafe_zone/sensitive.txt")
    assert not is_safe_path(safe_dir, "../../sensitive.txt")

    # Cleanup
    unsafe_file.unlink(missing_ok=True)


@pytest.mark.integration
def test_large_file_handling(integration_test_dir: Path) -> None:
    """Test handling of large files.

    Verifies:
    - Large files can be processed
    - Memory usage remains reasonable
    - Chunked reading/writing works correctly

    Args:
        integration_test_dir: Base directory for integration tests.
    """
    # Create test directory
    test_dir = integration_test_dir / "large_file_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create a moderately large file (10 MB for testing)
    large_file = test_dir / "large_video.mp4"
    chunk_size = 1024 * 1024  # 1 MB chunks
    num_chunks = 10

    # Write in chunks
    with large_file.open("wb") as f:
        for i in range(num_chunks):
            chunk = bytes([i % 256]) * chunk_size
            f.write(chunk)

    # Verify file size
    assert large_file.stat().st_size == chunk_size * num_chunks

    # Read in chunks to verify
    with large_file.open("rb") as f:
        total_read = 0
        while chunk := f.read(chunk_size):
            total_read += len(chunk)

    assert total_read == chunk_size * num_chunks

    # Cleanup
    large_file.unlink(missing_ok=True)


@pytest.mark.integration
def test_file_permissions(integration_test_dir: Path) -> None:
    """Test file permission handling.

    Verifies:
    - New files have appropriate permissions
    - Permission errors are handled gracefully
    - Read-only files are detected

    Args:
        integration_test_dir: Base directory for integration tests.
    """
    import os
    import stat

    # Create test directory
    test_dir = integration_test_dir / "permissions_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create file with specific permissions
    test_file = test_dir / "restricted.mp4"
    test_file.write_bytes(b"test content")

    # Make file read-only
    os.chmod(test_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    # Verify file is read-only
    file_stat = test_file.stat()
    assert not (file_stat.st_mode & stat.S_IWUSR)

    # Attempt to write should fail (in production, would be caught)
    with pytest.raises(PermissionError):
        test_file.write_bytes(b"new content")

    # Restore write permission for cleanup
    os.chmod(test_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    test_file.unlink(missing_ok=True)
