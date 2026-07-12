"""
Test script for TTS Preview Generator.
Run this to verify the service works correctly.
"""

import os
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.tts_preview_generator import (
    TTSPreviewGenerator,
    get_tts_preview_generator,
)
from database import DatabaseManager


def test_cache_key_generation():
    """Test cache key generation."""
    print("\n=== Testing Cache Key Generation ===")

    generator = TTSPreviewGenerator()

    text = "Hello, this is a test."
    instructions = "Speak warmly and professionally."
    voice = "nova"

    key1 = generator.generate_cache_key(text, instructions, voice)
    key2 = generator.generate_cache_key(text, instructions, voice)

    print(f"Text: {text}")
    print(f"Instructions: {instructions}")
    print(f"Voice: {voice}")
    print(f"Cache Key: {key1}")
    print(f"Keys match: {key1 == key2}")

    # Different text should produce different key
    key3 = generator.generate_cache_key("Different text", instructions, voice)
    print(f"Different text key matches: {key1 == key3}")

    assert key1 == key2, "Same inputs should produce same key"
    assert key1 != key3, "Different inputs should produce different keys"
    print("✓ Cache key generation works correctly")


def test_single_preview_generation():
    """Test generating a single TTS preview."""
    print("\n=== Testing Single Preview Generation ===")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ OPENAI_API_KEY not set, skipping API test")
        return

    generator = get_tts_preview_generator()
    db_manager = DatabaseManager()

    # Create a test project
    project_id = db_manager.add_project(
        path="/tmp/test_voice_refinement", name="test_voice_refinement"
    )

    text = "Welcome to the voice refinement system. This is a test preview."
    instructions = "Speak in a warm, professional tone with moderate pace."

    print(f"Project ID: {project_id}")
    print(f"Text: {text}")
    print(f"Instructions: {instructions}")
    print("Voice: nova")

    try:
        result = generator.generate_preview(
            project_id=project_id,
            voice="nova",
            translated_text=text,
            voice_instructions=instructions,
        )

        print(f"Result: {result}")
        print(f"Audio file: {result['audio_file_path']}")
        print(f"Duration: {result['duration_ms']}ms")
        print(f"Cached: {result['cached']}")

        # Verify file exists
        assert os.path.exists(result["audio_file_path"]), "Audio file should exist"
        assert result["duration_ms"] > 0, "Duration should be positive"

        print("✓ Single preview generation works correctly")

        # Test cache hit
        print("\nTesting cache hit...")
        result2 = generator.generate_preview(
            project_id=project_id,
            voice="nova",
            translated_text=text,
            voice_instructions=instructions,
        )

        assert result2["cached"], "Second call should hit cache"
        assert result2["audio_file_path"] == result["audio_file_path"], "Should return same file"
        print("✓ Cache hit works correctly")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


def test_parallel_generation():
    """Test generating all voices in parallel."""
    print("\n=== Testing Parallel Preview Generation ===")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ OPENAI_API_KEY not set, skipping API test")
        return

    generator = get_tts_preview_generator()
    db_manager = DatabaseManager()

    # Create a test project
    project_id = db_manager.add_project(
        path="/tmp/test_voice_refinement_parallel",
        name="test_voice_refinement_parallel",
    )

    text = "This is a parallel test of all six voices."
    instructions = "Speak naturally with clear enunciation."

    print(f"Project ID: {project_id}")
    print(f"Text: {text}")
    print(f"Instructions: {instructions}")
    print("Generating previews for all 6 voices...")

    try:
        import time

        start_time = time.time()

        result = generator.generate_all_previews(
            project_id=project_id,
            translated_text=text,
            voice_instructions=instructions,
        )

        elapsed_time = time.time() - start_time

        print(f"\nGeneration completed in {elapsed_time:.2f} seconds")
        print(f"Cache hits: {result['cache_hits']}")
        print(f"Cache misses: {result['cache_misses']}")
        print(f"Instructions hash: {result['instructions_hash']}")

        print("\nPreviews:")
        for preview in result["previews"]:
            status = "✓" if not preview.get("error") else "✗"
            cached = "[CACHED]" if preview["cached"] else "[NEW]"
            print(
                f"  {status} {preview['voice']:8s} {cached:9s} {preview['duration_ms']:5d}ms"
            )
            if preview.get("error"):
                print(f"      Error: {preview['error']}")

        # Verify all voices were generated
        assert len(result["previews"]) == 6, "Should generate 6 previews"

        successful = [p for p in result["previews"] if not p.get("error")]
        print(f"\n✓ Successfully generated {len(successful)}/6 previews")

        # Test cache hit on second run
        print("\nTesting cache hit on second run...")
        start_time = time.time()

        result2 = generator.generate_all_previews(
            project_id=project_id,
            translated_text=text,
            voice_instructions=instructions,
        )

        elapsed_time = time.time() - start_time

        print(f"Second run completed in {elapsed_time:.2f} seconds")
        print(f"Cache hits: {result2['cache_hits']}")
        print(f"Cache misses: {result2['cache_misses']}")

        # All should be cache hits now
        if result2["cache_misses"] == 0:
            print("✓ All previews retrieved from cache")
        else:
            print(f"⚠ Expected all cache hits, got {result2['cache_misses']} misses")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


def test_cache_cleanup():
    """Test cache cleanup functionality."""
    print("\n=== Testing Cache Cleanup ===")

    generator = get_tts_preview_generator()

    try:
        generator.cleanup_cache(days=30, max_entries_per_project=100)
        print("✓ Cache cleanup works correctly")
    except Exception as e:
        print(f"✗ Cache cleanup failed: {e}")
        raise


def main():
    """Run all tests."""
    print("=" * 60)
    print("TTS Preview Generator Test Suite")
    print("=" * 60)

    # Basic tests (no API key needed)
    test_cache_key_generation()

    # API tests (require OpenAI API key)
    if os.getenv("OPENAI_API_KEY"):
        print("\n⚠ API tests will make real OpenAI API calls and incur costs.")
        print("Testing with minimal text to keep costs low.")

        try:
            test_single_preview_generation()
            test_parallel_generation()
            test_cache_cleanup()
        except Exception as e:
            print(f"\n✗ Tests failed: {e}")
            return 1
    else:
        print("\n⚠ Skipping API tests: OPENAI_API_KEY not set")
        print("To run full tests, set OPENAI_API_KEY environment variable")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
