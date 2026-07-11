"""
Example usage of TTS Preview Generator Service.

This demonstrates how to use the TTS Preview Generator in a typical
voice refinement workflow.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.tts_preview_generator import get_tts_preview_generator
from app.services.voice_instruction_generator import get_voice_instruction_generator
from database import DatabaseManager


def example_complete_workflow():
    """
    Complete voice refinement workflow example.

    Steps:
    1. Create a test project
    2. Generate voice instructions using LLM
    3. Generate TTS previews for all voices
    4. Simulate user selection
    5. Save voice settings to project
    """
    print("=" * 70)
    print("Voice Refinement System - Complete Workflow Example")
    print("=" * 70)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it before running this example:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return

    # Initialize services
    print("\n1. Initializing services...")
    db_manager = DatabaseManager()
    tts_generator = get_tts_preview_generator(db_manager=db_manager)
    instruction_generator = get_voice_instruction_generator()
    print("   ✓ Services initialized")

    # Create test project
    print("\n2. Creating test project...")
    project_id = db_manager.add_project(
        path="/tmp/voice_refinement_example", name="voice_refinement_example"
    )
    print(f"   ✓ Project created (ID: {project_id})")

    # Sample transcription segment
    original_text = """
    Welcome to this comprehensive tutorial on artificial intelligence.
    Today, we'll explore the fascinating world of machine learning,
    neural networks, and their real-world applications.
    """

    translated_text = """
    欢迎来到这个关于人工智能的综合教程。
    今天，我们将探索机器学习、神经网络及其实际应用的迷人世界。
    """

    segment_id = "video1_segment_0"

    print("\n3. Analyzing transcription segment...")
    print(f"   Original: {original_text.strip()[:60]}...")
    print(f"   Translated: {translated_text.strip()[:60]}...")

    # Generate voice instructions using LLM
    print("\n4. Generating voice instructions with LLM...")
    instruction_result = instruction_generator.generate_instructions(
        original_text=original_text.strip(),
        translated_text=translated_text.strip(),
        context={
            "content_type": "educational",
            "speaker_gender": "neutral",
            "speaker_age": "adult",
        },
    )

    voice_instructions = instruction_result["voice_instructions"]
    characteristics = instruction_result["detected_characteristics"]

    print("   ✓ Instructions generated:")
    print(f"     Tone: {characteristics.get('tone', 'N/A')}")
    print(f"     Pace: {characteristics.get('pace', 'N/A')}")
    print(f"     Emotion: {characteristics.get('emotion', 'N/A')}")
    print(f"     Style: {characteristics.get('style', 'N/A')}")
    print(f"\n   Full instructions: {voice_instructions}")

    # Save instruction generation to database
    generation_id = db_manager.save_voice_instruction_generation(
        project_id=project_id,
        segment_id=segment_id,
        original_text=original_text.strip(),
        translated_text=translated_text.strip(),
        voice_instructions=voice_instructions,
        llm_model=instruction_result["llm_model"],
        detected_characteristics=characteristics,
    )
    print(f"   ✓ Saved to database (Generation ID: {generation_id})")

    # Generate TTS previews for all voices
    print("\n5. Generating TTS previews for all 6 voices...")
    print("   (This will make API calls and may take a few seconds)")

    import time

    start_time = time.time()

    preview_result = tts_generator.generate_all_previews(
        project_id=project_id,
        translated_text=translated_text.strip(),
        voice_instructions=voice_instructions,
    )

    elapsed_time = time.time() - start_time

    print(f"\n   ✓ Generation completed in {elapsed_time:.2f} seconds")
    print(f"   Cache hits: {preview_result['cache_hits']}")
    print(f"   Cache misses: {preview_result['cache_misses']}")
    print(f"   Instructions hash: {preview_result['instructions_hash']}")

    # Display preview results
    print("\n   Preview Results:")
    print("   " + "-" * 66)
    print(f"   {'Voice':<12} {'Status':<10} {'Duration':<12} {'File Path':<30}")
    print("   " + "-" * 66)

    for preview in preview_result["previews"]:
        voice = preview["voice"]
        cached = "[CACHED]" if preview["cached"] else "[NEW]"
        duration = f"{preview['duration_ms']}ms"
        file_path = (
            Path(preview["audio_file_path"]).name if preview.get("audio_file_path") else "N/A"
        )

        if preview.get("error"):
            print(f"   {voice:<12} {'[ERROR]':<10} {'N/A':<12} {preview['error']}")
        else:
            print(f"   {voice:<12} {cached:<10} {duration:<12} {file_path}")

    print("   " + "-" * 66)

    # Simulate user selection
    selected_voice = "nova"
    print(f"\n6. User selects voice: {selected_voice}")

    # Save voice settings to project
    print("\n7. Saving voice settings to project...")
    db_manager.set_voice_settings(
        project_id=project_id, voice=selected_voice, voice_instructions=voice_instructions
    )

    # Save selection history
    history_id = db_manager.save_voice_selection(
        project_id=project_id,
        voice_name=selected_voice,
        voice_instructions=voice_instructions,
        segment_used=segment_id,
    )

    print(f"   ✓ Voice settings saved (History ID: {history_id})")

    # Verify settings were saved
    voice_settings = db_manager.get_voice_settings(project_id)
    print("\n8. Verifying saved settings...")
    print(f"   Voice: {voice_settings['voice']}")
    print(f"   Instructions: {voice_settings['voice_instructions'][:60]}...")
    print("   ✓ Settings verified")

    # Test cache hit on subsequent generation
    print("\n9. Testing cache efficiency...")
    print("   Regenerating same previews to test cache...")

    start_time = time.time()

    cache_test_result = tts_generator.generate_all_previews(
        project_id=project_id,
        translated_text=translated_text.strip(),
        voice_instructions=voice_instructions,
    )

    elapsed_time = time.time() - start_time

    print(f"   ✓ Second generation completed in {elapsed_time:.2f} seconds")
    print(f"   Cache hits: {cache_test_result['cache_hits']}")
    print(f"   Cache misses: {cache_test_result['cache_misses']}")

    if cache_test_result["cache_misses"] == 0:
        speedup = (
            preview_result["cache_misses"] * 2.0 / elapsed_time
            if elapsed_time > 0
            else float("inf")
        )
        print(f"   🎉 100% cache hit rate! (~{speedup:.0f}x faster)")
    else:
        print(f"   ⚠ Expected all cache hits, got {cache_test_result['cache_misses']} misses")

    print("\n" + "=" * 70)
    print("Workflow completed successfully!")
    print("=" * 70)

    # Cleanup suggestion
    print("\nTo clean up test data:")
    print(f"  - Database: Delete project ID {project_id}")
    print(f"  - Cache files: See {tts_generator.cache_dir}/")


def example_single_voice_generation():
    """
    Simple example: Generate preview for a single voice.
    """
    print("\n" + "=" * 70)
    print("Single Voice Generation Example")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ ERROR: OPENAI_API_KEY not set")
        return

    # Initialize
    db_manager = DatabaseManager()
    tts_generator = get_tts_preview_generator(db_manager=db_manager)

    # Create test project
    project_id = db_manager.add_project(
        path="/tmp/single_voice_test", name="single_voice_test"
    )

    # Generate single preview
    print("\nGenerating preview for 'nova' voice...")
    result = tts_generator.generate_preview(
        project_id=project_id,
        voice="nova",
        translated_text="This is a simple test of voice preview generation.",
        voice_instructions="Speak in a warm, friendly tone with moderate pace.",
    )

    print("✓ Generation complete:")
    print(f"  Voice: {result['voice']}")
    print(f"  File: {result['audio_file_path']}")
    print(f"  Duration: {result['duration_ms']}ms")
    print(f"  Cached: {result['cached']}")


def example_custom_voices():
    """
    Example: Generate previews for a subset of voices.
    """
    print("\n" + "=" * 70)
    print("Custom Voice Selection Example")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ ERROR: OPENAI_API_KEY not set")
        return

    # Initialize
    db_manager = DatabaseManager()
    tts_generator = get_tts_preview_generator(db_manager=db_manager)

    # Create test project
    project_id = db_manager.add_project(
        path="/tmp/custom_voices_test", name="custom_voices_test"
    )

    # Generate only female voices
    print("\nGenerating previews for female voices only (nova, shimmer)...")
    result = tts_generator.generate_all_previews(
        project_id=project_id,
        translated_text="Testing female voices for this project.",
        voice_instructions="Speak naturally with clear enunciation.",
        voices=["nova", "shimmer"],  # Only these two voices
    )

    print(f"✓ Generated {len(result['previews'])} previews:")
    for preview in result["previews"]:
        print(f"  - {preview['voice']}: {preview['duration_ms']}ms")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("TTS Preview Generator - Usage Examples")
    print("=" * 70)
    print("\nThese examples demonstrate how to use the TTS Preview Generator")
    print("in various scenarios within the Voice Refinement System.")

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ ERROR: OPENAI_API_KEY environment variable not set")
        print("\nPlease set it before running these examples:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nNote: These examples make real API calls and will incur costs.")
        return 1

    print("\n⚠ WARNING: These examples make real OpenAI API calls.")
    print("Estimated cost: ~$0.05 for all examples")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return 0

    # Run examples
    try:
        example_single_voice_generation()
        example_custom_voices()
        example_complete_workflow()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\n✗ Example failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
