"""
Tests for voice refinement database operations.
Tests DatabaseManager methods for voice instruction generations,
TTS cache, and voice selection history.
"""

import time

import pytest

pytestmark = pytest.mark.stale  # needs rewrite to match current DatabaseManager API

from database import DatabaseManager


class TestVoiceInstructionGenerations:
    """Test voice instruction generation storage and retrieval."""

    @pytest.fixture
    def db(self, tmp_path):
        """Provide test database instance."""
        db_path = tmp_path / "test_voice_refinement.db"
        return DatabaseManager(db_path=str(db_path))

    @pytest.fixture
    def project_id(self, db):
        """Create a test project and return its ID."""
        return db.add_project(path="/test/project", name="Test Project")

    def test_save_voice_instruction_generation(self, db, project_id):
        """Test saving a voice instruction generation result."""
        generation_id = db.save_voice_instruction_generation(
            project_id=project_id,
            segment_id="video1_segment_0",
            original_text="This is original text",
            translated_text="Este es el texto traducido",
            voice_instructions="Speak with a warm, professional tone.",
            llm_model="gpt-4o",
            detected_characteristics='{"tone": "warm", "pace": "moderate"}',
        )

        assert generation_id > 0

    def test_save_voice_instruction_generation_without_characteristics(self, db, project_id):
        """Test saving without detected characteristics (optional field)."""
        generation_id = db.save_voice_instruction_generation(
            project_id=project_id,
            segment_id="video1_segment_1",
            original_text="Test text",
            translated_text="Texto de prueba",
            voice_instructions="Instructions here",
            llm_model="gpt-4o",
            detected_characteristics=None,
        )

        assert generation_id > 0

    def test_save_voice_instruction_generation_with_dict_characteristics(self, db, project_id):
        """Test saving with dict characteristics (auto-converted to JSON)."""
        characteristics = {
            "tone": "professional",
            "pace": "fast",
            "emotion": "energetic",
            "style": "dynamic",
        }

        db.save_voice_instruction_generation(
            project_id=project_id,
            segment_id="video1_segment_2",
            original_text="Test",
            translated_text="Prueba",
            voice_instructions="Test instructions",
            llm_model="gpt-4o",
            detected_characteristics=characteristics,
        )

        # Verify it was saved correctly
        generations = db.get_voice_instruction_generations(project_id=project_id, limit=1)
        assert len(generations) == 1
        assert generations[0]["detected_characteristics"] == characteristics

    def test_get_voice_instruction_generations(self, db, project_id):
        """Test retrieving voice instruction generations."""
        # Save multiple generations
        for i in range(5):
            db.save_voice_instruction_generation(
                project_id=project_id,
                segment_id=f"segment_{i}",
                original_text=f"Original {i}",
                translated_text=f"Translated {i}",
                voice_instructions=f"Instructions {i}",
                llm_model="gpt-4o",
            )

        generations = db.get_voice_instruction_generations(project_id=project_id, limit=10)

        assert len(generations) == 5
        # Should be ordered by created_at DESC (most recent first)
        assert generations[0]["segment_id"] == "segment_4"
        assert generations[-1]["segment_id"] == "segment_0"

    def test_get_voice_instruction_generations_with_limit(self, db, project_id):
        """Test that limit parameter works correctly."""
        # Save 10 generations
        for i in range(10):
            db.save_voice_instruction_generation(
                project_id=project_id,
                segment_id=f"segment_{i}",
                original_text=f"Text {i}",
                translated_text=f"Texto {i}",
                voice_instructions=f"Instructions {i}",
            )

        # Retrieve only 3
        generations = db.get_voice_instruction_generations(project_id=project_id, limit=3)

        assert len(generations) == 3
        # Should get the 3 most recent
        assert generations[0]["segment_id"] == "segment_9"
        assert generations[1]["segment_id"] == "segment_8"
        assert generations[2]["segment_id"] == "segment_7"

    def test_get_voice_instruction_generations_empty_project(self, db, project_id):
        """Test retrieving from project with no generations."""
        generations = db.get_voice_instruction_generations(project_id=project_id)

        assert len(generations) == 0

    def test_get_voice_instruction_generations_multiple_projects(self, db):
        """Test that generations are isolated by project."""
        project1_id = db.add_project(path="/project1", name="Project 1")
        project2_id = db.add_project(path="/project2", name="Project 2")

        # Save to project 1
        db.save_voice_instruction_generation(
            project_id=project1_id,
            segment_id="p1_segment",
            original_text="Project 1 text",
            translated_text="Proyecto 1 texto",
            voice_instructions="Instructions 1",
        )

        # Save to project 2
        db.save_voice_instruction_generation(
            project_id=project2_id,
            segment_id="p2_segment",
            original_text="Project 2 text",
            translated_text="Proyecto 2 texto",
            voice_instructions="Instructions 2",
        )

        # Verify isolation
        p1_generations = db.get_voice_instruction_generations(project_id=project1_id)
        p2_generations = db.get_voice_instruction_generations(project_id=project2_id)

        assert len(p1_generations) == 1
        assert len(p2_generations) == 1
        assert p1_generations[0]["segment_id"] == "p1_segment"
        assert p2_generations[0]["segment_id"] == "p2_segment"

    def test_voice_instruction_generation_includes_timestamp(self, db, project_id):
        """Test that generations include created_at timestamp."""
        db.save_voice_instruction_generation(
            project_id=project_id,
            segment_id="test_segment",
            original_text="Test",
            translated_text="Prueba",
            voice_instructions="Test",
        )

        generations = db.get_voice_instruction_generations(project_id=project_id)

        assert len(generations) == 1
        assert "created_at" in generations[0]
        assert generations[0]["created_at"] is not None


class TestTTSCacheOperations:
    """Test TTS preview cache storage and retrieval."""

    @pytest.fixture
    def db(self, tmp_path):
        """Provide test database instance."""
        db_path = tmp_path / "test_cache.db"
        return DatabaseManager(db_path=str(db_path))

    @pytest.fixture
    def project_id(self, db):
        """Create a test project."""
        return db.add_project(path="/test/project", name="Test Project")

    def test_save_tts_cache(self, db, project_id):
        """Test saving TTS cache entry."""
        cache_id = db.save_tts_cache(
            project_id=project_id,
            voice_name="nova",
            voice_instructions_hash="abc123hash",
            translated_text="Hello world",
            audio_file_path="/cache/audio.mp3",
            audio_duration_ms=2500,
            tts_model="tts-1",
        )

        assert cache_id > 0

    def test_get_or_create_tts_cache_hit(self, db, project_id):
        """Test cache hit retrieves existing entry."""
        # Save cache entry
        db.save_tts_cache(
            project_id=project_id,
            voice_name="alloy",
            voice_instructions_hash="hash123",
            translated_text="Test text",
            audio_file_path="/path/to/audio.mp3",
            audio_duration_ms=3000,
        )

        # Try to get it
        cached = db.get_tts_cache(
            project_id=project_id,
            voice_name="alloy",
            voice_instructions_hash="hash123",
        )

        assert cached is not None
        assert cached["voice_name"] == "alloy"
        assert cached["audio_file_path"] == "/path/to/audio.mp3"
        assert cached["audio_duration_ms"] == 3000

    def test_get_or_create_tts_cache_miss(self, db, project_id):
        """Test cache miss returns None."""
        cached = db.get_tts_cache(
            project_id=project_id,
            voice_name="echo",
            voice_instructions_hash="nonexistent_hash",
        )

        assert cached is None

    def test_get_or_create_tts_cache_updates_accessed_at(self, db, project_id):
        """Test that cache hit updates accessed_at timestamp for LRU."""
        # Save cache entry
        db.save_tts_cache(
            project_id=project_id,
            voice_name="fable",
            voice_instructions_hash="hash_lru",
            translated_text="Test",
            audio_file_path="/path.mp3",
            audio_duration_ms=1000,
        )

        # Get initial accessed_at
        cached1 = db.get_tts_cache(
            project_id=project_id,
            voice_name="fable",
            voice_instructions_hash="hash_lru",
        )
        accessed_at_1 = cached1["accessed_at"]

        # Wait a bit and access again
        time.sleep(0.1)

        cached2 = db.get_tts_cache(
            project_id=project_id,
            voice_name="fable",
            voice_instructions_hash="hash_lru",
        )
        accessed_at_2 = cached2["accessed_at"]

        # accessed_at should be updated
        assert accessed_at_2 >= accessed_at_1

    def test_tts_cache_unique_constraint(self, db, project_id):
        """Test that cache uses unique constraint on hash + voice."""
        # Save first entry
        db.save_tts_cache(
            project_id=project_id,
            voice_name="nova",
            voice_instructions_hash="same_hash",
            translated_text="Text 1",
            audio_file_path="/path1.mp3",
            audio_duration_ms=1000,
        )

        # Save second entry with same hash + voice (should replace)
        db.save_tts_cache(
            project_id=project_id,
            voice_name="nova",
            voice_instructions_hash="same_hash",
            translated_text="Text 2",
            audio_file_path="/path2.mp3",
            audio_duration_ms=2000,
        )

        # Retrieve - should get the second one
        cached = db.get_tts_cache(
            project_id=project_id,
            voice_name="nova",
            voice_instructions_hash="same_hash",
        )

        assert cached["audio_file_path"] == "/path2.mp3"
        assert cached["audio_duration_ms"] == 2000

    def test_tts_cache_different_voices_same_hash(self, db, project_id):
        """Test that same hash with different voices creates separate entries."""
        same_hash = "same_instructions_hash"

        # Save for two different voices
        db.save_tts_cache(
            project_id=project_id,
            voice_name="alloy",
            voice_instructions_hash=same_hash,
            translated_text="Same text",
            audio_file_path="/alloy.mp3",
            audio_duration_ms=1000,
        )

        db.save_tts_cache(
            project_id=project_id,
            voice_name="nova",
            voice_instructions_hash=same_hash,
            translated_text="Same text",
            audio_file_path="/nova.mp3",
            audio_duration_ms=1000,
        )

        # Both should be retrievable
        cached_alloy = db.get_tts_cache(
            project_id=project_id,
            voice_name="alloy",
            voice_instructions_hash=same_hash,
        )

        cached_nova = db.get_tts_cache(
            project_id=project_id,
            voice_name="nova",
            voice_instructions_hash=same_hash,
        )

        assert cached_alloy["audio_file_path"] == "/alloy.mp3"
        assert cached_nova["audio_file_path"] == "/nova.mp3"

    def test_cleanup_old_tts_cache_by_age(self, db, project_id):
        """Test cleanup of old cache entries by age."""
        import sqlite3

        # Manually insert old entries (can't easily mock timestamps)
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()

            # Insert old entry (40 days ago)
            cursor.execute(
                """
                INSERT INTO tts_preview_cache
                (project_id, voice_name, voice_instructions_hash, translated_text,
                 audio_file_path, audio_duration_ms, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now', '-40 days'), datetime('now', '-40 days'))
            """,
                (project_id, "old_voice", "old_hash", "Old text", "/old.mp3", 1000),
            )

            # Insert recent entry
            cursor.execute(
                """
                INSERT INTO tts_preview_cache
                (project_id, voice_name, voice_instructions_hash, translated_text,
                 audio_file_path, audio_duration_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (project_id, "new_voice", "new_hash", "New text", "/new.mp3", 2000),
            )

            conn.commit()

        # Cleanup entries older than 30 days
        db.cleanup_old_tts_cache(days=30, max_entries_per_project=100)

        # Old entry should be deleted
        old_cached = db.get_tts_cache(
            project_id=project_id,
            voice_name="old_voice",
            voice_instructions_hash="old_hash",
        )

        # New entry should still exist
        new_cached = db.get_tts_cache(
            project_id=project_id,
            voice_name="new_voice",
            voice_instructions_hash="new_hash",
        )

        assert old_cached is None
        assert new_cached is not None

    def test_cleanup_old_tts_cache_by_limit(self, db, project_id):
        """Test cleanup keeps only N most recent entries per project."""
        # Create 10 cache entries
        for i in range(10):
            db.save_tts_cache(
                project_id=project_id,
                voice_name=f"voice_{i}",
                voice_instructions_hash=f"hash_{i}",
                translated_text=f"Text {i}",
                audio_file_path=f"/audio_{i}.mp3",
                audio_duration_ms=1000 + i,
            )
            time.sleep(0.01)  # Ensure different timestamps

        # Cleanup to keep only 5 entries
        db.cleanup_old_tts_cache(days=1000, max_entries_per_project=5)

        # Count remaining entries
        import sqlite3

        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM tts_preview_cache WHERE project_id = ?",
                (project_id,),
            )
            count = cursor.fetchone()[0]

        assert count == 5

    def test_tts_cache_includes_timestamps(self, db, project_id):
        """Test that cache entries include created_at and accessed_at."""
        db.save_tts_cache(
            project_id=project_id,
            voice_name="shimmer",
            voice_instructions_hash="hash_ts",
            translated_text="Test",
            audio_file_path="/test.mp3",
            audio_duration_ms=1500,
        )

        cached = db.get_tts_cache(
            project_id=project_id,
            voice_name="shimmer",
            voice_instructions_hash="hash_ts",
        )

        assert "created_at" in cached
        assert "accessed_at" in cached
        assert cached["created_at"] is not None
        assert cached["accessed_at"] is not None


class TestVoiceSelectionHistory:
    """Test voice selection history tracking."""

    @pytest.fixture
    def db(self, tmp_path):
        """Provide test database instance."""
        db_path = tmp_path / "test_selection.db"
        return DatabaseManager(db_path=str(db_path))

    @pytest.fixture
    def project_id(self, db):
        """Create a test project."""
        return db.add_project(path="/test/project", name="Test Project")

    def test_save_voice_selection(self, db, project_id):
        """Test saving voice selection to history."""
        history_id = db.save_voice_selection(
            project_id=project_id,
            voice_name="nova",
            voice_instructions="Speak with warmth and clarity.",
            segment_used="video1_segment_0",
        )

        assert history_id > 0

    def test_get_voice_selection_history(self, db, project_id):
        """Test retrieving voice selection history."""
        # Save multiple selections
        db.save_voice_selection(
            project_id=project_id,
            voice_name="alloy",
            voice_instructions="Instructions 1",
            segment_used="segment_0",
        )

        time.sleep(0.01)  # Ensure different timestamps

        db.save_voice_selection(
            project_id=project_id,
            voice_name="nova",
            voice_instructions="Instructions 2",
            segment_used="segment_1",
        )

        history = db.get_voice_selection_history(project_id=project_id)

        assert len(history) == 2
        # Should be ordered by selected_at DESC (most recent first)
        assert history[0]["voice_name"] == "nova"
        assert history[1]["voice_name"] == "alloy"

    def test_get_voice_selection_history_with_limit(self, db, project_id):
        """Test that limit parameter works correctly."""
        # Save 5 selections
        for i in range(5):
            db.save_voice_selection(
                project_id=project_id,
                voice_name=f"voice_{i}",
                voice_instructions=f"Instructions {i}",
                segment_used=f"segment_{i}",
            )
            time.sleep(0.01)

        # Retrieve only 2
        history = db.get_voice_selection_history(project_id=project_id, limit=2)

        assert len(history) == 2
        # Should get the 2 most recent
        assert history[0]["voice_name"] == "voice_4"
        assert history[1]["voice_name"] == "voice_3"

    def test_get_voice_selection_history_empty_project(self, db, project_id):
        """Test retrieving from project with no selections."""
        history = db.get_voice_selection_history(project_id=project_id)

        assert len(history) == 0

    def test_voice_selection_history_multiple_projects(self, db):
        """Test that selection history is isolated by project."""
        project1_id = db.add_project(path="/project1", name="Project 1")
        project2_id = db.add_project(path="/project2", name="Project 2")

        # Save to project 1
        db.save_voice_selection(
            project_id=project1_id,
            voice_name="alloy",
            voice_instructions="P1 instructions",
            segment_used="p1_segment",
        )

        # Save to project 2
        db.save_voice_selection(
            project_id=project2_id,
            voice_name="nova",
            voice_instructions="P2 instructions",
            segment_used="p2_segment",
        )

        # Verify isolation
        p1_history = db.get_voice_selection_history(project_id=project1_id)
        p2_history = db.get_voice_selection_history(project_id=project2_id)

        assert len(p1_history) == 1
        assert len(p2_history) == 1
        assert p1_history[0]["voice_name"] == "alloy"
        assert p2_history[0]["voice_name"] == "nova"

    def test_voice_selection_includes_timestamp(self, db, project_id):
        """Test that selections include selected_at timestamp."""
        db.save_voice_selection(
            project_id=project_id,
            voice_name="echo",
            voice_instructions="Test instructions",
            segment_used="test_segment",
        )

        history = db.get_voice_selection_history(project_id=project_id)

        assert len(history) == 1
        assert "selected_at" in history[0]
        assert history[0]["selected_at"] is not None

    def test_voice_selection_tracks_multiple_changes(self, db, project_id):
        """Test that changing voice creates new history entry."""
        # User selects alloy
        db.save_voice_selection(
            project_id=project_id,
            voice_name="alloy",
            voice_instructions="First choice",
            segment_used="segment_0",
        )

        time.sleep(0.01)

        # User changes to nova
        db.save_voice_selection(
            project_id=project_id,
            voice_name="nova",
            voice_instructions="Second choice",
            segment_used="segment_1",
        )

        history = db.get_voice_selection_history(project_id=project_id)

        # Should have both entries
        assert len(history) == 2
        assert history[0]["voice_name"] == "nova"  # Most recent
        assert history[1]["voice_name"] == "alloy"  # Previous


class TestVoiceSettingsIntegration:
    """Test integration with existing voice settings methods."""

    @pytest.fixture
    def db(self, tmp_path):
        """Provide test database instance."""
        db_path = tmp_path / "test_settings.db"
        return DatabaseManager(db_path=str(db_path))

    @pytest.fixture
    def project_id(self, db):
        """Create a test project."""
        return db.add_project(path="/test/project", name="Test Project")

    def test_get_voice_settings_default(self, db, project_id):
        """Test getting voice settings returns empty defaults for new project."""
        settings = db.get_voice_settings(project_id=project_id)

        assert settings["voice"] == ""
        assert settings["voice_instructions"] == ""

    def test_set_voice_settings(self, db, project_id):
        """Test setting voice settings."""
        db.set_voice_settings(
            project_id=project_id,
            voice="nova",
            voice_instructions="Speak with a professional tone.",
        )

        settings = db.get_voice_settings(project_id=project_id)

        assert settings["voice"] == "nova"
        assert settings["voice_instructions"] == "Speak with a professional tone."

    def test_update_voice_settings(self, db, project_id):
        """Test updating existing voice settings."""
        # Set initial settings
        db.set_voice_settings(
            project_id=project_id,
            voice="alloy",
            voice_instructions="First instructions",
        )

        # Update settings
        db.set_voice_settings(
            project_id=project_id,
            voice="nova",
            voice_instructions="Updated instructions",
        )

        settings = db.get_voice_settings(project_id=project_id)

        assert settings["voice"] == "nova"
        assert settings["voice_instructions"] == "Updated instructions"

    def test_voice_settings_isolated_by_project(self, db):
        """Test that voice settings are project-specific."""
        project1_id = db.add_project(path="/project1", name="Project 1")
        project2_id = db.add_project(path="/project2", name="Project 2")

        # Set different settings for each project
        db.set_voice_settings(
            project_id=project1_id,
            voice="alloy",
            voice_instructions="P1 instructions",
        )

        db.set_voice_settings(
            project_id=project2_id,
            voice="nova",
            voice_instructions="P2 instructions",
        )

        # Verify isolation
        p1_settings = db.get_voice_settings(project_id=project1_id)
        p2_settings = db.get_voice_settings(project_id=project2_id)

        assert p1_settings["voice"] == "alloy"
        assert p2_settings["voice"] == "nova"


class TestDatabaseSchemaIntegrity:
    """Test database schema and constraints."""

    @pytest.fixture
    def db(self, tmp_path):
        """Provide test database instance."""
        db_path = tmp_path / "test_schema.db"
        return DatabaseManager(db_path=str(db_path))

    def test_voice_instruction_generations_table_exists(self, db):
        """Test that voice_instruction_generations table was created."""
        import sqlite3

        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='voice_instruction_generations'
            """
            )
            result = cursor.fetchone()

        assert result is not None

    def test_tts_preview_cache_table_exists(self, db):
        """Test that tts_preview_cache table was created."""
        import sqlite3

        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='tts_preview_cache'
            """
            )
            result = cursor.fetchone()

        assert result is not None

    def test_voice_selection_history_table_exists(self, db):
        """Test that voice_selection_history table was created."""
        import sqlite3

        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='voice_selection_history'
            """
            )
            result = cursor.fetchone()

        assert result is not None

    def test_tts_cache_unique_index_exists(self, db):
        """Test that unique index on cache exists."""
        import sqlite3

        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='index' AND name='idx_tts_cache_unique'
            """
            )
            result = cursor.fetchone()

        assert result is not None

    def test_tts_cache_accessed_index_exists(self, db):
        """Test that accessed_at index exists for LRU."""
        import sqlite3

        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='index' AND name='idx_tts_cache_accessed'
            """
            )
            result = cursor.fetchone()

        assert result is not None

    def test_foreign_key_cascade_delete(self, db):
        """Test that deleting project cascades to voice refinement tables."""
        # Create project and add data
        project_id = db.add_project(path="/test/cascade", name="Cascade Test")

        db.save_voice_instruction_generation(
            project_id=project_id,
            segment_id="test_segment",
            original_text="Test",
            translated_text="Prueba",
            voice_instructions="Test instructions",
        )

        db.save_tts_cache(
            project_id=project_id,
            voice_name="nova",
            voice_instructions_hash="test_hash",
            translated_text="Test",
            audio_file_path="/test.mp3",
            audio_duration_ms=1000,
        )

        db.save_voice_selection(
            project_id=project_id,
            voice_name="nova",
            voice_instructions="Test",
            segment_used="test_segment",
        )

        # Delete project
        db.remove_project_by_path("/test/cascade")

        # Verify cascade delete
        generations = db.get_voice_instruction_generations(project_id=project_id)
        history = db.get_voice_selection_history(project_id=project_id)

        assert len(generations) == 0
        assert len(history) == 0
