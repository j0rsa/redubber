# Redubber - Audio Redub Project Manager

High-performance video redubbing system with FastAPI backend and React PWA frontend. Redubber v2.0 delivers 5x faster processing with async TTS and modern web architecture.

## Features

- 🎥 AI-powered video redubbing with OpenAI Whisper, GPT, and TTS
- 🚀 5x performance boost with async TTS (100 concurrent API calls)
- 📁 Project folder indexing and management
- 🎯 Automatic video file detection with language recognition
- 📝 Subtitle file detection with language identification
- 🗄️ SQLite database for fast project re-opening
- 🔄 Real-time progress tracking via WebSocket
- 🐳 Production-ready Docker deployment
- 🌐 Modern React PWA frontend with offline support

## Quick Start (Docker - Recommended)

1. **Prerequisites**: Docker Engine 20.10+ and OpenAI API key

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here
```

3. **Build and run**:
```bash
docker-compose build
docker-compose up -d
```

4. **Access the application**: http://localhost:8000

📘 **Full Docker documentation**: See [DOCKER.md](DOCKER.md) for production deployment, resource tuning, and troubleshooting.

## Development Installation

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

4. Run backend (development mode):
```bash
poetry run uvicorn app.main:app --reload
```

5. Run frontend (development mode):
```bash
cd frontend
npm run dev
```

Or use the legacy Streamlit interface:
```bash
poetry run streamlit run app.py
# or use the convenience script
./run.sh
```

## Usage

1. Enter the path to your video project folder in the sidebar
2. Click "Load Project" to index the folder
3. View video files with their detected languages and available subtitles
4. Use "Refresh Project" to update the index after adding new files

## Voice Refinement

Find the perfect AI voice for your dubbing project with AI-powered voice analysis and preview generation.

### Quick Start

1. **Load your project** and transcribe at least one video
2. **Click "Refine Voice"** from the project detail page
3. **Select a segment** (5-15 seconds) representative of your content
4. **Analyze with AI** to generate voice instructions
5. **Preview all 6 voices** with your custom instructions
6. **Save your selection** - it applies to all future TTS

### Features

- **AI Voice Analysis:** GPT-4o analyzes your audio to generate detailed TTS instructions
- **Smart Previews:** Generate audio samples for all 6 OpenAI voices in ~10 seconds
- **Intelligent Caching:** Reuse cached previews for instant voice switching (80%+ cache hit rate)
- **Cost Effective:** ~$0.02 per refinement session with caching
- **Iterative Refinement:** Edit instructions or regenerate with feedback

### Available Voices

| Voice | Description | Best For |
|-------|-------------|----------|
| **Alloy** | Neutral, balanced | General purpose, versatile content |
| **Echo** | Male, clear | Technical, professional, instruction |
| **Fable** | British, expressive | Storytelling, entertainment |
| **Onyx** | Deep, authoritative | News, reports, authority |
| **Nova** | Warm, engaging | Education, friendly content |
| **Shimmer** | Soft, gentle | Calm content, meditation |

### Documentation

- **[User Guide](VOICE_REFINEMENT_USER_GUIDE.md)** - Step-by-step instructions with tips and troubleshooting
- **[API Reference](VOICE_REFINEMENT_API.md)** - Complete API documentation with code examples
- **[Integration Guide](VOICE_REFINEMENT_INTEGRATION.md)** - Frontend/backend integration and best practices

### Cost Estimate

| Operation | Cost |
|-----------|------|
| Voice analysis (GPT-4o) | ~$0.005 |
| 6 voice previews (10s each) | ~$0.018 |
| **Total per session** | **~$0.023** |
| Cached re-runs | **$0.00** (Free!) |

---

## Project Structure

The application expects project folders containing:
- Video files (mp4, avi, mkv, mov, etc.)
- Subtitle files (srt, vtt, ass, etc.)
- Files can be in subfolders within the project directory

Language detection is based on filename patterns and subtitle content analysis.