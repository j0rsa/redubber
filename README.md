# Redubber - Audio Redub Project Manager

A Streamlit application for managing audio-redub projects with automatic video and subtitle file indexing.

## Features

- 📁 Project folder indexing and management
- 🎥 Automatic video file detection with language recognition
- 📝 Subtitle file detection with language identification
- 🗄️ SQLite database for fast project re-opening
- 🔄 Project refresh functionality to update file index

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter the path to your video project folder in the sidebar
2. Click "Load Project" to index the folder
3. View video files with their detected languages and available subtitles
4. Use "Refresh Project" to update the index after adding new files

## Project Structure

The application expects project folders containing:
- Video files (mp4, avi, mkv, mov, etc.)
- Subtitle files (srt, vtt, ass, etc.)
- Files can be in subfolders within the project directory

Language detection is based on filename patterns and subtitle content analysis.