from pydantic import BaseModel
import os
from enum import Enum

class Reproj(BaseModel):
    root: str
    source: str
    file_path: str
    filename: str

    class Section(Enum):
        ROOT = ""
        SOURCE_AUDIO_CHUNKS = "01_source_audio_chunks"
        STT = "02_stt"
        SUBTITLES = "03_subtitles"
        TTS = "04_tts"
        TARGET_AUDIO_CHUNKS = "05_target_audio_chunks"

    def __init__(self, source: str, file_path: str, root: str = "redubber_tmp"):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        super().__init__(root=root, source=source, file_path=file_path, filename=filename)
        
    def get_file_working_dir(self, section: Section) -> str:
        """
        Get the working directory for the given section.
        """
        rel_file = os.path.relpath(self.file_path, self.source)
        path = os.path.join(self.root, rel_file, section.value)
        os.makedirs(path, exist_ok=True)
        return path


    