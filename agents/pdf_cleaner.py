

import re
from typing import List

class TextCleanerChunker:

    def clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text:
        - Remove multiple newlines
        - Remove extra spaces
        - Remove weird characters
        """
        text = re.sub(r'\n+', '\n', text)                # remove excessive newlines
        text = re.sub(r'\s+', ' ', text)                 # collapse spaces
        text = text.replace("\x0c", " ")                 # remove form feed
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 800) -> List[str]:
        """
        Split into chunks of ~800 words.
        """
        words = text.split(" ")
        chunks = []

        current = []
        count = 0

        for w in words:
            current.append(w)
            count += 1

            if count >= chunk_size:
                chunks.append(" ".join(current))
                current = []
                count = 0

        if current:
            chunks.append(" ".join(current))

        return chunks

    def clean_and_chunk_with_metadata(self, raw_text: str, chunk_size: int = 300) -> List[dict]:
        """
        Splits raw text into pages using '--- Page X ---' markers, cleans them,
        and generates chunks mapped to metadata.
        Returns: List of dicts, e.g., [{"text": "...", "metadata": {"page": 1}}]
        """
        pattern = r'--- Page (\d+) ---'
        parts = re.split(pattern, raw_text)
        
        chunks_with_metadata = []
        
        # If there are no page markers, treat entire text as page 1
        if len(parts) < 3:
            cleaned = self.clean_text(raw_text)
            chunks = self.chunk_text(cleaned, chunk_size=chunk_size)
            for chunk in chunks:
                chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": {"page": 1}
                })
            return chunks_with_metadata

        # Loop through split parts to build page chunks
        for i in range(1, len(parts), 2):
            try:
                page_num = int(parts[i])
            except ValueError:
                page_num = (i // 2) + 1
            
            page_text = parts[i+1]
            cleaned = self.clean_text(page_text)
            page_chunks = self.chunk_text(cleaned, chunk_size=chunk_size)
            
            for chunk in page_chunks:
                if chunk.strip():
                    chunks_with_metadata.append({
                        "text": chunk,
                        "metadata": {"page": page_num}
                    })
                    
        return chunks_with_metadata
