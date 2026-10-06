import re
from typing import List
from story_rp_engine.core.types import Lorebook, LorebookEntry

class LorebookEngine:
    @staticmethod
    def find_matching_entries(lorebooks: List[Lorebook], text: str) -> List[LorebookEntry]:
        """Finds all lorebook entries whose keys appear as words/phrases in the given text."""
        if not text or not lorebooks:
            return []

        matched: List[LorebookEntry] = []
        seen_contents = set()

        for book in lorebooks:
            for entry in book.entries:
                if not entry.enabled or entry.content in seen_contents:
                    continue
                for key in entry.keys:
                    stripped_key = key.strip()
                    if not stripped_key:
                        continue
                    # ASCII word boundaries: \b treats CJK characters as word characters, so a Chinese key
                    # inside unspaced Chinese text would never match.
                    pattern = r'(?<![A-Za-z0-9_])' + re.escape(stripped_key) + r'(?![A-Za-z0-9_])'
                    if re.search(pattern, text, re.IGNORECASE):
                        matched.append(entry)
                        seen_contents.add(entry.content)
                        break

        # Sort by insertion_order ascending
        matched.sort(key=lambda x: x.insertion_order)
        return matched
