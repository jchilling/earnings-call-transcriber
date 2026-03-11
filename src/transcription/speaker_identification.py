"""Speaker name extraction from transcript text.

Scans transcript segments for self-introduction patterns common in
earnings calls (e.g. "This is Jeff Su, Director of IR") and maps
diarization speaker IDs to real names and titles.
"""

import re
from collections import defaultdict

import structlog

from src.transcription import SpeakerInfo, TranscriptSegment

logger = structlog.get_logger(__name__)

# Patterns that capture (name, title) from common earnings call introductions.
# Each pattern should have named groups: 'name' and optionally 'title'.
_INTRO_PATTERNS: list[re.Pattern[str]] = [
    # "This is Jeff Su, Director of Investor Relations"
    re.compile(
        r"[Tt]his is (?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+)+),?\s+(?P<title>[A-Z][\w\s&,]+)",
    ),
    # "My name is Jeff Su, Director of IR"
    re.compile(
        r"[Mm]y name is (?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+)+),?\s+(?P<title>[A-Z][\w\s&,]+)",
    ),
    # "I'm Jeff Su, Director of IR" / "I am Jeff Su, Director of IR"
    re.compile(
        r"I[''']?m (?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+)+),?\s+(?P<title>[A-Z][\w\s&,]+)",
    ),
    re.compile(
        r"I am (?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+)+),?\s+(?P<title>[A-Z][\w\s&,]+)",
    ),
    # "Jeff Su, Director of IR speaking"
    re.compile(
        r"(?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+)+),\s+(?P<title>[A-Z][\w\s&,]+?)\s+speaking",
    ),
    # "I am Jeff Su from TSMC"
    re.compile(
        r"I am (?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+)+) from (?P<title>[\w\s&]+)",
    ),
    # Simpler fallback: "This is Jeff Su" (no title)
    re.compile(
        r"[Tt]his is (?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+)+)",
    ),
    # "My name is Jeff Su"
    re.compile(
        r"[Mm]y name is (?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+)+)",
    ),
]

# Common filler words that get falsely matched as names
_EXCLUDE_NAMES = {
    "Thank You", "Good Morning", "Good Afternoon", "Good Evening",
    "Ladies And", "Next Slide", "Operator Please",
}


def _clean_title(title: str | None) -> str | None:
    """Clean up extracted title string."""
    if not title:
        return None
    # Strip trailing punctuation and whitespace
    title = title.strip().rstrip(".,;:")
    # Reject if too short or just a company name preposition
    if len(title) < 3:
        return None
    return title


def extract_speaker_names(
    segments: list[TranscriptSegment],
) -> dict[str, SpeakerInfo]:
    """Extract speaker names from self-introductions in transcript segments.

    Scans the text of each segment for introduction patterns and maps the
    diarization speaker_id to a real name and title. Uses the first
    introduction found for each speaker (people typically introduce
    themselves once at the start).

    Args:
        segments: Transcript segments with speaker_id already assigned
            from diarization alignment.

    Returns:
        Dict mapping speaker_id → SpeakerInfo with resolved name/title.
        Includes entries for all speaker IDs found in segments, even
        those without a detected introduction.
    """
    # Count segments per speaker
    segment_counts: dict[str, int] = defaultdict(int)
    for seg in segments:
        if seg.speaker_id:
            segment_counts[seg.speaker_id] += 1

    # Try to find introductions
    identified: dict[str, tuple[str, str | None]] = {}  # speaker_id → (name, title)

    for seg in segments:
        if not seg.speaker_id or seg.speaker_id in identified:
            continue

        for pattern in _INTRO_PATTERNS:
            match = pattern.search(seg.text)
            if match:
                name = match.group("name")

                # Skip false positives
                if name in _EXCLUDE_NAMES:
                    continue

                title = match.groupdict().get("title")
                title = _clean_title(title)

                identified[seg.speaker_id] = (name, title)
                logger.info(
                    "speaker_identified",
                    speaker_id=seg.speaker_id,
                    name=name,
                    title=title,
                    segment_text=seg.text[:100],
                )
                break

    # Build SpeakerInfo for all speakers
    speakers: dict[str, SpeakerInfo] = {}
    for speaker_id, count in segment_counts.items():
        name, title = identified.get(speaker_id, (None, None))
        speakers[speaker_id] = SpeakerInfo(
            id=speaker_id,
            name=name,
            title=title,
            segments_count=count,
        )

    logger.info(
        "speaker_extraction_complete",
        total_speakers=len(speakers),
        identified=len(identified),
    )
    return speakers
