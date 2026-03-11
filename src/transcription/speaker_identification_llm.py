"""LLM-based speaker identification using DeepSeek.

Two-pass approach:
1. deepseek-chat: Fast, cheap identification of speaker IDs → names
2. deepseek-reasoner: Corrects speaker assignments per-segment using
   chain-of-thought reasoning (catches mid-block speaker handoffs that
   VAD misses because there's no silence gap)

Cache optimization: Transcript (large, stable) goes in system message.
DeepSeek auto-caches matching prefixes — repeated calls on the same
transcript hit cache at 74% discount ($0.07/M vs $0.27/M).
"""

import json

import structlog

from src.config import settings
from src.transcription import SpeakerInfo, TranscriptSegment

logger = structlog.get_logger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _build_transcript_text(segments: list[TranscriptSegment]) -> str:
    """Build compact full transcript for the LLM.

    Format: [SPEAKER_XX] (M:SS): text
    """
    lines: list[str] = []
    for seg in segments:
        speaker = seg.speaker_id or "UNKNOWN"
        ts = _format_timestamp(seg.start_time)
        lines.append(f"[{speaker}] ({ts}): {seg.text}")
    return "\n".join(lines)


def _build_system_message(
    transcript_text: str,
    speakers: dict[str, SpeakerInfo],
) -> str:
    """Build the system message containing the transcript.

    This is the large, stable content that benefits from DeepSeek's
    automatic prefix caching. Repeated calls with the same transcript
    will hit cache at $0.07/M vs $0.27/M (74% savings).
    """
    speaker_summary = []
    for sid, info in sorted(speakers.items()):
        label = f"  {sid}: {info.segments_count} segments"
        if info.name:
            label += f" (already identified: {info.name})"
        speaker_summary.append(label)

    return f"""You are an expert at analyzing earnings call transcripts to identify speakers.

SPEAKER IDS FOUND (assigned by silence-gap detection, may over-segment):
{chr(10).join(speaker_summary)}

FULL TRANSCRIPT:
{transcript_text}"""


# Task instructions — kept separate from transcript so we can iterate
# on the prompt without invalidating the transcript cache.
IDENTIFICATION_PROMPT = """Identify who each speaker ID corresponds to by analyzing the transcript above.

Look for:
1. Self-introductions ("This is...", "My name is...")
2. Others addressing them ("Thank you, [name]", "Next one, [name] from [firm]")
3. Operator announcing analysts ("[Name], [Firm], go ahead please")
4. Context: the person presenting financials is likely the CFO, the person discussing strategy/technology is likely the CEO, the moderator introduces and transitions between speakers

IMPORTANT: Each speaker ID may or may not be a unique person. The same person might have multiple IDs (split by pauses). You cannot know from text alone — so identify each one independently. The same name CAN appear for multiple speaker IDs.

Return ONLY a JSON object mapping speaker_id to name and title:
{
  "SPEAKER_00": {"name": "Jeff Su", "title": "Director of IR"},
  "SPEAKER_01": {"name": "Wendell Huang", "title": "CFO"},
  "SPEAKER_05": {"name": "Gokul Hariharan", "title": "Analyst, JP Morgan"}
}

Rules:
- Only include speaker IDs you can identify with reasonable confidence
- Use null for title if unknown
- If you cannot identify a speaker, omit it
- Return {} if you cannot identify anyone
"""


def _parse_llm_response(content: str) -> dict:
    """Parse LLM JSON response, handling markdown code blocks."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(content)


def _apply_identifications(
    llm_result: dict,
    speakers: dict[str, SpeakerInfo],
) -> dict[str, SpeakerInfo]:
    """Apply LLM identifications to speaker dict. Regex results take priority."""
    updated = dict(speakers)
    filled = 0
    for sid, info in llm_result.items():
        if sid not in updated:
            continue
        if updated[sid].name:
            continue  # regex already identified, don't overwrite
        if not isinstance(info, dict):
            continue
        name = info.get("name")
        if not name or not isinstance(name, str):
            continue

        raw_title = info.get("title")
        title = raw_title if isinstance(raw_title, str) else None

        updated[sid] = SpeakerInfo(
            id=sid,
            name=name,
            title=title,
            segments_count=updated[sid].segments_count,
        )
        filled += 1
        logger.info(
            "llm_speaker_identified",
            speaker_id=sid,
            name=name,
            title=title,
            segments=updated[sid].segments_count,
        )

    logger.info(
        "llm_speaker_id_completed",
        filled=filled,
        total_speakers=len(updated),
    )
    return updated


async def enhance_speaker_names(
    segments: list[TranscriptSegment],
    speakers: dict[str, SpeakerInfo],
) -> tuple[list[TranscriptSegment], dict[str, SpeakerInfo]]:
    """Use DeepSeek to identify speaker names from the full transcript.

    Does NOT merge speaker IDs — only identifies names. Merging is
    handled by _merge_speakers_by_name in the pipeline.

    Message structure for cache optimization:
    - system: transcript text (large, stable → cached by DeepSeek)
    - user: task instructions (small, can iterate freely)

    Gracefully returns unchanged data on any error.

    Args:
        segments: Transcript segments with speaker IDs.
        speakers: Current speaker info from regex extraction.

    Returns:
        Tuple of (segments_unchanged, updated_speakers).
    """
    # Skip if no API key configured
    if not settings.deepseek_api_key:
        logger.info("llm_speaker_id_skipped", reason="no deepseek_api_key")
        return segments, speakers

    # Skip if all speakers already identified
    unidentified = [sid for sid, info in speakers.items() if not info.name]
    if not unidentified:
        logger.info("llm_speaker_id_skipped", reason="all speakers already identified")
        return segments, speakers

    logger.info(
        "llm_speaker_id_started",
        total_speakers=len(speakers),
        unidentified=len(unidentified),
        total_segments=len(segments),
    )

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )

        transcript_text = _build_transcript_text(segments)
        system_msg = _build_system_message(transcript_text, speakers)

        response = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": IDENTIFICATION_PROMPT},
            ],
            temperature=0.1,
            max_tokens=2000,
        )

        content = response.choices[0].message.content.strip()

        # Log token usage with cache hit info
        usage = response.usage
        if usage:
            extra = {}
            # DeepSeek returns cache stats in usage
            cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
            cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)
            if cache_hit is not None:
                extra["cache_hit_tokens"] = cache_hit
                extra["cache_miss_tokens"] = cache_miss
            logger.info(
                "llm_token_usage",
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                **extra,
            )

        llm_result = _parse_llm_response(content)
        updated = _apply_identifications(llm_result, speakers)

        return segments, updated

    except ImportError:
        logger.warning("llm_speaker_id_failed", reason="openai package not installed")
        return segments, speakers
    except json.JSONDecodeError as e:
        logger.warning("llm_speaker_id_failed", reason="invalid JSON response", error=str(e))
        return segments, speakers
    except Exception as e:
        logger.warning("llm_speaker_id_failed", reason=str(e))
        return segments, speakers


# --- Pass 2: Reasoner-based speaker correction ---

def _find_transition_windows(
    segments: list[TranscriptSegment],
) -> list[tuple[int, int]]:
    """Find windows around likely speaker transitions.

    Returns (start, end) index ranges for segments that likely contain
    a speaker handoff within a single speaker block. These are segments
    where the assigned speaker stays the same but the text contains
    handoff cues (e.g. "let me turn the microphone over to...").
    """
    handoff_keywords = (
        "turn the microphone", "turn it over", "hand over",
        "let me turn", "now let me turn",
        "thank you, jeff", "thank you, wendell", "thank you, cc",
        "thank you, c.c", "this concludes",
        "good afternoon, everyone",
    )

    windows: list[tuple[int, int]] = []
    for i, seg in enumerate(segments):
        text_lower = seg.text.lower()
        if any(kw in text_lower for kw in handoff_keywords):
            start = max(0, i - 2)
            end = min(len(segments), i + 5)
            windows.append((start, end))

    # Merge overlapping windows
    if not windows:
        return []
    merged: list[tuple[int, int]] = [windows[0]]
    for start, end in windows[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def _build_correction_transcript(
    segments: list[TranscriptSegment],
    windows: list[tuple[int, int]],
) -> str:
    """Build compact transcript of only the transition windows.

    Format: [idx] (M:SS) Speaker Name: text
    Typically produces 30-60 segments — small enough for the reasoner.
    """
    lines: list[str] = []
    prev_end = -1
    for start, end in windows:
        if prev_end >= 0:
            lines.append(f"... (segments {prev_end}-{start - 1} omitted, same speakers continue) ...")
        for i in range(start, end):
            seg = segments[i]
            speaker = seg.speaker_name or seg.speaker_id or "UNKNOWN"
            ts = _format_timestamp(seg.start_time)
            lines.append(f"[{i}] ({ts}) {speaker}: {seg.text}")
        prev_end = end

    total_included = sum(end - start for start, end in windows)
    logger.info(
        "correction_transcript_built",
        total_segments=len(segments),
        included_segments=total_included,
        windows=len(windows),
    )

    return "\n".join(lines)


CORRECTION_PROMPT = """Review the transcript above. The speaker names were assigned automatically and contain errors — especially where one speaker hands off to another mid-block without a pause.

Common errors to look for:
1. Speaker handoffs: "Now let me turn the microphone over to CC. Thank you, Wendell." — everything after the handoff belongs to the NEW speaker, not the old one
2. Q&A transitions: The moderator summarizes a question, then a different executive answers
3. Operator segments: Brief "Next one, [Name], [Firm]" segments attributed to the wrong person

For each segment that has the WRONG speaker, output a correction.

Return ONLY a JSON object mapping segment index (as string) to the correct speaker name:
{
  "42": "C.C. Wei",
  "43": "C.C. Wei",
  "105": "Jeff Su"
}

Rules:
- Only include segments that need correction — do NOT include segments that are already correct
- Use the exact speaker names from the transcript (e.g. "Jeff Su", not "Jeff")
- Return {} if no corrections are needed
"""


def _log_usage(usage: object, label: str) -> None:
    """Log token usage with cache hit info."""
    if not usage:
        return
    extra = {}
    cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
    cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)
    if cache_hit is not None:
        extra["cache_hit_tokens"] = cache_hit
        extra["cache_miss_tokens"] = cache_miss
    logger.info(
        f"llm_token_usage_{label}",
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        **extra,
    )


async def correct_speaker_assignments(
    segments: list[TranscriptSegment],
    speakers: dict[str, SpeakerInfo],
) -> tuple[list[TranscriptSegment], dict[str, SpeakerInfo]]:
    """Use DeepSeek-reasoner to correct speaker mis-assignments.

    The reasoner uses chain-of-thought to detect speaker handoffs and
    Q&A transitions that VAD missed (no silence gap between speakers).

    Args:
        segments: Segments with speaker names already assigned.
        speakers: Current speaker dict.

    Returns:
        Tuple of (corrected_segments, updated_speakers).
    """
    if not settings.deepseek_api_key:
        logger.info("llm_correction_skipped", reason="no deepseek_api_key")
        return segments, speakers

    identified_names = {info.name for info in speakers.values() if info.name}
    if not identified_names:
        logger.info("llm_correction_skipped", reason="no identified speakers to correct")
        return segments, speakers

    # Find transition windows
    windows = _find_transition_windows(segments)
    if not windows:
        logger.info("llm_correction_skipped", reason="no transition points found")
        return segments, speakers

    logger.info("llm_correction_started", segments=len(segments), windows=len(windows))

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )

        correction_transcript = _build_correction_transcript(segments, windows)

        # deepseek-reasoner only supports user messages (no system message).
        # The transcript prefix still benefits from auto-caching.
        user_msg = f"""You are an expert at analyzing earnings call transcripts. Review the speaker assignments for errors.

KNOWN SPEAKERS: {', '.join(sorted(identified_names))}

I'm showing you ONLY the segments around likely speaker transitions (handoffs). The segments in between are omitted.

TRANSCRIPT SEGMENTS AROUND TRANSITIONS:
{correction_transcript}

{CORRECTION_PROMPT}"""

        response = await client.chat.completions.create(
            model=settings.deepseek_reasoner_model,
            messages=[
                {"role": "user", "content": user_msg},
            ],
            max_tokens=16000,
        )

        content = response.choices[0].message.content.strip()
        _log_usage(response.usage, "correction")

        # Log reasoning if available
        reasoning = getattr(response.choices[0].message, "reasoning_content", None)
        if reasoning:
            # Truncate for logging
            logger.info("llm_correction_reasoning", preview=reasoning[:500])

        corrections = _parse_llm_response(content)

        if not corrections:
            logger.info("llm_correction_completed", corrections=0)
            return segments, speakers

        # Apply corrections
        corrected = list(segments)
        applied = 0
        for idx_str, new_speaker in corrections.items():
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            if idx < 0 or idx >= len(corrected):
                continue
            if not isinstance(new_speaker, str):
                continue

            old_seg = corrected[idx]
            if old_seg.speaker_name == new_speaker:
                continue  # already correct

            corrected[idx] = TranscriptSegment(
                text=old_seg.text,
                start_time=old_seg.start_time,
                end_time=old_seg.end_time,
                speaker_id=old_seg.speaker_id,
                speaker_name=new_speaker,
                language=old_seg.language,
                confidence=old_seg.confidence,
            )
            applied += 1

        # Rebuild speaker dict with corrected counts
        from collections import Counter
        name_counts: Counter[str] = Counter()
        name_titles: dict[str, str | None] = {}
        for seg in corrected:
            name = seg.speaker_name
            if name:
                name_counts[name] += 1
                if name not in name_titles:
                    # Find title from existing speakers
                    for info in speakers.values():
                        if info.name == name:
                            name_titles[name] = info.title
                            break

        corrected_speakers: dict[str, SpeakerInfo] = {}
        for i, (name, count) in enumerate(name_counts.most_common()):
            sid = f"SPEAKER_{i:02d}"
            corrected_speakers[sid] = SpeakerInfo(
                id=sid,
                name=name,
                title=name_titles.get(name),
                segments_count=count,
            )

        # Update segment speaker_ids to match new speaker dict
        name_to_sid = {info.name: sid for sid, info in corrected_speakers.items()}
        final_segments = []
        for seg in corrected:
            new_sid = name_to_sid.get(seg.speaker_name, seg.speaker_id)
            final_segments.append(
                TranscriptSegment(
                    text=seg.text,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    speaker_id=new_sid,
                    speaker_name=seg.speaker_name,
                    language=seg.language,
                    confidence=seg.confidence,
                )
            )

        logger.info(
            "llm_correction_completed",
            corrections=applied,
            speakers=len(corrected_speakers),
        )

        return final_segments, corrected_speakers

    except ImportError:
        logger.warning("llm_correction_failed", reason="openai package not installed")
        return segments, speakers
    except json.JSONDecodeError as e:
        logger.warning(
            "llm_correction_failed",
            reason="invalid JSON response",
            error=str(e),
            raw_content=content[:500] if content else "(empty)",
        )
        return segments, speakers
    except Exception as e:
        logger.warning("llm_correction_failed", reason=str(e))
        return segments, speakers
