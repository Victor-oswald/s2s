"""
Manages voice cloning profiles on disk.

Profile structure on disk:
  <profile_dir>/
    <profile_id>/
      ref_audio.wav     <- reference audio
      meta.json         <- {"name": str, "ref_text": str|null, "created_at": str}
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnivoice.models.omnivoice import VoiceClonePrompt

logger = logging.getLogger(__name__)

PROFILE_META_FILE = "meta.json"
PROFILE_AUDIO_FILE = "ref_audio.wav"
PROFILE_PROMPT_CACHE_FILE = "voice_clone_prompt.pt"


class ProfileNotFoundError(Exception):
    pass


class ProfileAlreadyExistsError(Exception):
    pass


class ProfileService:
    def __init__(self, profile_dir: Path) -> None:
        self._dir = profile_dir
        # In-memory cache of VoiceClonePrompt per profile_id. This is the
        # fix for the "every TTS call re-encodes the reference audio"
        # latency tax: OmniVoice.create_voice_clone_prompt() runs silence
        # trimming, RMS normalization, and a real GPU audio_tokenizer.encode()
        # forward pass — that's the ~1.1-1.2s fixed cost we were paying on
        # every single request, one word or eight. Doing it once per profile
        # and reusing the resulting (small) token tensor skips all of that
        # on every subsequent call.
        self._prompt_cache: dict[str, "VoiceClonePrompt"] = {}

    def get_voice_clone_prompt(self, profile_id: str, model) -> "VoiceClonePrompt":
        """
        Return a cached VoiceClonePrompt for profile_id, building (and
        persisting) it on first use. Safe to call on every request — cache
        hits are a dict lookup, no GPU work.
        """
        cached = self._prompt_cache.get(profile_id)
        if cached is not None:
            return cached

        from omnivoice.models.omnivoice import VoiceClonePrompt

        profile_path = self._profile_path(profile_id)
        disk_cache_path = profile_path / PROFILE_PROMPT_CACHE_FILE

        # Reuse a prompt persisted by a previous process (e.g. before a
        # pod restart) instead of re-encoding from the raw wav again.
        if disk_cache_path.exists():
            try:
                prompt = VoiceClonePrompt.load(str(disk_cache_path))
                self._prompt_cache[profile_id] = prompt
                logger.info(
                    f"[VoiceClonePromptCache] Loaded persisted prompt for "
                    f"'{profile_id}' from disk (skipped re-encode)."
                )
                return prompt
            except Exception:
                logger.warning(
                    f"[VoiceClonePromptCache] Failed to load persisted prompt "
                    f"for '{profile_id}'; re-encoding.",
                    exc_info=True,
                )

        ref_audio_path = self.get_ref_audio_path(profile_id)  # raises ProfileNotFoundError
        ref_text = self.get_ref_text(profile_id)

        logger.info(
            f"[VoiceClonePromptCache] Encoding reference audio for profile "
            f"'{profile_id}' (one-time cost; all future calls reuse this)..."
        )
        t0 = time.monotonic()
        prompt = model.create_voice_clone_prompt(str(ref_audio_path), ref_text)
        elapsed = time.monotonic() - t0
        logger.info(f"[VoiceClonePromptCache] Encoded '{profile_id}' in {elapsed:.2f}s.")

        self._prompt_cache[profile_id] = prompt

        try:
            prompt.save(str(disk_cache_path))
        except Exception:
            logger.warning(
                f"[VoiceClonePromptCache] Failed to persist prompt cache for "
                f"'{profile_id}' to disk (non-fatal, still cached in memory).",
                exc_info=True,
            )

        return prompt

    def invalidate_voice_clone_prompt(self, profile_id: str) -> None:
        """Drop cached prompt (memory + disk). Call after re-registering a profile."""
        self._prompt_cache.pop(profile_id, None)
        cache_path = self._profile_path(profile_id) / PROFILE_PROMPT_CACHE_FILE
        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                logger.warning(f"Failed to remove stale prompt cache for '{profile_id}'")

    def list_profiles(self) -> list[dict]:
        """Return list of profile metadata dicts."""
        profiles = []
        for p in sorted(self._dir.iterdir()) if self._dir.exists() else []:
            if p.is_dir():
                meta = self._read_meta(p)
                if meta:
                    profiles.append({"profile_id": p.name, **meta})
        return profiles

    def get_ref_audio_path(self, profile_id: str) -> Path:
        """Return path to ref audio file. Raises ProfileNotFoundError if missing."""
        logger.debug(f"[TRACE] get_ref_audio_path called: profile_id={profile_id!r}")
        path = self._profile_path(profile_id) / PROFILE_AUDIO_FILE
        logger.debug(f"[TRACE] Looking for audio at: {path}")
        if not path.exists():
            logger.warning(f"[TRACE] Profile audio NOT FOUND: {profile_id!r} at path {path}")
            raise ProfileNotFoundError(f"Profile '{profile_id}' not found")
        logger.info(f"[TRACE] Profile audio found: {profile_id!r} at {path}")
        return path

    def get_ref_text(self, profile_id: str) -> str | None:
        """Return ref_text from profile metadata, or None."""
        logger.debug(f"[TRACE] get_ref_text called: profile_id={profile_id!r}")
        meta = self._read_meta(self._profile_path(profile_id))
        result = meta.get("ref_text") if meta else None
        logger.debug(f"[TRACE] ref_text for {profile_id!r}: {result!r}")
        return result

    def save_profile(
        self,
        profile_id: str,
        audio_bytes: bytes,
        ref_text: str | None = None,
        overwrite: bool = False,
    ) -> dict:
        """
        Save a new profile. Raises ProfileAlreadyExistsError if exists and overwrite=False.
        Returns the saved metadata dict.
        """
        profile_path = self._profile_path(profile_id)
        if profile_path.exists() and not overwrite:
            raise ProfileAlreadyExistsError(
                f"Profile '{profile_id}' already exists. Use overwrite=true to replace."
            )

        # New/changed reference audio means any cached embedding is stale.
        self.invalidate_voice_clone_prompt(profile_id)

        profile_path.mkdir(parents=True, exist_ok=True)

        # Write audio
        audio_path = profile_path / PROFILE_AUDIO_FILE
        audio_path.write_bytes(audio_bytes)

        # Write metadata
        now = datetime.now(timezone.utc).isoformat()
        meta = {
            "name": profile_id,
            "ref_text": ref_text,
            "created_at": now,
        }
        (profile_path / PROFILE_META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )

        logger.info(f"Saved profile '{profile_id}'")
        return {"profile_id": profile_id, **meta}

    def delete_profile(self, profile_id: str) -> None:
        profile_path = self._profile_path(profile_id)
        if not profile_path.exists():
            raise ProfileNotFoundError(f"Profile '{profile_id}' not found")
        self._prompt_cache.pop(profile_id, None)
        shutil.rmtree(profile_path)
        logger.info(f"Deleted profile '{profile_id}'")

    def _profile_path(self, profile_id: str) -> Path:
        # Sanitize: only allow alphanumeric + dash + underscore
        safe = "".join(c for c in profile_id if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError(f"Invalid profile_id: '{profile_id}'")
        return self._dir / safe

    def _read_meta(self, profile_path: Path) -> dict | None:
        meta_file = profile_path / PROFILE_META_FILE
        if not meta_file.exists():
            return None
        try:
            return json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None