# Taiwan Audio Resolver — Multi-Strategy

## Plan
- [x] `src/sources/hinet_ott.py` — HiNet OTT Live client
- [x] `src/sources/audio_resolver.py` — Multi-strategy audio URL resolver
- [x] `src/audio/downloader.py` — yt-dlp based audio downloader
- [x] Update `src/sources/taiwan.py` — Wire resolver into `get_audio_url()`
- [x] `tests/test_sources/test_hinet_ott.py` — Unit tests (mocked HTTP)
- [x] `tests/test_sources/test_audio_resolver.py` — Unit tests (mocked strategies)
- [x] Update `tests/test_sources/test_taiwan.py` — Update get_audio_url test
- [x] `scripts/test_audio_resolver_live.py` — Live integration test → JSON output
- [x] Run unit tests — 76/76 pass
- [x] Lint clean (ruff)
- [ ] Run live test with TSMC

## Verification
- [x] All 76 unit tests pass
- [x] Ruff lint clean
- [ ] Live test: TSMC → HiNet → M3U8 reachable
- [ ] Results dumped to JSON
