# Lessons Learned

## Audio Downloading
- **Never use yt-dlp for large direct video URLs** — it downloads the full file before extracting audio. A 2.3 GB MP4 will timeout. Use ffmpeg instead, which streams only the audio track directly from HTTP (~230 MB for a 2h40m call).
- Route YouTube / HLS (.m3u8) to yt-dlp, everything else to ffmpeg.
- Default timeout should be 30 min (1800s), not 10 min — earnings calls are long.

## Audio Resolution
- Cache winning strategies per ticker. TCC always has MP4 on media.taiwancement.com — no need to try HiNet first every time.
- MOPS `video_info` column (column 9) is the most valuable field — often contains direct MP4 URLs from company CDNs.

## Taiwan MOPS
- MOPS requires ROC year dates (Gregorian - 1911).
- Rate limit to 2s between requests or you get blocked.
- SSL needs relaxed X509 strict mode for MOPS certs.
