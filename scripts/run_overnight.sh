#!/bin/bash
# Overnight runner — restarts scrapers if they crash, runs HF upload at the end.
# Usage: nohup bash scripts/run_overnight.sh > data/audio/overnight.log 2>&1 &

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "$(date) — Starting overnight scrape"

# ── AlphaMemo: retry loop ──────────────────────────────────────────────────
run_alphamemo() {
    for attempt in $(seq 1 20); do
        echo "$(date) — AlphaMemo attempt $attempt"
        python scripts/scrape_alphamemo.py 2>&1 | tail -5

        # Check if done (all downloaded or only non-retryable errors left)
        local remaining=$(python3 -c "
import json
r = json.load(open('data/audio/alphamemo/download_results.json'))
total = 2876
done = len(r['downloads']) + len(r['errors'])
print(total - done)
" 2>/dev/null || echo "999")

        if [ "$remaining" -le 5 ]; then
            echo "$(date) — AlphaMemo complete!"
            return 0
        fi

        # Clear network errors for retry
        python3 -c "
import json
r = json.load(open('data/audio/alphamemo/download_results.json'))
retryable = [k for k,v in r['errors'].items() if any(x in str(v) for x in ['nodename','Errno','api_401','api_502','retry_failed','ConnectionError','TimeoutException'])]
for k in retryable: del r['errors'][k]
print(f'Cleared {len(retryable)} retryable errors')
with open('data/audio/alphamemo/download_results.json','w') as f: json.dump(r,f,indent=2,ensure_ascii=False)
" 2>/dev/null

        echo "$(date) — AlphaMemo crashed/finished, waiting 30s before retry..."
        sleep 30
    done
}

# ── TWSE: retry loop ──────────────────────────────────────────────────────
run_twse() {
    for attempt in $(seq 1 20); do
        echo "$(date) — TWSE attempt $attempt"
        python scripts/scrape_twse_webpro.py --concurrency 5 2>&1 | tail -5

        local remaining=$(python3 -c "
import json
r = json.load(open('data/audio/twse_webpro/download_results.json'))
total = 2674
done = len(r['downloads']) + len(r['errors'])
print(total - done)
" 2>/dev/null || echo "999")

        if [ "$remaining" -le 5 ]; then
            echo "$(date) — TWSE complete!"
            return 0
        fi

        # Only clear errors on first retry — after that, failed URLs are truly dead
        if [ "$attempt" -le 2 ]; then
            python3 -c "
import json
r = json.load(open('data/audio/twse_webpro/download_results.json'))
retryable = [k for k,v in r['errors'].items() if v == 'ffmpeg_failed']
for k in retryable: del r['errors'][k]
print(f'Cleared {len(retryable)} retryable errors')
with open('data/audio/twse_webpro/download_results.json','w') as f: json.dump(r,f,indent=2,ensure_ascii=False)
" 2>/dev/null
        else
            echo "$(date) — Not clearing errors (attempt $attempt > 2), treating as permanent"
        fi

        echo "$(date) — TWSE crashed/finished, waiting 30s before retry..."
        sleep 30
    done
}

# ── HuggingFace upload ─────────────────────────────────────────────────────
upload_hf() {
    echo "$(date) — Uploading to HuggingFace..."
    python scripts/upload_hf.py 2>&1
    echo "$(date) — HuggingFace upload done!"
}

# ── Kill existing scrapers ─────────────────────────────────────────────────
echo "$(date) — Killing existing scraper processes..."
pkill -f "scrape_alphamemo" 2>/dev/null || true
pkill -f "scrape_twse_webpro" 2>/dev/null || true
sleep 3

# ── Run both in parallel, then upload ──────────────────────────────────────
run_alphamemo &
ALPHA_PID=$!

run_twse &
TWSE_PID=$!

echo "$(date) — AlphaMemo PID: $ALPHA_PID, TWSE PID: $TWSE_PID"

# Wait for both to finish
wait $ALPHA_PID 2>/dev/null
echo "$(date) — AlphaMemo loop finished"

wait $TWSE_PID 2>/dev/null
echo "$(date) — TWSE loop finished"

# Upload everything to HuggingFace
upload_hf

echo "$(date) — ALL DONE!"

# Print final stats
python3 -c "
import json
a = json.load(open('data/audio/alphamemo/download_results.json'))
t = json.load(open('data/audio/twse_webpro/download_results.json'))
a_ok = len(a['downloads'])
a_audio = sum(1 for v in a['downloads'].values() if v.get('has_audio'))
t_ok = len(t['downloads'])
print(f'AlphaMemo: {a_ok} transcripts, {a_audio} with audio, {len(a[\"errors\"])} errors')
print(f'TWSE:      {t_ok} videos downloaded, {len(t[\"errors\"])} errors')
"
