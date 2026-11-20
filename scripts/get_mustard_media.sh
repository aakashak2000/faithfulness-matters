#!/usr/bin/env bash
# Download MUStARD++ utterance videos into data/mustard_pp/ (gitignored), check them against the label CSV,
# and extract 16 kHz mono audio for WavLM. Only `final_utterance_videos` is fetched (no context or augmented clips).
#
# Route A (recommended, no 50-file limit): rclone with your own Google login.
#   brew install rclone
#   rclone config create gdrive drive scope=drive.readonly      # opens a browser to log in, once
#   bash scripts/get_mustard_media.sh rclone
#
# Route B: in the Drive web page, right-click `final_utterance_videos` > Download (Drive may split it into several zips), then
#   bash scripts/get_mustard_media.sh zips ~/Downloads/final_utterance_videos*.zip
#
# Either route then runs:  bash scripts/get_mustard_media.sh check     (also run automatically after A or B)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/mustard_pp"
VID="$OUT/final_utterance_videos"
WAV="$OUT/wav16k"
CSV="$OUT/mustard++_text.csv"
PARENT_ID="1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW"      # shared "all_videos" folder linked from the MUStARD++ README
CSV_URL="https://raw.githubusercontent.com/cfiltnlp/MUStARD_Plus_Plus/main/mustard%2B%2B_text.csv"

mkdir -p "$VID" "$WAV"
[ -f "$CSV" ] || curl -sfL -o "$CSV" "$CSV_URL"

check() {
  # every utterance KEY (rows ending in _u) must have exactly one video; then extract audio for the matched ones
  python3 - "$CSV" "$VID" <<'EOF'
import sys, pathlib, csv
csv_path, vid = sys.argv[1], pathlib.Path(sys.argv[2])
keys = {r["KEY"] for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["KEY"].endswith("_u")}
vids = {p.stem: p for p in vid.rglob("*") if p.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov")}
missing, extra = sorted(keys - vids.keys()), sorted(vids.keys() - keys)
print(f"label utterances {len(keys)}, videos found {len(vids)}, matched {len(keys & vids.keys())}, missing {len(missing)}, unmatched videos {len(extra)}")
if missing: print("first missing:", missing[:10])
if extra: print("first unmatched:", extra[:10])
pathlib.Path(vid.parent / "matched_videos.txt").write_text("\n".join(str(vids[k]) for k in sorted(keys & vids.keys())) + "\n")
EOF
  local n=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    out="$WAV/$(basename "${f%.*}").wav"
    [ -s "$out" ] && continue
    ffmpeg -nostdin -loglevel error -y -i "$f" -vn -ac 1 -ar 16000 -sample_fmt s16 "$out" && n=$((n + 1))
  done < "$OUT/matched_videos.txt"
  echo "extracted $n new wav files; total in $WAV: $(ls "$WAV" | wc -l | tr -d ' ')"
  du -sh "$VID" "$WAV" | sed 's/^/size: /'
}

case "${1:-}" in
  rclone)
    command -v rclone >/dev/null || { echo "rclone not installed: brew install rclone"; exit 1; }
    rclone listremotes | grep -q '^gdrive:$' || { echo "no 'gdrive' remote: rclone config create gdrive drive scope=drive.readonly"; exit 1; }
    rclone copy "gdrive:final_utterance_videos" "$VID" --drive-root-folder-id "$PARENT_ID" --drive-acknowledge-abuse -P
    check ;;
  zips)
    shift
    [ "$#" -gt 0 ] || { echo "give the zip files: bash scripts/get_mustard_media.sh zips ~/Downloads/*.zip"; exit 1; }
    for z in "$@"; do unzip -n -q "$z" -d "$VID"; done
    check ;;
  check)
    check ;;
  *)
    sed -n '2,15p' "$0"; exit 1 ;;
esac
