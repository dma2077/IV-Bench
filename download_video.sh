#!/bin/bash
COOKIE_FILE="your_cookies.txt"  
VIDEO_ID_LIST="dataset/test_video_ids.txt"
OUTPUT_DIR="videos/"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r video_id; do
    if [[ -z "$video_id" ]]; then
        continue
    fi

    echo "⬇️ downloading: $video_id"
    yt-dlp \
        --cookies "$COOKIE_FILE" \
        --ignore-errors \
        --no-overwrites \
        --continue \
        --format "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" \
        --merge-output-format "mp4" \
        --output "$OUTPUT_DIR/%(id)s.%(ext)s" \
        "https://www.youtube.com/watch?v=$video_id"

    echo "✅: $video_id"
    echo
done < "$VIDEO_ID_LIST"

echo "🎉 Finished"