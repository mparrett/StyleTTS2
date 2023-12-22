#!/bin/sh

# Capture 10 seconds of audio from the default input device
# Trim silence
# Save to audios/$1.wav

if [ -z "$1" ]; then
		echo "Usage: $0 <output name>"
		exit 1
fi

ffmpeg -f avfoundation -i ":0" -t 12 audiocapture.wav
sox audiocapture.wav audios/$1.wav silence 1 0.1 1%
soxi -D audios/$1.wav
afplay audios/$1.wav
