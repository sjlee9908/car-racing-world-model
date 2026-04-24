#!/bin/bash

# 1. ffmpeg를 이용한 화면 녹화 시작 (test_output.mp4로 저장)
ffmpeg -y -f x11grab -video_size 1400x900 -framerate 30 -i $DISPLAY -c:v libx264 -pix_fmt yuv420p /app/test_output.mp4 > /dev/null 2>&1 &
FFMPEG_PID=$!

# 2. 수정된 테스트 스크립트 실행
# python test_controller_rnn.py --logdir exp_dir
python test_controller_attn.py --logdir exp_dir

# 3. 테스트 종료 후 녹화 프로세스에 종료 신호 전송 및 대기
kill -s SIGINT $FFMPEG_PID
wait $FFMPEG_PID
