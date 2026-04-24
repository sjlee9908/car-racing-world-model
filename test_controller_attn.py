""" Test controller """
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
args = parser.parse_args()

ctrl_file = join(args.logdir, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

generator = RolloutGenerator(args.logdir, device, 1000)

with torch.no_grad():
    # render=True를 추가하여 화면 렌더링을 활성화
    # rollout 함수가 반환하는 값은 음수화된 누적 보상(-cumulative)입니다.
    minus_score = generator.rollout(None, render=True)
    
    # 실제 누적 점수로 복구
    actual_score = -minus_score

# 결과 점수를 로그 디렉토리 내에 저장
score_log_path = join(args.logdir, 'test_score.txt')
with open(score_log_path, 'w') as f:
    f.write(f"Test Score: {actual_score}\n")