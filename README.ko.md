# "World Models"의 PyTorch 구현

논문: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.5281/zenodo.1207631. 논문의 빠른 요약과 추가 실험은 [github 페이지](https://ctallec.github.io/world-models/)를 참고하세요.

## 사전 준비

이 구현은 Python3와 PyTorch를 기반으로 합니다. 설치 방법은 [PyTorch 공식 사이트](https://pytorch.org)를 확인하세요. 나머지 의존성은 [requirements 파일](requirements.txt)에 포함되어 있으며, 다음 명령으로 설치할 수 있습니다.

```bash
pip3 install -r requirements.txt
```

## World Models 실행

모델은 세 부분으로 구성됩니다.

1. Variational Auto-Encoder (VAE): 입력 이미지를 압축된 잠재 표현(latent representation)으로 인코딩합니다.
2. Mixture-Density Recurrent Network (MDN-RNN): 과거 잠재 인코딩과 행동(action)이 주어졌을 때 다음 프레임의 잠재 인코딩을 예측하도록 학습합니다.
3. 선형 Controller (C): 현재 프레임의 잠재 인코딩과, 과거 잠재값/행동으로부터 얻은 MDN-RNN의 은닉 상태를 입력으로 받아 행동을 출력합니다. 이 모듈은 `cma` 파이썬 패키지의 Covariance-Matrix Adaptation Evolution-Strategy([CMA-ES](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf))를 사용해 누적 보상을 최대화하도록 학습합니다.

현재 코드에서는 세 부분을 각각 분리해 `trainvae.py`, `trainmdrnn.py`, `traincontroller.py` 스크립트로 학습합니다.

학습 스크립트 공통 인자:
- **--logdir**: 모델을 저장할 디렉터리입니다. 지정한 logdir이 이미 존재하면 기존 모델을 불러와 이어서 학습합니다.
- **--noreload**: 기존 모델을 불러오지 않고 *logdir*의 모델을 덮어쓰려면 이 옵션을 사용합니다.

### 1. 데이터 생성
VAE와 MDN-RNN 학습을 시작하기 전에 랜덤 롤아웃 데이터셋을 생성해 `datasets/carracing` 폴더에 배치해야 합니다.

데이터 생성은 `data/generation_script.py`로 수행합니다. 예:

```bash
python data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 8
```

롤아웃은 gym의 *white noise* 랜덤 정책(`action_space.sample()`) 대신 *brownian* 랜덤 정책을 사용해 더 일관된 롤아웃을 제공합니다.

### 2. VAE 학습
VAE는 `trainvae.py`로 학습합니다. 예:

```bash
python trainvae.py --logdir exp_dir
```

### 3. MDN-RNN 학습
MDN-RNN은 `trainmdrnn.py`로 학습합니다. 예:

```bash
python trainmdrnn.py --logdir exp_dir
```

이 스크립트가 동작하려면 동일한 `exp_dir`에 VAE가 먼저 학습되어 있어야 합니다.

### 4. Controller 학습 및 테스트
마지막으로 Controller는 CMA-ES로 학습합니다. 예:

```bash
python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
```

학습된 정책은 `test_controller.py`로 테스트할 수 있습니다. 예:

```bash
python test_controller.py --logdir exp_dir
```

### 참고 사항
헤드리스 서버에서 실행할 때는 controller 학습 스크립트를 `xvfb-run`으로 실행해야 합니다. 예:

```bash
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
```

디스플레이가 없는 환경에서 `xvfb-run` 없이 `traincontroller`를 실행하면 스크립트가 조용히 실패할 수 있습니다(로그는 `logdir/tmp`에서 확인 가능).

GPU에서 `traincontroller`를 실행하면 메모리 사용량이 매우 클 수 있습니다. 메모리 부담을 줄이려면 `--max-workers` 인자로 최대 워커 수를 직접 조정하세요.

여러 GPU를 사용할 수 있다면 `traincontroller`는 `CUDA_VISIBLE_DEVICES`에 지정한 모든 GPU를 활용합니다.

## 작성자

- **Corentin Tallec** - [ctallec](https://github.com/ctallec)
- **Léonard Blier** - [leonardblier](https://github.com/leonardblier)
- **Diviyan Kalainathan** - [diviyan-kalainathan](https://github.com/diviyan-kalainathan)

## 라이선스

이 프로젝트는 MIT 라이선스를 따르며, 자세한 내용은 [LICENSE.md](LICENSE.md)를 참고하세요.
