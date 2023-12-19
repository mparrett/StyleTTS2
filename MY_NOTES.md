download notebook
https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Demo_LibriTTS.ipynb#scrollTo=eJdB_nCOIVIN
download models

conda_create_arm64 py3.10.12_arm64-torch2-styletts2 python=3.10.12
conda activate py3.10.12_arm64-torch2-styletts2
cp ../conda-env-specs/py3.10.11_arm64-torch2/environment.yml .
cp ../conda-env-specs/py3.10.11_arm64-torch2/requirements.in .
cp ../conda-env-specs/update.sh ./
./update.sh
fix tabs in spec
pip install gdown
# for reference_audio
gdown --id 1YhQO4O4dAsvkMzWZM8nVFMglYyi554YT
unzip Models.zip

put theese in requirements.in
# pip install SoundFile torchaudio munch torch pydub pyyaml librosa nltk matplotlib 
# accelerate transformers phonemizer einops einops-exts tqdm typing-extensions 
# git+https://github.com/resemble-ai/monotonic_align.git
# sudo apt-get install espeak-ng
./compile.sh, inspect
move some things out to conda env
./update.sh
# add conda-forge since many not found
# wait for long time on solving
# https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community
# ❯ conda --version
# conda 23.5.2
# um we already have it? but have to install?
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
# ok this seems to have helped
./compile.sh 
# ok and now this requirements.txt looks good
we are using the main deps from conda
should be ok
 python3 -m pip install -r requirements.txt

# Run notebook
# RuntimeError: espeak not installed on your system

after some hunting
already brew installed it
but need this:
https://github.com/bootphon/phonemizer/issues/117

PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib python3 styletts2_demo_libritts.py
# FileNotFoundError: [Errno 2] No such file or directory: 'Models/LibriTTS/config.yml'
❯ cp ~/Downloads/Models\ \(1\).zip ./Models1.zip

(no such file), '/usr/lib/liblapack.3.dylib' (no such file, not in dyld cache)

add scipy to environment
./update.sh
oops replace with scikit-learn

https://developer.apple.com/forums/thread/707316

add transformers to requirements.in, recompile, inistall
seems to almost run now! but need to unzip the refs again
unzip reference_audio.zip

PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib python3 styletts2_demo_libritts.py
mv reference_audio Demo/
NameError: name 'display' is not defined
# notebook stuff


# now test script mps to address wonky tensor

PYTORCH_ENABLE_MPS_FALLBACK=1 USE_MPS=1 python3 test_mps.py
