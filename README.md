## Conda Installation

``` bash
# 1. Create a conda virtual environment.
conda create -n PIDM python=3.8
conda activate PIDM
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# 2. Clone the Repo and Install dependencies
git clone https://github.com/mengoat/PIDM
pip install -r requirements.txt

```

## Web Start
``` bash
# 1. Get into the workdir.
cd PIDM

# 2. Start the web.
python start_web.py 
# 3. Running on local URL:  http://127.0.0.1:7860
```


