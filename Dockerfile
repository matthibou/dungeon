FROM python:3.10


RUN pip install torch torchvision torchaudio \
                   --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
