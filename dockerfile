FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /

# Needed because requirements installs diffusers from GitHub
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY rp_handler.py /rp_handler.py
CMD ["python", "-u", "/rp_handler.py"]
