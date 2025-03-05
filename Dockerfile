
FROM rocm/dev-ubuntu-24.04:latest
USER root
RUN apt update && apt install -y software-properties-common wget curl python3 python3-pip python3.12-venv git
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER 1000
WORKDIR /app
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . \
    && python3 -m venv ./venv \ 
    && . ./venv/bin/activate \ 
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4 \
    && pip install -r requirements.txt

ADD root/start.sh /app/
RUN sudo chmod +x /app/start.sh
EXPOSE 8188
SHELL ["/bin/bash", "-c"]
CMD ./start.sh
