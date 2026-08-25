# Build as:
# docker build --tag hydrusbeta/hay_say:moss_tts_triton_server .

FROM nvcr.io/nvidia/tritonserver:26.07-py3
RUN apt update && apt install -y --no-install-recommends \
    git \
    vim \
    wget

# Switch to a limited user
ARG LIMITED_USER=luna
RUN useradd --create-home --shell /bin/bash $LIMITED_USER
USER $LIMITED_USER

# Some Docker directives (such as COPY and WORKDIR) and linux command options (such as wget's directory-prefix option)
# do not expand the tilde (~) character to /home/<user>, so define a temporary variable to use instead.
ARG HOME_DIR=/home/$LIMITED_USER

# Download the text tokenizer config and the main Moss TTS model
RUN mkdir -p ~/hay_say/temp_downloads/moss_tts_local_clipper_checkpoint/ && \
    wget https://huggingface.co/ZDisket/MOSS-TTS-PNY/resolve/main/moss_tts_local_clipper_checkpoint/tokenizer.json --directory-prefix=$HOME_DIR/hay_say/temp_downloads/moss_tts_local_clipper_checkpoint/ && \
    wget https://huggingface.co/ZDisket/MOSS-TTS-PNY/resolve/main/moss_tts_local_clipper_checkpoint/pytorch_model-00001-of-00002.bin --directory-prefix=$HOME_DIR/hay_say/temp_downloads/moss_tts_local_clipper_checkpoint/ && \
    wget https://huggingface.co/ZDisket/MOSS-TTS-PNY/resolve/main/moss_tts_local_clipper_checkpoint/pytorch_model-00002-of-00002.bin --directory-prefix=$HOME_DIR/hay_say/temp_downloads/moss_tts_local_clipper_checkpoint/

# Download the pretrained audio tokenizer models.
RUN mkdir -p ~/hay_say/temp_downloads/moss_audio_tokenizer/ && \
    wget https://huggingface.co/ZDisket/MOSS-TTS-PNY/resolve/main/moss_audio_tokenizer/model-00001-of-00002.safetensors --directory-prefix=$HOME_DIR/hay_say/temp_downloads/moss_audio_tokenizer/ && \
    wget https://huggingface.co/ZDisket/MOSS-TTS-PNY/resolve/main/moss_audio_tokenizer/model-00002-of-00002.safetensors --directory-prefix=$HOME_DIR/hay_say/temp_downloads/moss_audio_tokenizer/

# Download the pretrained Inverse Short-Time Fourier Transform vocoder
RUN mkdir -p ~/hay_say/temp_downloads/istftnet2_decoder4_50hz/ && \
    wget https://huggingface.co/ZDisket/MOSS-TTS-PNY/resolve/main/istftnet2_decoder4_50hz/istftnet2_decoder.onnx --directory-prefix=$HOME_DIR/hay_say/temp_downloads/istftnet2_decoder4_50hz/ && \
    wget https://huggingface.co/ZDisket/MOSS-TTS-PNY/resolve/main/istftnet2_decoder4_50hz/istftnet2_decoder_cpu.ts --directory-prefix=$HOME_DIR/hay_say/temp_downloads/istftnet2_decoder4_50hz/ && \
    wget https://huggingface.co/ZDisket/MOSS-TTS-PNY/resolve/main/istftnet2_decoder4_50hz/istftnet2_decoder_cuda.ts --directory-prefix=$HOME_DIR/hay_say/temp_downloads/istftnet2_decoder4_50hz/

# Install all python dependencies for the Hay Say interface code and for MOSS-TTS that are needed for inference.
# Clear the pip cache beforehand because the base tritonserver image comes with stuff in its cache that can interfere with pip install.
# Note: This is done *before* cloning the repository because the dependencies are likely to change less often than the
# MOSS-TTS code itself. Cloning the repo after installing the requirements helps the Docker cache optimize build time.
# See https://docs.docker.com/build/cache
RUN pip cache purge && \
    pip install \
    --timeout=300 \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128/ \
    gradio \
    huggingface-hub\<1.0 \
    jsonschema==4.25.1 \
    numpy\<2 \
    onnxruntime-gpu \
    safetensors \
    torch==2.11.0+cu128 \
    torchaudio==2.11.0+cu128 \
    transformers \
    triton

# Clone ZDisket's modified version of MOSS-TTS and checkout a specific commit that is known to work with this docker
# file and with Hay Say.
RUN git clone -b main --single-branch -q https://huggingface.co/ZDisket/MOSS-TTS-PNY ~/hay_say/moss_tts
WORKDIR $HOME_DIR/hay_say/moss_tts
RUN git reset --hard 74ac54a5ab89d1803b68a0a83450eb4682603f1d # Jun 30, 2026

# Edit a couple of files to remove a python feature that is not supported in python 3.9
RUN sed -i 's\| None\\'  ~/hay_say/moss_tts/moss_audio_tokenizer/configuration_moss_audio_tokenizer.py && \
    sed -i 's\| None\\'  ~/hay_say/moss_tts/moss_audio_tokenizer/modeling_moss_audio_tokenizer.py

# Remove a couple of lines that prevent the CPU torch_fp16 vocoder from loading
RUN sed -i '177,178d'  ~/hay_say/moss_tts/moss_tts_torchopt_runner_bundle/portable_tts_runtime.py

# Edit files as needed to enable Audio Top-k filtering
RUN sed -i '149i\ \ \ \ parser.add_argument("--audio-top-k", type=int, default=0)' ~/hay_say/moss_tts/moss_tts_torchopt_runner_bundle/run_tts_torchopt.py && \
    sed -i 's\audio_top_k=None,\audio_top_k=args.audio_top_k,\' ~/hay_say/moss_tts/moss_tts_torchopt_runner_bundle/run_tts_torchopt.py && \
    sed -i 's\audio_top_k=None,\audio_top_k=audio_top_k,\' ~/hay_say/moss_tts/moss_tts_torchopt_runner_bundle/portable_tts_runtime.py && \
    sed -i 's\if layer_config.get("top_k") is not None:\if layer_config.get("top_k") is not None and layer_config.get("top_k") != 0:\' ~/hay_say/moss_tts/moss_tts_local_clipper_checkpoint/modeling_moss_tts.py

# Delete LFS pointers and move the pretrained models to the expected directories.
RUN rm ~/hay_say/moss_tts/moss_tts_local_clipper_checkpoint/tokenizer.json && \
    rm ~/hay_say/moss_tts/istftnet2_decoder4_50hz/istftnet2_* &&\
    mv ~/hay_say/temp_downloads/istftnet2_decoder4_50hz/* ~/hay_say/moss_tts/istftnet2_decoder4_50hz && \
    mv ~/hay_say/temp_downloads/moss_tts_local_clipper_checkpoint/* ~/hay_say/moss_tts/moss_tts_local_clipper_checkpoint && \
    mv ~/hay_say/temp_downloads/moss_audio_tokenizer/* ~/hay_say/moss_tts/moss_audio_tokenizer && \
    rm -r ~/hay_say/temp_downloads/

# Clone the tritonserver code (this repo)
COPY --chown=$LIMITED_USER:$LIMITED_USER . $HOME_DIR/hay_say/tritonserver/
# Todo: use the following clone command instead if this repo goes public
# RUN git clone -b main --single-branch -q https://github.com/hydrusbeta/moss_tts_triton_server.git ~/hay_say/tritonserver
WORKDIR $HOME_DIR/hay_say/tritonserver

# Combine the two projects
RUN mv ~/hay_say/moss_tts/istftnet2_decoder4_50hz/ ~/hay_say/tritonserver/model_repository/moss_tts/1/ && \
	mv ~/hay_say/moss_tts/moss_audio_tokenizer/ ~/hay_say/tritonserver/model_repository/moss_tts/1/ && \
	mv ~/hay_say/moss_tts/moss_tts_local_clipper_checkpoint/ ~/hay_say/tritonserver/model_repository/moss_tts/1/ && \
	mv ~/hay_say/moss_tts/moss_tts_torchopt_runner_bundle/ ~/hay_say/tritonserver/model_repository/moss_tts/1/ && \
	rm -r ~/hay_say/moss_tts/

# Make a small correction to an import statement:
RUN sed -i 's\from decoder4_features_torch\from moss_tts_torchopt_runner_bundle.decoder4_features_torch\' ~/hay_say/tritonserver/model_repository/moss_tts/1/moss_tts_torchopt_runner_bundle/portable_tts_runtime.py

EXPOSE 8000

CMD ["tritonserver", "--model-repository=~/hay_say/tritonserver/model_repository/"]
