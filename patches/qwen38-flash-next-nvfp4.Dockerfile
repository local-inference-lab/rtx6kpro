FROM vllm/vllm-openai:qwen38-flash-next

COPY patches/qwen38-flash-next-nvfp4-ple-fp8.patch /tmp/ple-fp8.patch
RUN cd /usr/local/lib/python3.12/dist-packages && patch -p1 < /tmp/ple-fp8.patch
