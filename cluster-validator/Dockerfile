FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libibverbs1 \
        ibverbs-providers && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cluster-validator

COPY validator/ /opt/cluster-validator/validator/
