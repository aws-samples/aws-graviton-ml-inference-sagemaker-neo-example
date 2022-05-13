FROM arm64v8/ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

ARG FUNCTION_DIR="/home/app/"
ARG MODEL_DIR="/home/model"

# Add model optimized using SageMaker Neo
ADD model ${MODEL_DIR}

RUN set -e \
    ### Install prerequisite packages
    && apt update \
    && apt install -y --no-install-recommends python3 python3-pip \
    ### Install Lambda runtime
    && mkdir -p ${FUNCTION_DIR} \
    && pip3 install awslambdaric --target ${FUNCTION_DIR} \
    ### Install required modules for ML inference code
    && pip3 install requests numpy Pillow dlr \
    ### Prepare ML model and labels
    && tar -xf ${MODEL_DIR}/*.tar.gz -C ${MODEL_DIR}

# Add custom ML inference code
COPY app.py ${FUNCTION_DIR}

WORKDIR ${FUNCTION_DIR}
ENTRYPOINT [ "/usr/bin/python3", "-m", "awslambdaric" ]
CMD [ "app.handler" ]
