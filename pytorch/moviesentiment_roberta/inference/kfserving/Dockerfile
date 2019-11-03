ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:19.07-py3
FROM $BASE_IMAGE
ARG BASE_IMAGE
RUN echo "Installing Apex on top of ${BASE_IMAGE}"
# make sure we don't overwrite some existing directory called "apex"
WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /workspace

RUN chmod -R a+w /workspace

# RUN git clone https://github.com/kubeflow/kfserving.git && \
RUN git clone --single-branch --branch server_workers https://github.com/cliveseldon/kfserving && \
    cd kfserving/python && \
    pip install --upgrade pip && pip install -e ./kfserving

COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "model.py"]

