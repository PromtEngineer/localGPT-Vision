FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04 AS app

ARG ARG_USERNAME="app"
ARG ARG_USER_UID=1337
ARG ARG_USER_GID=$ARG_USER_UID
ARG ARG_WORKSPACE_ROOT="/app"

ENV DEBIAN_FRONTEND=noninteractive
ENV USERNAME $ARG_USERNAME
ENV USER_UID $ARG_USER_UID
ENV USER_GID $ARG_USER_GID
ENV WORKSPACE_ROOT $ARG_WORKSPACE_ROOT

RUN \
  groupadd --gid $USER_GID $USERNAME && \
  adduser --uid $USER_UID --gid $USER_GID $USERNAME

RUN apt-get update && \
  apt-get install -y python3 python-is-python3 python3-pip git curl && \
  rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 https://github.com/PromtEngineer/localGPT-Vision $WORKSPACE_ROOT && \
  rm -rf $WORKSPACE_ROOT/.git

RUN sed -i 's/port=5050, debug=True/port=5050, debug=False, host="0.0.0.0"/g' $WORKSPACE_ROOT/app.py

VOLUME [ "$WORKSPACE_ROOT/.byaldi", "$WORKSPACE_ROOT/uploaded_documents", "$WORKSPACE_ROOT/sessions", "/home/app/.cache/huggingface" ]

RUN chown -R $USERNAME:$USERNAME $WORKSPACE_ROOT

USER $USERNAME
WORKDIR $WORKSPACE_ROOT

RUN pip install --upgrade pip setuptools wheel \
  && pip install -r requirements.txt hf_transfer && \
  pip install git+https://github.com/huggingface/transformers

HEALTHCHECK --interval=2m --timeout=3s --start-period=10s --retries=3 \
  CMD curl -q -f http://localhost:5050/ || exit 1

EXPOSE 5050

CMD ["python", "app.py"]
