# Use an official Python runtime as a parent image
# FROM python:3.10-bullseye



# ENTRYPOINT [ "bash" ]

FROM library/python:3.11-bullseye

ARG USER=vscode
ARG DEBIAN_FRONTEND=noninteractive
# RUN apt update \
#     && apt install -y --no-install-recommends sudo \
#     && apt autoremove -y \
#     && rm -rf /var/lib/apt/lists/* \
#     && useradd -m -s /usr/bin/bash ${USER} \
#     && echo "${USER} ALL=(ALL) NOPASSWD: ALL" >/etc/sudoers.d/${USER} \
#     && chmod 0440 /etc/sudoers.d/${USER}

RUN mkdir -p /home/usr/code
# # Set the working directory in the container
WORKDIR /home/usr/code

ENV PATH="/usr/local/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

USER root

# I need environment variables for elastic
