FROM python:3.9.1
LABEL maintainer="jwlee230@unist.ac.kr"

# Basic setting
RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
RUN apt-get update && apt-get upgrade -y

# PIP
ADD requirements.txt /
RUN pip install --requirement /requirements.txt

# Add Volumes
VOLUME /Python
VOLUME /Data /data
VOLUME /Output /output

CMD ["/bin/bash"]
