FROM python:3.9.1
LABEL maintainer="jwlee230@unist.ac.kr"

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

RUN apt-get update && apt-get upgrade -y

CMD ["/bin/bash"]
