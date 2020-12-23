FROM antipd1-jwlee230:latest
LABEL maintainer="jwlee230@unist.ac.kr"

# ADD FILES
ADD Python /
ADD Output /

ENTRYPOINT ["python3", "/Python/q3-final.py"]
