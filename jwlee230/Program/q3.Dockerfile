FROM antipd1-jwlee230:latest
LABEL maintainer="jwlee230@unist.ac.kr"

# ADD FILES
ADD Python /Python
ADD Output /Output

ENTRYPOINT ["python3", "/Python/q3-final.py"]
