
FROM python:3.11-slim AS app

ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/app \
    PATH="${PATH}:/app/.local/bin"


WORKDIR /app
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential
RUN apt install git -y 
#RUN git clone https://github.com/abacaj/mpt-30B-inference.git .
COPY requirements.txt /app/requirements.txt
# I don't want to do this, but im having trouble mounting the volumes for some reasons
COPY models2/mpt-30b-chat.ggmlv0.q4_1.bin /app/models/mpt-30b-chat.ggmlv0.q4_1.bin
#COPY models2/mpt-30b-chat.ggmlv0.q8_0.bin /app/models/mpt-30b-chat.ggmlv0.q8_0.bin

RUN pip install -r requirements.txt
COPY inference.py /app/inference.py
CMD [ "python", "inference.py" ]`