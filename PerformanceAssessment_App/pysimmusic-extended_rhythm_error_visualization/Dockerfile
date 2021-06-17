FROM mtg-docker.sb.upf.edu/kaldi

RUN apt-get update -y && apt-get install python3-pip python3-setuptools wget python3-tk libsndfile1-dev -y

RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code/

# PYIN Vamp plugin
WORKDIR /var/tmp
RUN mkdir pyin && cd pyin \
    && wget -q -O - https://code.soundsoftware.ac.uk/attachments/download/1461/pyin-v1.1-linux64.tar.gz | tar -xz \
    && mkdir -p /usr/local/lib/vamp && mv pyin.so /usr/local/lib/vamp \
    && cd ../ && rm -rf pyin

# NNLS-CHROMA Vamp plugin
RUN mkdir -p /usr/local/lib/vamp \
    && cd /usr/local/lib/vamp \
    && wget https://code.soundsoftware.ac.uk/attachments/download/1693/nnls-chroma-linux64-v1.1.tar.bz2 \
    && tar xvjf nnls-chroma-linux64-v1.1.tar.bz2 \
    && rm nnls-chroma-linux64-v1.1.tar.bz2 \
    && mv nnls-chroma-linux64-v1.1/* /usr/local/lib/vamp/ \
    && rm -r nnls-chroma-linux64-v1.1

# Lilypond
RUN wget http://lilypond.org/download/binaries/linux-64/lilypond-2.18.2-1.linux-64.sh \
    && chmod u+x lilypond-2.18.2-1.linux-64.sh \
    && ./lilypond-2.18.2-1.linux-64.sh \
    && rm lilypond-2.18.2-1.linux-64.sh

WORKDIR /code
RUN pip3 install -r requirements.txt --no-cache-dir \
    && pip3 install madmom --no-cache-dir \
    && pip3 install git+https://github.com/seffka/pychord_tools.git@v0.3#egg=pychord_tools --no-cache-dir

ADD pypirc /root/.pypirc

ADD . /code/

CMD ["python3", "setup.py", "bdist_wheel", "upload", "-r", "dev"]
