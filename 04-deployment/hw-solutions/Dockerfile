FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip
RUN pip install pipenv

COPY Pipfile* .
RUN pipenv --python /usr/local/bin/python
RUN pipenv install --deploy --system

COPY hw4.py hw4.py

ENTRYPOINT [ "python", "hw4.py" ]