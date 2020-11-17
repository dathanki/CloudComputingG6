FROM python:3.8

WORKDIR /cs4312-project-group6
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /cs4312-project-group6

EXPOSE 8080

CMD ["python","app.py"]
