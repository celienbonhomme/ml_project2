FROM python:3.9

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY src /src
COPY data /data

CMD ["python", "/src/app.py"]