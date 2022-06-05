FROM python:3.8.13
WORKDIR /ROOFINDER
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python","app.py"]




