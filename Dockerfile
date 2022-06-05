FROM python
WORKDIR /ROOFINDER
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python","app.py"]




