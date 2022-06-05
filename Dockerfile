FROM python
WORKDIR /ROOFINDER
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python","app.py"]




