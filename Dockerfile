FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt # tells docker not to save packages as cache on disk

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]