FROM python:3.12.7-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install pandas numpy scikit-learn matplotlib seaborn mlflow dagshub

CMD ["python", "MLProject/modelling.py"]