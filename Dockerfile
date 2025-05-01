FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/
RUN mkdir -p /app/.streamlit
EXPOSE 8501

CMD ["streamlit", "run", "second.py", "--server.address", "0.0.0.0", "--server.port", "8501"]