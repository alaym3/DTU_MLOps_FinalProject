FROM python:3.9-slim
EXPOSE 8080

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY .streamlit/ .streamlit/
COPY models/ models/
COPY app/ app/
COPY requirements.txt requirements.txt

WORKDIR /

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]