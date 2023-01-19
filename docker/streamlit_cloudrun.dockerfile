FROM python:3.9-slim
EXPOSE 8080

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY .streamlit/ .streamlit/
COPY streamlit/ streamlit/
COPY requirements.txt requirements.txt
COPY setup.py setup.py

WORKDIR /

RUN pip install --upgrade google-cloud
RUN pip install google-cloud-storage
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "streamlit/app.py", "--server.port=8080", "--server.address=0.0.0.0"]