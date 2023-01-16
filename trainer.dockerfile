# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# cope the essential parts of our application
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
# COPY models/ models/
# COPY reports/ reports/

# set working directory and install dependencies
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# run make data
# RUN make data

# set training to be the entry point
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]