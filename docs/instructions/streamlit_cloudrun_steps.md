[steps](https://medium.com/@faizififita1/how-to-deploy-your-streamlit-web-app-to-google-cloud-run-ba776487c5fe)

# THESE STEPS WORK TO DEPLOY A CLOUD RUN
docker build --platform linux/amd64 -f streamlit.dockerfile . -t streamlit:latest
docker tag streamlit:latest gcr.io/final-project-374512/streamlit
docker push gcr.io/final-project-374512/streamlit
gcloud run deploy streamlit --image gcr.io/final-project-374512/streamlit


# testing image locally:
docker build -f streamlit.dockerfile . -t streamlit:latest
docker run -p 8080:8080 streamlit:latest

# tag and push to the project (not needed for this)
docker tag streamlit gcr.io/final-project-374512/streamlit
docker push gcr.io/final-project-374512/streamlit

# tag and build a new container image automatically using a Dockerfile - no other setup needed
gcloud builds submit --tag gcr.io/final-project-374512/streamlit --timeout=2h
gcloud builds submit --tag gcr.io/<PROJECT_ID>/<SOME_PROJECT_NAME> --timeout=2h


# alternative to directly build the run
docker build -f streamlit.dockerfile . -t streamlit:latest
docker tag streamlit gcr.io/final-project-374512/streamlit
docker push gcr.io/final-project-374512/streamlit
gcloud run deploy streamlit --image gcr.io/final-project-374512/streamlit


# alternative to auto deploy cloud run, with linux..
docker build --platform linux/amd64 -f streamlit.dockerfile . -t streamlit:latest
docker tag streamlit:latest gcr.io/final-project-374512/streamlit
docker push gcr.io/final-project-374512/streamlit
gcloud run deploy streamlit --image gcr.io/final-project-374512/streamlit



# if the alternative doesn't work, add thsi to dockerfile
ENV PATH=“${PATH}:/root/.local/bin” 