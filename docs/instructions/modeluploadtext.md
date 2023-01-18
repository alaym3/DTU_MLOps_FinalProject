build image for predict.dockerfile

<!-- Steps in order -->
docker build --platform linux/amd64 -f predict.dockerfile . -t predict:latest
docker tag predict gcr.io/final-project-374512/predict
docker push gcr.io/final-project-374512/predict

gcloud ai models upload \
  --region=europe-west1 \
  --display-name=text-classification \
  --container-image-uri=gcr.io/final-project-374512/predict \
  --artifact-uri='gs://dtu_mlops_final_model'