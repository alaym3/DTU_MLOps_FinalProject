1. build an image from the trainer.dockerfile that we have previously built in our mnist repo
2. tag and push the image to our project in gcp using docker tag and docker push
3. build config.yaml pointing to our project-id and newly made image - image must match
4. run gcloud ai custom-jobs create (as in 3b)

<!-- Steps in order -->
docker build --platform linux/amd64 -f trainer.dockerfile . -t trainerc:latest
docker tag trainerc gcr.io/final-project-374512/trainerc
docker push gcr.io/final-project-374512/trainerc

gcloud ai custom-jobs create \
   --region=europe-west1 \
   --display-name=test-run-c \
   --config=config.yaml

<!-- try copying src ./src -->

<!-- docker build --platform linux/amd64  . -->
