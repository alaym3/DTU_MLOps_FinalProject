1. build an image from the trainer.dockerfile that we have previously built in our mnist repo
2. tag and push the image to our project in gcp using docker tag and docker push
3. build config.yaml pointing to our project-id and newly made image - image must match
4. run gcloud ai custom-jobs create (as in 3b)

<!-- Steps in order -->
docker build --platform linux/amd64 -f trainer.dockerfile . -t trainerd:latest
docker tag trainerd gcr.io/final-project-374512/trainerd
docker push gcr.io/final-project-374512/trainerd

gcloud ai custom-jobs create \
   --region=europe-west1 \
   --display-name=test-run-d-gpu \
   --config=config_gpu.yaml

<!-- try copying src ./src -->

<!-- docker build --platform linux/amd64  . -->



gpu:
# config_gpu.yaml
workerPoolSpecs:
   machineSpec:
      machineType: n1-standard-8
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
   replicaCount: 1
   containerSpec:
      imageUri: gcr.io/<project-id>/<docker-img>
