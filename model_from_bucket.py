import os
from google.cloud import storage
from google.cloud import language
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file("creds/creds.json")
client = language.LanguageServiceClient(credentials=credentials)




# function to list items in a bucket
def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client(credentials=credentials)

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    # Note: The call returns a response only when the iterator is consumed.
    # for blob in blobs:
    #     print(blob.name, blob.id)
    names = []
    ids = []
    for blob in blobs:
        names.append(blob.name)
        ids.append(blob.id)

    return names, ids


# function to download items from a bucket to local
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client(credentials=credentials)

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )


BUCKET_NAME = "dtu_mlops_final_model"  # name of bucket where model is stored
names, ids = list_blobs(BUCKET_NAME)  # retrieving names and ids of items in bucket
dest = "models/"  # local destination where files should be stored

for file, id in zip(names, ids):
    # print(file, id)
    download_blob(BUCKET_NAME, file, os.path.join(dest, file))  # downloading files
