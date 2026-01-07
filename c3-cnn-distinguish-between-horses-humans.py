import urllib.request
import zipfile


# Download the dataset - set the URL for the dataset
url = 'https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip'

file_name = 'horse-or-human.zip'
training_dir = 'horse-or-human/training/'

# Download the zip file
urllib.request.urlretrieve(url, file_name)

# Unzip the dataset
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall(training_dir)

zip_ref.close()
print("Dataset downloaded and extracted.")


