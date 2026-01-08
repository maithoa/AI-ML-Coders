from model_factory import create_model
from train import train_model
from data_loader import create_image_generator 
from file_downloader import download_file
import zipfile
import keras.models 
import os
from pathlib import Path
# Assuming you have these from your previous steps
# from utils import dataloader, file_downloader

def run():
    # Step 1: Prepare data 
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    # Download Train and Validation Data if not already done
    train_set_url = 'https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip'
    training_dir = project_root / 'data' / 'training'
    
    validation_set_url = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    validation_dir = project_root / 'data' / 'validation'

    save_path = project_root / 'saved_models' / 'horse_human_classifier_model.keras'
    save_dir = os.path.dirname(save_path)

    if not os.path.exists(save_dir):
        print(f"--- Creating directory: {save_dir} ---")
        os.makedirs(save_dir)

    # Download the zip file
    train_file_name = download_file(train_set_url, training_dir)

    # Unzip the dataset
    with zipfile.ZipFile(train_file_name, 'r') as zip_ref:
        zip_ref.extractall(training_dir)
 

    # Repeat for validation data
    val_file_name = download_file(validation_set_url, validation_dir)
    with zipfile.ZipFile(val_file_name, 'r') as zip_ref:
        zip_ref.extractall(validation_dir)
 
    print("Datasets downloaded and extracted.")

    # Create Image Generators
    train_gen = create_image_generator(training_dir, target_size=(300, 300), batch_size=32, class_mode='binary', shuffle_param=True) 
    val_gen = create_image_generator(validation_dir, target_size=(300, 300), batch_size=32, class_mode='binary', shuffle_param=False) 
   
    # Step 2: Initialize the model architecture
    my_model = create_model(input_shape_param=(300, 300, 3))

    # Step 3: Define hyperparameters for a quick test  
    params = {
        'learning_rate': 0.0001,
        'epochs': 20,
        'steps_per_epoch': 32,
        'validation_steps': 8
    }

    # Step 3: Start training and get history
    print("Initiating training sequence...")
    training_log = train_model(
        model=my_model,
        train_data=train_gen, 
        val_data=val_gen,
        hyperparams=params
    )
    
    # Step 4: Export the model for inference later


    print(f"--- Saving model to {save_path} ---")
    keras.models.save_model(my_model, save_path)
    print ("training_log:", training_log.history)
    print("Model saved successfully.")

if __name__ == "__main__":
    run()