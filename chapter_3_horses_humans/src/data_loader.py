import os
try:
    from keras.preprocessing.image import ImageDataGenerator
except ImportError:
    from keras.src.legacy.preprocessing.image import ImageDataGenerator

def create_image_generator(dir_path, target_size=(300, 300), batch_size=32, class_mode='binary', fs=os, shuffle_param=True):
    """
    Load images from the specified directory using Keras ImageDataGenerator.

    Parameters:
    dir_path (str): Path to the directory containing images.
    target_size (tuple): Desired image size.
    batch_size (int): Number of images to be yielded from the generator per batch.

    Returns:
    image_generator: A Keras ImageDataGenerator yielding batches of images and labels.
    """
    if not fs.path.exists(dir_path):
            raise FileNotFoundError(f"Folder Not Found: {dir_path}")

    # Continue logic to load images...
    pass

    # Create an instance of ImageDataGenerator with rescaling
    image_datagen = ImageDataGenerator(rescale=1./255)

    # Create a generator that reads images from the directory
    image_generator = image_datagen.flow_from_directory(
        dir_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle_param
    )

    return image_generator
# Example usage:
# train_generator = create_image_generator('path/to/training/data')
# validation_generator = create_image_generator('path/to/validation/data')