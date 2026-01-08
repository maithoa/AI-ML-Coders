import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from src.data_loader import create_image_generator

class TestImageGenerator(unittest.TestCase):

    @patch('src.data_loader.ImageDataGenerator')
    def test_create_image_generator_logic (self, MockIDG):
        """
        Config mock for file system
        """
        mock_fs = MagicMock()
        mock_fs.path.exists.return_value = True

        # Config the mock structure to simulate the return value of Keras Mock ImageDataGenerator
        mock_instance = MockIDG.return_value
        mock_gen = MagicMock()
        mock_instance.flow_from_directory.return_value = mock_gen

        # Call the function to test
        from src.data_loader import create_image_generator

        fake_dir = "fake_directory"
        target_size = (150, 150)
        batch_size = 64

        result_gen = create_image_generator(
            fake_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            fs=mock_fs
        )

        # Assertions to verify correct calls and parameters

        # Check if create_image_generator checked for directory existence
        mock_fs.path.exists.assert_called_with(fake_dir)

        # Check if ImageDataGenerator was instantiated with rescale parameter
        MockIDG.assert_called_with(rescale=1./255)

        # Check if flow_from_directory was called with correct parameters
        mock_instance.flow_from_directory.assert_called_with(
            fake_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Check if the returned generator is the mock generator
        self.assertEqual(result_gen, mock_gen)

if __name__ == '__main__':
    unittest.main()