import gc
import shutil
import unittest
import os

from pathlib import Path
from src.file_downloader import download_file as download_data

class TestDownloader(unittest.TestCase):
    def setUp(self):
        """Set up test variables"""
        self.test_dir = Path("test_data")
        self.test_file = self.test_dir / "logo.png"
        # Test URL for a small file
        self.sample_url = "https://raw.githubusercontent.com/tqdm/tqdm/master/images/logo.png"

    def tearDown(self):
        """Clean up after each test: remove file and folder"""
        # Force garbage collection to release file handles
        gc.collect()

        if self.test_file.exists():
            try:
                os.remove(self.test_file)
            except PermissionError:
                #shell to remove file 
                os.system(f"rm -f {self.test_file}")
        if self.test_dir.exists():
            #shell to remove directory
            shutil.rmtree(self.test_dir)

    def test_download_creates_file(self):
        """Verify if the file is downloaded and saved correctly"""
        download_data(self.sample_url, self.test_dir.as_posix())
        
        # Check if file exists
        self.assertTrue(self.test_file.exists(), "File should be created after download")
        
        # Check file size > 0
        self.assertGreater(os.path.getsize(self.test_file), 0, "Downloaded file should not be empty")

if __name__ == '__main__':
    unittest.main()