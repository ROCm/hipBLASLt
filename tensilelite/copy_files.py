import shutil  
import sys  
from pathlib import Path  
  
def copy_files(source_dir, dest_dir, extensions):  
    source = Path(source_dir)  
    destination = Path(dest_dir)  
 
    for ext in extensions:  
        for file_path in source.rglob(f"*{ext}"):  
            try:  
                shutil.copy2(file_path, destination)  
                print(f"Copied {file_path} to {destination}")  
            except Exception as e:  
                print(f"Failed to copy {file_path}: {e}", file=sys.stderr)  
  
if __name__ == "__main__":        
    source_directory = sys.argv[1]  
    destination_directory = sys.argv[2]  
    file_extensions = ['.co', '.dat']  
      
    copy_files(source_directory, destination_directory, file_extensions) 