import pytest
import subprocess

def test_template():
    # Call the CLI program
    result = subprocess.run(['terratorch', 'fit', '-c', 'test_template_config.yaml'], capture_output=True, text=True)
    
    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # Check the return code
    assert result.returncode == 0, f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
