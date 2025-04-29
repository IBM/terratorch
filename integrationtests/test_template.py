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

def test_eo_v2_100_burns():
    # Call the CLI program
    result = subprocess.run(['terratorch', 'fit', '-c', 'test_encoderdecoder_eo_v2_100_model_factory.yaml'], capture_output=True, text=True)
    
    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # Check the return code
    assert result.returncode == 0, f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_eo_v2_300_burns():
    # Call the CLI program
    result = subprocess.run(['terratorch', 'fit', '-c', 'test_encoderdecoder_eo_v2_300_model_factory.yaml'], capture_output=True, text=True)
    
    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # Check the return code
    assert result.returncode == 0, f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"



def test_e0_v2_600_burns():
    # Call the CLI program
    result = subprocess.run(['terratorch', 'fit', '-c', 'test_encoderdecoder_eo_v2_600_model_factory.yaml'], capture_output=True, text=True)
    
    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # Check the return code
    assert result.returncode == 0, f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

def test_prithvi_swinl_burns():
    # Call the CLI program
    result = subprocess.run(['terratorch', 'fit', '-c', 'test_prithvi_swinL_model_factory_config.yaml'], capture_output=True, text=True)
    
    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # Check the return code
    assert result.returncode == 0, f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    

def test_prithvi_swinb_burns():
    # Call the CLI program
    result = subprocess.run(['terratorch', 'fit', '-c', 'test_prithvi_swinB_model_factory_config.yaml'], capture_output=True, text=True)
    
    # Print the captured output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # Check the return code
    assert result.returncode == 0, f"Test failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

def test_resnet_enc_dec_builds():
    # TODO: Copy data to ccc and update the tests
    assert True

def test_resnet_smp_builds():
    # TODO: Copy data to ccc and update the tests
    assert True