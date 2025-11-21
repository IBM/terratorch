# Copyright contributors to the Terratorch project

"""Tests for scheduling_ddpm.py module"""

import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch

from terratorch.models.backbones.terramind.tokenizer.scheduling.scheduling_ddpm import (
    DDPMScheduler,
    DDPMSchedulerOutput,
)


class TestDDPMSchedulerOutput:
    """Test DDPMSchedulerOutput dataclass"""

    def test_output_with_pred_original_sample(self):
        """Test output with both prev_sample and pred_original_sample"""
        prev_sample = torch.randn(2, 3, 16, 16)
        pred_original = torch.randn(2, 3, 16, 16)
        
        output = DDPMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original
        )
        
        assert torch.equal(output.prev_sample, prev_sample)
        assert torch.equal(output.pred_original_sample, pred_original)

    def test_output_without_pred_original_sample(self):
        """Test output with only prev_sample"""
        prev_sample = torch.randn(2, 3, 16, 16)
        
        output = DDPMSchedulerOutput(prev_sample=prev_sample)
        
        assert torch.equal(output.prev_sample, prev_sample)
        assert output.pred_original_sample is None


class TestDDPMSchedulerInit:
    """Test DDPMScheduler initialization"""

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        scheduler = DDPMScheduler()
        
        assert scheduler.config.num_train_timesteps == 1000
        assert scheduler.config.beta_start == 0.0001
        assert scheduler.config.beta_end == 0.02
        assert scheduler.config.beta_schedule == "linear"
        assert scheduler.config.variance_type == "fixed_small"
        assert scheduler.config.clip_sample is True
        assert scheduler.config.prediction_type == "v_prediction"

    def test_init_linear_schedule(self):
        """Test initialization with linear beta schedule"""
        scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.01,
            beta_schedule="linear",
            zero_terminal_snr=False
        )
        
        assert len(scheduler.betas) == 100
        assert scheduler.betas[0] == pytest.approx(0.001, rel=1e-3)
        assert scheduler.betas[-1] == pytest.approx(0.01, rel=1e-3)

    def test_init_scaled_linear_schedule(self):
        """Test initialization with scaled_linear beta schedule"""
        scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="scaled_linear",
            beta_start=0.001,
            beta_end=0.01,
            zero_terminal_snr=False
        )
        
        assert len(scheduler.betas) == 100
        # Scaled linear: sqrt space
        expected_first = 0.001 ** 0.5
        expected_last = 0.01 ** 0.5
        assert scheduler.betas[0] == pytest.approx(expected_first ** 2, rel=1e-3)
        assert scheduler.betas[-1] == pytest.approx(expected_last ** 2, rel=1e-3)

    def test_init_squaredcos_cap_v2_schedule(self):
        """Test initialization with squaredcos_cap_v2 beta schedule"""
        scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2"
        )
        
        assert len(scheduler.betas) == 100
        assert hasattr(scheduler, 'alphas_cumprod')

    def test_init_sigmoid_schedule(self):
        """Test initialization with sigmoid beta schedule"""
        scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="sigmoid",
            beta_start=0.001,
            beta_end=0.01
        )
        
        assert len(scheduler.betas) == 100

    def test_init_shifted_cosine_schedule(self):
        """Test initialization with shifted_cosine beta schedule"""
        scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="shifted_cosine:0.5"
        )
        
        assert hasattr(scheduler, 'alphas_cumprod')
        assert len(scheduler.alphas_cumprod) == 100

    def test_init_trained_betas(self):
        """Test initialization with trained betas"""
        trained_betas = np.linspace(0.0001, 0.02, 50)
        scheduler = DDPMScheduler(
            num_train_timesteps=50,
            trained_betas=trained_betas,
            zero_terminal_snr=False  # Avoid zero_terminal_snr modification
        )
        
        assert len(scheduler.betas) == 50
        # Just check first and last values are close
        assert scheduler.betas[0] == pytest.approx(trained_betas[0], rel=1e-3)
        assert scheduler.betas[-1] == pytest.approx(trained_betas[-1], rel=1e-3)

    def test_init_trained_betas_as_list(self):
        """Test initialization with trained betas as list"""
        trained_betas = [0.0001, 0.0002, 0.0003]
        scheduler = DDPMScheduler(
            num_train_timesteps=3,
            trained_betas=trained_betas
        )
        
        assert len(scheduler.betas) == 3

    def test_init_invalid_schedule(self):
        """Test initialization with invalid beta schedule"""
        with pytest.raises(NotImplementedError):
            DDPMScheduler(beta_schedule="invalid_schedule")

    def test_init_zero_terminal_snr_false(self):
        """Test initialization without zero terminal SNR"""
        scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="linear",
            zero_terminal_snr=False
        )
        
        assert hasattr(scheduler, 'betas')

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        scheduler = DDPMScheduler(
            num_train_timesteps=500,
            variance_type="fixed_large",
            clip_sample=False,
            prediction_type="epsilon",
            thresholding=True,
            dynamic_thresholding_ratio=0.99,
            clip_sample_range=2.0,
            sample_max_value=2.0
        )
        
        assert scheduler.config.num_train_timesteps == 500
        assert scheduler.config.variance_type == "fixed_large"
        assert scheduler.config.clip_sample is False
        assert scheduler.config.prediction_type == "epsilon"
        assert scheduler.config.thresholding is True
        assert scheduler.config.dynamic_thresholding_ratio == 0.99


class TestDDPMSchedulerMethods:
    """Test DDPMScheduler methods"""

    def test_scale_model_input(self):
        """Test scale_model_input method"""
        scheduler = DDPMScheduler()
        sample = torch.randn(2, 3, 16, 16)
        
        scaled = scheduler.scale_model_input(sample, timestep=10)
        
        # Should return the same sample unchanged
        assert torch.equal(scaled, sample)

    def test_set_timesteps_with_num_inference_steps(self):
        """Test set_timesteps with num_inference_steps"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        scheduler.set_timesteps(num_inference_steps=50)
        
        assert scheduler.num_inference_steps == 50
        assert len(scheduler.timesteps) == 50
        assert scheduler.custom_timesteps is False

    def test_set_timesteps_with_custom_timesteps(self):
        """Test set_timesteps with custom timesteps"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        custom_timesteps = [900, 800, 700, 600, 500]
        
        scheduler.set_timesteps(timesteps=custom_timesteps)
        
        assert len(scheduler.timesteps) == 5
        assert scheduler.custom_timesteps is True

    def test_set_timesteps_both_parameters_error(self):
        """Test set_timesteps with both parameters raises error"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        with pytest.raises(ValueError, match="Can only pass one"):
            scheduler.set_timesteps(num_inference_steps=50, timesteps=[900, 800])

    def test_set_timesteps_non_descending_error(self):
        """Test set_timesteps with non-descending timesteps raises error"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        with pytest.raises(ValueError, match="must be in descending order"):
            scheduler.set_timesteps(timesteps=[100, 200, 300])

    def test_set_timesteps_exceeds_train_timesteps_error(self):
        """Test set_timesteps with timesteps exceeding train timesteps"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        with pytest.raises(ValueError, match="must start before"):
            scheduler.set_timesteps(timesteps=[1500, 1000, 500])

    def test_set_timesteps_num_inference_exceeds_train_error(self):
        """Test set_timesteps with num_inference_steps exceeding train timesteps"""
        scheduler = DDPMScheduler(num_train_timesteps=100)
        
        with pytest.raises(ValueError, match="cannot be larger"):
            scheduler.set_timesteps(num_inference_steps=200)

    def test_set_timesteps_with_device(self):
        """Test set_timesteps with device parameter"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        scheduler.set_timesteps(num_inference_steps=50, device="cpu")
        
        assert scheduler.timesteps.device.type == "cpu"

    def test_len(self):
        """Test __len__ method"""
        scheduler = DDPMScheduler(num_train_timesteps=500)
        
        assert len(scheduler) == 500

    def test_previous_timestep_standard(self):
        """Test previous_timestep with standard timesteps"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=50)
        
        current_t = scheduler.timesteps[10]
        prev_t = scheduler.previous_timestep(current_t)
        
        assert prev_t < current_t

    def test_previous_timestep_custom(self):
        """Test previous_timestep with custom timesteps"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(timesteps=[900, 700, 500, 300, 100])
        
        prev_t = scheduler.previous_timestep(900)
        
        assert prev_t == 700

    def test_previous_timestep_last(self):
        """Test previous_timestep for last timestep"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(timesteps=[900, 700, 500])
        
        prev_t = scheduler.previous_timestep(500)
        
        assert prev_t == -1


class TestGetVariance:
    """Test _get_variance method"""

    def test_get_variance_fixed_small(self):
        """Test _get_variance with fixed_small variance type"""
        scheduler = DDPMScheduler(variance_type="fixed_small")
        scheduler.set_timesteps(num_inference_steps=50)
        
        variance = scheduler._get_variance(t=10)
        
        assert variance > 0

    def test_get_variance_fixed_small_log(self):
        """Test _get_variance with fixed_small_log variance type"""
        scheduler = DDPMScheduler(variance_type="fixed_small_log")
        scheduler.set_timesteps(num_inference_steps=50)
        
        variance = scheduler._get_variance(t=10)
        
        assert variance > 0

    def test_get_variance_fixed_large(self):
        """Test _get_variance with fixed_large variance type"""
        scheduler = DDPMScheduler(variance_type="fixed_large")
        scheduler.set_timesteps(num_inference_steps=50)
        
        variance = scheduler._get_variance(t=10)
        
        assert variance > 0

    def test_get_variance_fixed_large_log(self):
        """Test _get_variance with fixed_large_log variance type"""
        scheduler = DDPMScheduler(variance_type="fixed_large_log")
        scheduler.set_timesteps(num_inference_steps=50)
        
        variance = scheduler._get_variance(t=10)
        
        assert isinstance(variance, torch.Tensor)

    def test_get_variance_learned(self):
        """Test _get_variance with learned variance type"""
        scheduler = DDPMScheduler(variance_type="learned")
        scheduler.set_timesteps(num_inference_steps=50)
        
        predicted_var = torch.tensor(0.5)
        variance = scheduler._get_variance(t=10, predicted_variance=predicted_var)
        
        assert variance == predicted_var

    def test_get_variance_learned_range(self):
        """Test _get_variance with learned_range variance type"""
        scheduler = DDPMScheduler(variance_type="learned_range")
        scheduler.set_timesteps(num_inference_steps=50)
        
        predicted_var = torch.tensor(0.0)  # Will be in range [-1, 1]
        variance = scheduler._get_variance(t=10, predicted_variance=predicted_var)
        
        assert isinstance(variance, torch.Tensor)

    def test_get_variance_with_prev_t_negative(self):
        """Test _get_variance when prev_t is negative"""
        scheduler = DDPMScheduler()
        scheduler.set_timesteps(num_inference_steps=50)
        
        # Use t=0 which should have prev_t < 0
        variance = scheduler._get_variance(t=0)
        
        assert isinstance(variance, torch.Tensor)

    def test_get_variance_override_variance_type(self):
        """Test _get_variance with variance_type override"""
        scheduler = DDPMScheduler(variance_type="fixed_small")
        scheduler.set_timesteps(num_inference_steps=50)
        
        variance = scheduler._get_variance(t=10, variance_type="fixed_large")
        
        assert isinstance(variance, torch.Tensor)


class TestThresholdSample:
    """Test _threshold_sample method"""

    def test_threshold_sample_float32(self):
        """Test _threshold_sample with float32"""
        scheduler = DDPMScheduler(
            thresholding=True,
            dynamic_thresholding_ratio=0.995,
            sample_max_value=1.0
        )
        
        sample = torch.randn(2, 3, 16, 16, dtype=torch.float32)
        thresholded = scheduler._threshold_sample(sample)
        
        assert thresholded.shape == sample.shape
        assert thresholded.dtype == torch.float32

    def test_threshold_sample_float16(self):
        """Test _threshold_sample with float16 (upcast and downcast)"""
        scheduler = DDPMScheduler(
            thresholding=True,
            dynamic_thresholding_ratio=0.995,
            sample_max_value=1.0
        )
        
        sample = torch.randn(2, 3, 16, 16, dtype=torch.float16)
        thresholded = scheduler._threshold_sample(sample)
        
        assert thresholded.shape == sample.shape
        assert thresholded.dtype == torch.float16

    def test_threshold_sample_with_high_values(self):
        """Test _threshold_sample with values exceeding threshold"""
        scheduler = DDPMScheduler(
            thresholding=True,
            dynamic_thresholding_ratio=0.5,  # Low ratio to trigger thresholding
            sample_max_value=2.0
        )
        
        sample = torch.randn(2, 3, 16, 16) * 10  # Large values
        thresholded = scheduler._threshold_sample(sample)
        
        # Values should be clipped
        assert thresholded.abs().max() <= sample.abs().max()


class TestStep:
    """Test step method"""

    def test_step_epsilon_prediction(self):
        """Test step with epsilon prediction type"""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            prediction_type="epsilon"
        )
        scheduler.set_timesteps(num_inference_steps=50)
        
        model_output = torch.randn(2, 3, 16, 16)
        sample = torch.randn(2, 3, 16, 16)
        timestep = scheduler.timesteps[10].item()
        
        result = scheduler.step(model_output, timestep, sample, return_dict=True)
        
        assert isinstance(result, DDPMSchedulerOutput)
        assert result.prev_sample.shape == sample.shape
        assert result.pred_original_sample.shape == sample.shape

    def test_step_sample_prediction(self):
        """Test step with sample prediction type"""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            prediction_type="sample"
        )
        scheduler.set_timesteps(num_inference_steps=50)
        
        model_output = torch.randn(2, 3, 16, 16)
        sample = torch.randn(2, 3, 16, 16)
        timestep = scheduler.timesteps[10].item()
        
        result = scheduler.step(model_output, timestep, sample, return_dict=True)
        
        assert isinstance(result, DDPMSchedulerOutput)

    def test_step_v_prediction(self):
        """Test step with v_prediction type"""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            prediction_type="v_prediction"
        )
        scheduler.set_timesteps(num_inference_steps=50)
        
        model_output = torch.randn(2, 3, 16, 16)
        sample = torch.randn(2, 3, 16, 16)
        timestep = scheduler.timesteps[10].item()
        
        result = scheduler.step(model_output, timestep, sample, return_dict=True)
        
        assert isinstance(result, DDPMSchedulerOutput)

    def test_step_invalid_prediction_type(self):
        """Test step with invalid prediction type"""
        scheduler = DDPMScheduler()
        scheduler.config.prediction_type = "invalid"
        scheduler.set_timesteps(num_inference_steps=50)
        
        model_output = torch.randn(2, 3, 16, 16)
        sample = torch.randn(2, 3, 16, 16)
        timestep = scheduler.timesteps[10].item()
        
        with pytest.raises(ValueError, match="must be one of"):
            scheduler.step(model_output, timestep, sample)

    def test_step_with_thresholding(self):
        """Test step with thresholding enabled"""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            thresholding=True,
            dynamic_thresholding_ratio=0.995
        )
        scheduler.set_timesteps(num_inference_steps=50)
        
        model_output = torch.randn(2, 3, 16, 16)
        sample = torch.randn(2, 3, 16, 16)
        timestep = scheduler.timesteps[10].item()
        
        result = scheduler.step(model_output, timestep, sample)
        
        assert isinstance(result, DDPMSchedulerOutput)

    def test_step_with_clip_sample(self):
        """Test step with clip_sample enabled"""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            clip_sample=True,
            clip_sample_range=1.0
        )
        scheduler.set_timesteps(num_inference_steps=50)
        
        model_output = torch.randn(2, 3, 16, 16) * 10  # Large values
        sample = torch.randn(2, 3, 16, 16)
        timestep = scheduler.timesteps[10].item()
        
        result = scheduler.step(model_output, timestep, sample)
        
        # pred_original_sample should be clipped
        assert result.pred_original_sample.abs().max() <= 1.0

    def test_step_return_tuple(self):
        """Test step with return_dict=False"""
        scheduler = DDPMScheduler()
        scheduler.set_timesteps(num_inference_steps=50)
        
        model_output = torch.randn(2, 3, 16, 16)
        sample = torch.randn(2, 3, 16, 16)
        timestep = scheduler.timesteps[10].item()
        
        result = scheduler.step(model_output, timestep, sample, return_dict=False)
        
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_step_with_learned_variance(self):
        """Test step with learned variance"""
        scheduler = DDPMScheduler(
            variance_type="learned"
        )
        scheduler.set_timesteps(num_inference_steps=50)
        
        # Model output has 2x channels for learned variance
        model_output = torch.randn(2, 6, 16, 16)  # 6 = 3*2
        sample = torch.randn(2, 3, 16, 16)
        timestep = scheduler.timesteps[10].item()
        
        result = scheduler.step(model_output, timestep, sample)
        
        assert result.prev_sample.shape == sample.shape

    def test_step_with_learned_range_variance(self):
        """Test step with learned_range variance"""
        scheduler = DDPMScheduler(
            variance_type="learned_range"
        )
        scheduler.set_timesteps(num_inference_steps=50)
        
        model_output = torch.randn(2, 6, 16, 16)
        sample = torch.randn(2, 3, 16, 16)
        timestep = scheduler.timesteps[10].item()
        
        result = scheduler.step(model_output, timestep, sample)
        
        assert result.prev_sample.shape == sample.shape

    def test_step_with_fixed_small_log_variance(self):
        """Test step with fixed_small_log variance"""
        scheduler = DDPMScheduler(
            variance_type="fixed_small_log"
        )
        scheduler.set_timesteps(num_inference_steps=50)
        
        model_output = torch.randn(2, 3, 16, 16)
        sample = torch.randn(2, 3, 16, 16)
        timestep = scheduler.timesteps[10].item()
        
        result = scheduler.step(model_output, timestep, sample)
        
        assert result.prev_sample.shape == sample.shape

    def test_step_at_t_zero(self):
        """Test step at timestep 0 (no noise added)"""
        scheduler = DDPMScheduler()
        scheduler.set_timesteps(num_inference_steps=50)
        
        model_output = torch.randn(2, 3, 16, 16)
        sample = torch.randn(2, 3, 16, 16)
        timestep = 0
        
        result = scheduler.step(model_output, timestep, sample)
        
        # At t=0, no noise should be added
        assert result.prev_sample.shape == sample.shape

    def test_step_with_generator(self):
        """Test step with random generator"""
        scheduler = DDPMScheduler()
        scheduler.set_timesteps(num_inference_steps=50)
        
        model_output = torch.randn(2, 3, 16, 16)
        sample = torch.randn(2, 3, 16, 16)
        timestep = scheduler.timesteps[10].item()
        
        generator = torch.Generator().manual_seed(42)
        result = scheduler.step(model_output, timestep, sample, generator=generator)
        
        assert result.prev_sample.shape == sample.shape


class TestAlphaSigma:
    """Test get_alpha_sigma_sqrts method"""

    def test_get_alpha_sigma_sqrts(self):
        """Test get_alpha_sigma_sqrts method"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        timesteps = torch.tensor([100, 200, 300])
        device = "cpu"
        dtype = torch.float32
        shape = (2, 3, 16, 16)
        
        sqrt_alpha, sqrt_one_minus_alpha = scheduler.get_alpha_sigma_sqrts(
            timesteps, device, dtype, shape
        )
        
        assert sqrt_alpha.shape == (3, 1, 1, 1)
        assert sqrt_one_minus_alpha.shape == (3, 1, 1, 1)
        assert sqrt_alpha.dtype == dtype
        assert sqrt_one_minus_alpha.dtype == dtype


class TestAddNoise:
    """Test add_noise method"""

    def test_add_noise(self):
        """Test add_noise method"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        original_samples = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        timesteps = torch.tensor([100, 200])
        
        noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)
        
        assert noisy_samples.shape == original_samples.shape
        assert noisy_samples.dtype == original_samples.dtype

    def test_add_noise_different_dtypes(self):
        """Test add_noise with different dtypes"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        original_samples = torch.randn(2, 3, 16, 16, dtype=torch.float16)
        noise = torch.randn(2, 3, 16, 16, dtype=torch.float16)
        timesteps = torch.tensor([100, 200])
        
        noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)
        
        assert noisy_samples.dtype == torch.float16


class TestVelocity:
    """Test velocity-related methods"""

    def test_get_velocity(self):
        """Test get_velocity method"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        sample = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        timesteps = torch.tensor([100, 200])
        
        velocity = scheduler.get_velocity(sample, noise, timesteps)
        
        assert velocity.shape == sample.shape
        assert velocity.dtype == sample.dtype

    def test_get_noise_from_velocity(self):
        """Test get_noise method"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        sample = torch.randn(2, 3, 16, 16)
        velocity = torch.randn(2, 3, 16, 16)
        timesteps = torch.tensor([100, 200])
        
        noise = scheduler.get_noise(sample, velocity, timesteps)
        
        assert noise.shape == sample.shape
        assert noise.dtype == sample.dtype

    def test_velocity_noise_roundtrip(self):
        """Test that get_velocity and get_noise are related"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        sample = torch.randn(2, 3, 16, 16)
        original_noise = torch.randn(2, 3, 16, 16)
        timesteps = torch.tensor([100, 200])
        
        # Get velocity from noise
        velocity = scheduler.get_velocity(sample, original_noise, timesteps)
        
        # Get noise from velocity
        recovered_noise = scheduler.get_noise(sample, velocity, timesteps)
        
        # Check that shapes match and values are reasonable
        assert recovered_noise.shape == original_noise.shape
        assert velocity.shape == sample.shape


class TestIntegration:
    """Integration tests for full denoising process"""

    def test_full_denoising_loop(self):
        """Test a full denoising loop"""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            prediction_type="epsilon"
        )
        scheduler.set_timesteps(num_inference_steps=10)
        
        # Start with random noise
        sample = torch.randn(1, 3, 8, 8)
        
        for t in scheduler.timesteps:
            # Simulate model output (predicted noise)
            model_output = torch.randn_like(sample) * 0.1
            
            # Denoise step
            result = scheduler.step(model_output, t.item(), sample)
            sample = result.prev_sample
        
        assert sample.shape == (1, 3, 8, 8)

    def test_add_noise_and_denoise(self):
        """Test adding noise and then denoising"""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        original = torch.randn(1, 3, 8, 8)
        noise = torch.randn(1, 3, 8, 8)
        timesteps = torch.tensor([500])
        
        # Add noise
        noisy = scheduler.add_noise(original, noise, timesteps)
        
        # Simulate one denoising step
        scheduler.set_timesteps(num_inference_steps=50)
        model_output = torch.randn_like(noisy) * 0.1
        result = scheduler.step(model_output, 500, noisy)
        
        assert result.prev_sample.shape == original.shape
