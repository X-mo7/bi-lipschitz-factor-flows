"""Scientific checks against analytic cases, plus a tiny optimization test."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.func import jacfwd

from bff.losses import (
    endpoint_separation, latent_early_lower_loss, latent_singular_values,
    latent_upper_loss, sample_pairs, state_jacobian_loss,
)
from bff.metrics import empirical_distortion, factor_r2, fit_probe, predict_probe
from bff.models import BFF
from bff.ode import euler_solve
from bff.synthetic import make_genes
from bff.train import Config, load_config, train
from examples.cancellation import rotating_velocity


@pytest.fixture(autouse=True)
def deterministic_small_tests():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    yield
    torch.set_num_threads(previous_threads)


def test_euler_endpoint_and_latent_derivative_match_constant_field():
    matrix = torch.tensor([[2., 0.], [0., 0.5], [1., -1.]], dtype=torch.float64)
    code = torch.tensor([0.3, -0.8], dtype=torch.float64)
    source = torch.tensor([[0.5, 0.7, -0.2]], dtype=torch.float64)
    velocity = lambda t, x, z: z @ matrix.T
    decode = lambda z: euler_solve(velocity, z[None], source, 7)[0]
    torch.testing.assert_close(decode(code), source[0] + matrix @ code)
    torch.testing.assert_close(jacfwd(decode)(code), matrix)


def test_latent_spectrum_and_hinge_gradients_on_known_matrix():
    diagonal = torch.tensor([2., 0.5], dtype=torch.float64, requires_grad=True)
    velocity = lambda t, x, z: z * diagonal
    t, x, z = torch.zeros(4, dtype=torch.float64), torch.zeros(4, 2, dtype=torch.float64), torch.ones(4, 2, dtype=torch.float64)
    spectrum = latent_singular_values(velocity, t, x, z)
    torch.testing.assert_close(spectrum, diagonal.detach().expand(4, 2))
    upper = latent_upper_loss(velocity, t, x, z, ceiling=1.)
    lower = latent_early_lower_loss(velocity, t, x, z, floor=1.)
    torch.testing.assert_close(upper, torch.tensor(1., dtype=torch.float64))
    torch.testing.assert_close(lower, torch.tensor(0.25, dtype=torch.float64))
    (upper + lower).backward()
    torch.testing.assert_close(diagonal.grad, torch.tensor([2., -1.], dtype=torch.float64))


def test_hutchinson_is_exact_on_diagonal_state_jacobian():
    diagonal = torch.tensor([2., -3.], dtype=torch.float64, requires_grad=True)
    velocity = lambda t, x, z: x * diagonal + z
    x = torch.ones(5, 2, dtype=torch.float64)
    loss = state_jacobian_loss(velocity, x[:, 0], x, torch.zeros_like(x))
    torch.testing.assert_close(loss, torch.tensor(13., dtype=torch.float64))
    loss.backward()
    torch.testing.assert_close(diagonal.grad, 2 * diagonal.detach())


def test_rotation_has_zero_local_penalties_but_positive_endpoint_loss():
    z = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    x = torch.zeros_like(z)
    t = torch.tensor([0.03, 0.09], dtype=torch.float64)
    assert state_jacobian_loss(rotating_velocity, t, x, z).item() == 0
    assert latent_upper_loss(rotating_velocity, t, x, z, 1.1).item() == 0
    assert latent_early_lower_loss(rotating_velocity, t, x, z, 0.9).item() == 0
    endpoints = euler_solve(rotating_velocity, z, x, 64)
    torch.testing.assert_close(endpoints, x, atol=1e-14, rtol=0)
    assert endpoint_separation(endpoints, z, margin=0.5).item() == pytest.approx(1.)


def test_endpoint_loss_formula_gate_and_gradients():
    z = torch.tensor([[0., 0.], [2., 0.]], requires_grad=True)
    endpoints = torch.tensor([[0., 0.], [0.5, 0.]], requires_grad=True)
    loss = endpoint_separation(endpoints, z, margin=1.)
    assert loss.item() == pytest.approx(2.25)
    loss.backward()
    assert z.grad[1, 0].item() == pytest.approx(3.)
    assert endpoints.grad[1, 0].item() == pytest.approx(-3.)
    assert endpoint_separation(endpoints, z, margin=1., delta=2.).item() == 0


def test_singleton_separation_is_zero_and_differentiable():
    z = torch.ones(1, 2, requires_grad=True)
    x = torch.ones(1, 3, requires_grad=True)
    loss = endpoint_separation(x, z, margin=1.)
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(z.grad).item() == 0


def test_pair_sampler_never_pairs_a_sample_with_itself():
    i, j = sample_pairs(10, 1000, device="cpu", generator=torch.Generator().manual_seed(5))
    assert (i != j).all()
    assert min(i.min(), j.min()) >= 0 and max(i.max(), j.max()) < 10


def test_distortion_of_scaled_isometry_is_one_and_collapse_is_degenerate():
    z = torch.randn(64, 2)
    result = empirical_distortion(z, 3 * z)
    assert result["kappa95"] == pytest.approx(1., rel=1e-6)
    assert result["r_p05"] == pytest.approx(3., rel=1e-6)
    collapsed = empirical_distortion(z, torch.zeros_like(z))
    assert collapsed["degenerate"] and collapsed["kappa95"] is None


def test_probe_recovers_affine_factors_on_unseen_codes():
    rng = np.random.RandomState(4)
    train_z, test_z = rng.normal(size=(50, 2)), rng.normal(size=(20, 2))
    matrix = np.array([[2., 1.], [-1., 3.]])
    train_y, test_y = train_z @ matrix + 0.4, test_z @ matrix + 0.4
    predictions = predict_probe(test_z, fit_probe(train_z, train_y))
    np.testing.assert_allclose(factor_r2(test_y, predictions), np.ones(2))


def test_gene_preprocessing_roundtrip_and_fixed_seeds():
    data = make_genes(20, 8, 10)
    other = make_genes(20, 8, 10)
    torch.testing.assert_close(data.train, other.train)
    np.testing.assert_array_equal(data.factors_test, other.factors_test)
    torch.testing.assert_close(data.train.mean(0), torch.zeros(10), atol=1e-7, rtol=0)
    counts = np.expm1((data.train.numpy() + data.mean) * data.scale)
    np.testing.assert_allclose(counts, np.round(counts), atol=0.001, rtol=0)


def test_decoder_uses_one_source_and_is_independent_of_batch_order():
    model = BFF(input_dim=4, latent_dim=2, hidden_dim=8, n_steps=3)
    z = torch.randn(3, 2)
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(model.decode(z)[order], model.decode(z[order]))


def test_geometric_gradients_reach_encoder_and_velocity_parameters():
    model = BFF(input_dim=4, latent_dim=2, hidden_dim=8, n_steps=3)
    x = torch.randn(5, 4)
    z = model.encoder(x)
    t = torch.linspace(0.01, 0.09, 5)
    loss = latent_early_lower_loss(model.velocity, t, t[:, None] * x, z, floor=1.)
    loss.backward()
    for module in (model.encoder, model.velocity):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0


def test_tiny_training_updates_parameters_and_is_repeatable():
    config = Config(n_train=9, n_test=4, n_genes=4, hidden_dim=8, n_steps=2,
                    epochs=2, batch_size=4, geometry_batch=2, n_pairs=4)
    x = make_genes(9, 4, 4).train
    model, history = train(config, x)
    repeated, other_history = train(config, x)
    assert history == other_history
    torch.manual_seed(config.seed)
    initial = BFF(4, 2, 8, 2)
    assert any(not torch.equal(a, b) for a, b in zip(model.parameters(), initial.parameters()))
    for a, b in zip(model.parameters(), repeated.parameters()):
        torch.testing.assert_close(a, b)
    assert all(np.isfinite(row["total"]) for row in history)


def test_public_configs_share_data_architecture_and_training():
    root = Path(__file__).resolve().parents[1]
    full = asdict(load_config(root / "configs/bff.yaml"))
    baseline = asdict(load_config(root / "configs/rec_fm.yaml"))
    ablation = asdict(load_config(root / "configs/bff_no_sep.yaml"))
    assert {k for k in full if full[k] != ablation[k]} == {"name", "lambda_sep"}
    assert {k for k in full if full[k] != baseline[k]} == {
        "name", "lambda_x", "lambda_upper", "lambda_early", "lambda_sep",
    }


@pytest.mark.parametrize("change", [
    {"epsilon": 0.}, {"lambda_sep": -1.}, {"epochs": 0},
    {"learning_rate": float("nan")}, {"latent_dim": 101},
])
def test_invalid_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        replace(Config(), **change).validate()

