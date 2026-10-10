"""Service reservations cannot demand more work than the arrived task contains."""
from types import SimpleNamespace

import numpy as np
import pytest

from planning import service_guard


def reserve(monkeypatch, *, work, rate, capacity, horizon, deadline):
    env = SimpleNamespace(current_step=0,
                          terminal_service_guard_version='arrived-service-reserve-v3',
                          bess_charge_power_max_kW=100., access_limit_kw=100.)
    monkeypatch.setattr(service_guard, 'power_upper_envelope',
                        lambda *args: (0., np.ones(len(capacity))))
    task = dict(task_id='known', remaining_work=work, max_rate=rate, priority=1,
                deadline=deadline, arrival=0, interruptible=True, started=False)
    return service_guard._coupled_reserve(
        env, [task], capacity, np.zeros((1, len(capacity))), [0],
        [25.] * horizon, [0.] * horizon, 0.)


def test_recorded_fragmented_task_never_creates_an_inverted_progress_interval(monkeypatch):
    # Actual group fragments from 9744/44 task45; ordinary summation exceeds work.
    capacity = [10.706640845696876, 11.763164192403284, 4.530194961899841]
    result = reserve(monkeypatch, work=27., rate=27., capacity=capacity,
                     horizon=4, deadline=2)
    assert result.known_service_required_end_work == [27.]
    assert result.known_service_required_due_work == [27.]
    for k in range(4):
        lower = max(result.known_service_required_end_work[0] - 27. * (3-k), 0.)
        if k < 2:
            lower = max(lower, result.known_service_required_due_work[0] - 27. * (1-k))
        assert lower <= min(27., (k+1)*27.)
    # The allocation is not rewritten, and no job/capacity is removed to pass.
    assert result.known_service_allocation[0][0] == capacity
    assert result.known_service_shortfall_work == 0.


@pytest.mark.parametrize(('deadline', 'due', 'shortfall'), [(0, 0., 9.), (1, 3., 6.),
                                                         (2, 6., 3.), (5, 9., 0.)])
def test_deadline_and_end_use_the_same_actual_service_prefix(
        monkeypatch, deadline, due, shortfall):
    result = reserve(monkeypatch, work=9., rate=3., capacity=[1., 2.],
                     horizon=3, deadline=deadline)
    assert result.known_service_required_end_work == [9.]
    assert result.known_service_required_due_work == [due]
    assert result.known_service_shortfall_work == shortfall


def test_positive_unserved_tail_is_not_zeroed_by_an_epsilon(monkeypatch):
    work = float(np.nextafter(3., np.inf))
    result = reserve(monkeypatch, work=work, rate=3., capacity=[1., 2.],
                     horizon=2, deadline=2)
    assert result.known_service_required_end_work == [3.]
    assert result.known_service_required_due_work == [3.]
    assert result.known_service_shortfall_work == work - 3. > 0.


def test_capacity_shortage_is_preserved_as_unserved_work(monkeypatch):
    result = reserve(monkeypatch, work=10., rate=10., capacity=[1., 1.],
                     horizon=3, deadline=2)
    assert result.known_service_required_end_work == [6.]
    assert result.known_service_required_due_work == [4.]
    assert result.known_service_shortfall_work == 6.


@pytest.mark.parametrize('horizon', [1, 3, 7, 12, 30, 48])
@pytest.mark.parametrize('rate', [.1, .3, 1.7])
def test_fractional_rates_cannot_accumulate_more_than_their_exact_capacity(
        monkeypatch, horizon, rate):
    from fractions import Fraction
    result = reserve(monkeypatch, work=100., rate=rate, capacity=[rate],
                     horizon=horizon, deadline=horizon)
    actual = result.known_service_required_end_work[0]
    assert Fraction(actual) <= horizon * Fraction(rate)
    assert result.known_service_required_due_work == [actual]
    assert actual > 0.


@pytest.mark.parametrize('rate', [.1, .3, 1.7, 27.])
def test_derived_progress_bounds_are_valid_for_exact_rate_arithmetic(rate):
    from fractions import Fraction
    # Check every prefix of all registered horizon lengths against exact inputs.
    for horizon in range(1, 49):
        total = horizon * Fraction(rate)
        end = float(total)
        if Fraction(end) > total:
            end = float(np.nextafter(end, -np.inf))
        for k in range(horizon):
            lo, hi = service_guard.service_progress_bounds(
                2000., rate, end, end, horizon, horizon, k)
            exact_lo = max(Fraction(0), Fraction(end) - (horizon-k-1)*Fraction(rate))
            exact_hi = (k+1)*Fraction(rate)
            assert Fraction(lo) <= exact_lo <= exact_hi <= Fraction(hi)
            assert lo <= hi


def test_progress_bounds_do_not_hide_an_inconsistent_original_requirement():
    lo, hi = service_guard.service_progress_bounds(
        27., 27., float(np.nextafter(27., np.inf)), 27., 1, 1, 0)
    assert lo > hi
