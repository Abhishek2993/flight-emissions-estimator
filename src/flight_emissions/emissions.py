"""Emissions (CO2, H2O, NOx, CO, HC) from the estimated fuel flow, using OpenAP.

* **CO2 and H2O** are proportional to fuel burned: 3.16 kg CO2 and 1.23 kg H2O per kg of
  jet fuel (stoichiometry of complete kerosene combustion, as used by ICAO).
* **NOx, CO and HC** depend on the engine operating point. The ICAO engine databank gives
  emission indices (g per kg of fuel) at four sea-level thrust settings (idle 7 %,
  approach 30 %, climb-out 85 %, take-off 100 %). OpenAP applies the Boeing Fuel Flow
  Method 2 (BFFM2, DuBois & Paynter 2006): the in-flight fuel flow is converted to a
  sea-level-equivalent fuel flow using ambient temperature and pressure ratios, the
  emission index is interpolated in the databank at that fuel flow, and corrected back to
  altitude (including a humidity term for NOx).

Rates are in g/s; :func:`emissions_by_phase` integrates them to kg with the same
trapezoidal rule and phase attribution as fuel (see :mod:`flight_emissions.fuel`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from flight_emissions.clean import elapsed_seconds
from flight_emissions.fuel import AircraftModel, integrate_by_phase
from flight_emissions.phases import PHASES

SPECIES: tuple[str, ...] = ("co2", "h2o", "nox", "co", "hc")
"""Emission species reported by the pipeline."""

CO2_PER_KG_FUEL = 3.16
H2O_PER_KG_FUEL = 1.23


@lru_cache(maxsize=64)
def emission_model(typecode: str) -> Any:
    """Cached :class:`openap.Emission` instance for a type code (synonyms allowed)."""
    from openap import Emission

    return Emission(typecode, use_synonym=True)


def emission_rates(
    fuel_flow_kgs: NDArray[np.float64],
    tas_kt: NDArray[np.float64],
    altitude_ft: NDArray[np.float64],
    model: AircraftModel,
) -> pd.DataFrame:
    """Emission rates (g/s, all engines) for every sample.

    Args:
        fuel_flow_kgs: Fuel flow, kg/s (all engines).
        tas_kt: True airspeed, knots.
        altitude_ft: Altitude, feet.
        model: Aircraft model (selects the engine's ICAO emission indices).

    Returns:
        DataFrame with columns ``co2_gs``, ``h2o_gs``, ``nox_gs``, ``co_gs``, ``hc_gs``.
    """
    em = emission_model(model.typecode)
    ff = np.asarray(fuel_flow_kgs, dtype=float)
    # BFFM2 needs a Mach number; clamp TAS to >= 1 kt (taxi) and altitude to >= 0 ft.
    tas = np.maximum(np.asarray(tas_kt, dtype=float), 1.0)
    alt = np.maximum(np.asarray(altitude_ft, dtype=float), 0.0)
    rates = {
        "co2_gs": em.co2(ff),
        "h2o_gs": em.h2o(ff),
        "nox_gs": em.nox(ff, tas=tas, alt=alt),
        "co_gs": em.co(ff, tas=tas, alt=alt),
        "hc_gs": em.hc(ff, tas=tas, alt=alt),
    }
    return pd.DataFrame(
        {k: np.nan_to_num(np.broadcast_to(v, ff.shape).astype(float)) for k, v in rates.items()}
    )


def emissions_by_phase(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Integrate emission rates to kg per phase.

    Args:
        df: Trajectory with ``timestamp`` and ``phase`` columns.
        rates: Output of :func:`emission_rates`, aligned with ``df``.

    Returns:
        DataFrame indexed by phase (all of :data:`PHASES`) with one column per species, in kg.
    """
    t_s = elapsed_seconds(df["timestamp"])
    phase = df["phase"].to_numpy()
    table = {}
    for sp in SPECIES:
        by_phase = integrate_by_phase(t_s, rates[f"{sp}_gs"].to_numpy(), phase)
        table[sp] = {p: v / 1000.0 for p, v in by_phase.items()}
    out = pd.DataFrame(table)
    order = [p for p in PHASES if p in out.index] + [p for p in out.index if p not in PHASES]
    return out.loc[order]
