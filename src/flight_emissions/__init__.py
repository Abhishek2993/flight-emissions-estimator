"""Flight emissions estimator.

Estimate fuel burn and emissions (CO2, H2O, NOx, CO, HC) per flight phase from ADS-B
trajectories with the OpenAP performance model, and quantify operational inefficiency:
extra distance flown versus the great circle, and fuel lost to level-offs in descent.
"""

from flight_emissions.config import PipelineConfig

__all__ = ["PipelineConfig", "__version__"]
__version__ = "0.1.0"
