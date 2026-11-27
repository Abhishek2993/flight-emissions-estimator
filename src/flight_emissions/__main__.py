"""Allow ``python -m flight_emissions``."""

from flight_emissions.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
