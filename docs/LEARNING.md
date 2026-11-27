# Learning guide: how the flight emissions estimator works

This document explains, in plain language, every stage of the pipeline: what it does, the
formulas it uses, and **why** each design decision was made. The aim is that you can walk
through the project in an interview, justify every choice, and say honestly what it cannot do.

Contents

1. [The problem in one minute](#1-the-problem-in-one-minute)
2. [Background you need](#2-background-you-need)
3. [Stage by stage](#3-stage-by-stage)
   - [config.py](#configpy--one-place-for-every-threshold)
   - [ingest.py](#ingestpy--one-schema-for-every-source)
   - [clean.py](#cleanpy--from-messy-messages-to-a-uniform-time-series)
   - [phases.py](#phasespy--what-is-the-aircraft-doing)
   - [fuel.py](#fuelpy--how-much-fuel-per-second)
   - [emissions.py](#emissionspy--from-fuel-to-emissions)
   - [inefficiency.py](#inefficiencypy--how-much-was-wasted)
   - [validate.py](#validatepy--can-we-trust-the-numbers)
   - [pipeline.py, cli.py, plotting.py, dashboard](#pipelinepy-clipy-plottingpy-and-the-dashboard)
4. [Formula sheet](#4-formula-sheet)
5. [Testing strategy](#5-testing-strategy)
6. [What I would do next](#6-what-i-would-do-next)
7. [Likely interview questions](#7-likely-interview-questions)

---

## 1. The problem in one minute

Aircraft broadcast their position, altitude and speed about twice a second via **ADS-B**.
Anyone with a receiver can record this, and networks such as the OpenSky Network archive it.
ADS-B does **not** tell you how much fuel the aircraft burned, its weight, or its engine
thrust. So we use a **physics model of the aircraft** (OpenAP) to work out how much thrust
was needed to fly the observed trajectory, and then how much fuel that thrust costs.
Emissions follow from fuel.

Then we ask two "could it have been better?" questions:

* **Route:** how much further did it fly than the shortest path (the great circle)?
* **Descent:** did it descend smoothly (idle thrust all the way down), or in steps with level
  segments that need extra thrust?

---

## 2. Background you need

**ADS-B fields.** `lat`, `lon`, barometric `altitude` (ft), `groundspeed` (kt),
`vertical_rate` (ft/min), `track` (deg), `icao24` (the transponder's unique 24-bit address)
and `callsign`. Aviation uses feet for altitude, knots for speed (1 kt = 1.852 km/h) and
feet per minute for vertical speed. The project keeps those units and **puts the unit in
every column name** (`altitude_ft`, `groundspeed_kt`) so no one has to guess.

**Speeds.**

* *Ground speed (GS)* is speed over the ground (what ADS-B reports).
* *True airspeed (TAS)* is speed through the air mass. `GS = TAS + wind component`.
* *Calibrated airspeed (CAS)* is what the pilot flies. It is TAS corrected for air density,
  so at constant CAS the TAS increases with altitude.
* *Mach* is TAS divided by the speed of sound. Airliners cruise at constant Mach (~0.78).

Drag and thrust depend on **TAS** (and density), not ground speed. That is why the
"TAS ≈ GS" simplification matters.

**OpenAP.** OpenAP is an open aircraft performance model from TU Delft (Junzi Sun et al.).
For about 40 aircraft types it provides mass limits (MTOW, OEW), a drag polar, engine
thrust models, a fuel-flow model fitted to the ICAO engine emissions databank, a fuzzy-logic
flight-phase classifier, and emission models.

**ICAO engine emissions databank.** For each certified jet engine it gives the fuel flow and
the emission indices (grams of NOx, CO and HC per kg of fuel) at four thrust settings: idle
7 %, approach 30 %, climb-out 85 % and take-off 100 %. This is the "LTO cycle" (landing and
take-off cycle).

**Great circle.** The shortest path between two points on a sphere. Real flights are longer
because of airways, departure and arrival procedures, airspace restrictions, weather and
holding.

**Continuous descent operation (CDO).** The aircraft descends from cruise to the runway with
the engines near idle and no level segments. A level-off at low altitude needs thrust to
hold altitude, where the air is dense and drag is high, so it costs fuel and makes noise.

---

## 3. Stage by stage

### config.py — one place for every threshold

Every tunable number (resampling step, spike thresholds, mass fraction, level-off
definition...) is a field of a frozen dataclass, `PipelineConfig`, with a docstring that
explains the chosen default.

*Why:* a result is only reproducible if you know every parameter that produced it. The
config is saved as `run_config.json` with each run. Because it is frozen (immutable), no stage
can silently change a setting for the stages after it.

### ingest.py — one schema for every source

Data sources name things differently: `latitude` vs `lat`, `vertical_speed` (FlightRadar24)
vs `vertical_rate` (OpenSky). `normalise()` maps a table of known aliases onto the canonical
columns, converts timestamps to UTC, converts numbers to float (pandas nullable integers
become NaN), and fills per-flight constants (type, callsign, origin, destination). These
can come from the data or be passed in, because **ADS-B never carries the aircraft type**.

`load_flights()` reads every parquet/CSV file in a directory (skipping `_`-prefixed metadata
such as `_catalogue.csv`) and splits multi-flight tables by `flight_id`.

`load_opensky()` uses `traffic`'s OpenSky interface. It refuses to run unless
`OPENSKY_USERNAME` and `OPENSKY_PASSWORD` are set in the environment. Credentials never
appear in code or arguments.

*Why a canonical schema:* every later stage can be written and tested once, against one
well-defined input. Adding a new data source only means adding aliases.

### clean.py — from messy messages to a uniform time series

Real ADS-B data has duplicates (several receivers), messages that carry only some fields,
decoding glitches (a position 200 km off track for one sample), and coverage gaps
(oceans). Cleaning runs in a deliberate order, and every step is counted in a
`CleaningReport`:

1. **Drop rows without time or position.**
2. **Merge duplicates.** Exact duplicates are dropped. Rows with the same timestamp are
   merged by taking the first non-missing value of each column, because a position message
   and a velocity message for the same second complement each other.
3. **Plausibility limits.** Values that are physically impossible are set to NaN, but the
   row is kept because its position may still be fine: ground speed > 750 kt, |vertical
   rate| > 8,000 ft/min, altitude outside −1,500 to 50,000 ft.
4. **Spike removal.** For each link between consecutive points we compute the implied speed
   `v = haversine distance / Δt`. A point is a spike if **both** its incoming and outgoing
   links are impossible (> 800 kt).
   *Why both links?* If point *k* is a spike, the links *k−1→k* and *k→k+1* are both bad,
   but the good point *k+1* has only one bad link. Flagging any point with a bad link would
   also delete the innocent neighbour. End points have only one link, so an end point is
   flagged when its link is bad and the next link is good. The detection repeats until
   nothing changes, because removing one spike changes its neighbours' links. Altitude
   spikes are handled the same way with an implied vertical rate (> 10,000 ft/min) and are
   set to NaN rather than deleting the row.
5. **Resampling to a uniform 10 s grid.** Each numeric column is linearly interpolated onto
   the grid. Interpolation only happens across gaps of up to `max_gap_s` (300 s); grid
   points inside longer gaps are **dropped, not invented**. The track angle is unwrapped
   before interpolation so that 359° → 1° does not swing through 180°.
   *Why a uniform grid:* integration, smoothing windows and the phase classifier all assume
   (or work better with) regular sampling. 10 s keeps an 11-hour flight at about 4,000 rows
   and still resolves a 20 s level-off with three samples.
   *Why not fill long gaps:* over the Atlantic, a straight line between two reports 30
   minutes apart is a guess. Fuel across such a gap is still counted (see fuel.py), but the
   pipeline reports the gap instead of pretending it had data.
6. **Derived fills.** Missing ground speed comes from position differences, and missing
   vertical rate from the altitude derivative (`np.gradient`). Missing altitude while slow on
   the ground (surface position messages carry no altitude) is filled with the nearest
   reported altitude.

### phases.py — what is the aircraft doing?

OpenAP's `FlightPhase` uses **fuzzy logic**. For each 60 s window it takes the mean altitude,
speed and rate of climb, and computes how much each value belongs to fuzzy sets such as
"altitude low/high", "speed medium/high" and "rate of climb zero/positive/negative".
Rules such as *climb = low altitude AND positive rate AND medium speed* are evaluated, and
the strongest rule wins (defuzzification). Labels: ground, climb, cruise, level, descent.

Fuzzy logic suits this problem because the boundaries really are fuzzy: there is no single
correct vertical rate at which "level" becomes "descent".

Three post-processing rules make it work on real data:

1. **Ground rule.** OpenAP's "ground" membership function fades out between 0 and 200 ft,
   which assumes a sea-level airport. At Zurich (1,416 ft) a taxiing aircraft would be called
   "level flight". Any sample slower than 50 kt is on the ground, because no airliner can fly
   that slowly. The rule is applied again *after* merging so that merging can never turn a
   taxiing aircraft into a flying one.
2. **Cruise band.** OpenAP separates "cruise" from "level" mainly by **speed**. Into a strong
   headwind (ground speed 380 kt) a normal cruise at FL380 looked like "level". Level flight
   within 4,000 ft of the flight's top altitude (and above 10,000 ft) is relabelled cruise.
   A level-off during the descent, which is far below the top altitude, stays "level".
3. **Short-segment merging.** Window-based labels flicker at boundaries (`climb, level,
   climb`). Segments shorter than 120 s are absorbed into a neighbour: the neighbours'
   common label if they agree, otherwise the longer neighbour. The *shortest* segment is
   merged first and the segmentation recomputed each time, so the result does not depend on
   scanning direction. Unclassified windows (OpenAP leaves the last window as "NA") are
   always absorbed.

### fuel.py — how much fuel per second?

**Aircraft lookup.** `resolve_aircraft()` reads OpenAP's database: MTOW, OEW, engine count,
default engine, cruise Mach, ceiling and seats. If the type is unknown, OpenAP synonyms are
tried (B762 → B763). If that fails, the flight is skipped, or modelled with an explicit
`--fallback-typecode` and a warning.

**The physics of the airborne fuel flow.** Treat the aircraft as a point mass flying along
its path. Along the flight path the forces balance as:

```
T − D − m g sin(γ) = m a          →     T = D + m g sin(γ) + m a
γ = atan(vertical speed / TAS)          (flight-path angle)
```

* `D` (drag) comes from OpenAP's drag polar: `D = ½ ρ V² S C_D` with
  `C_D = C_D0 + k C_L²` (plus a wave-drag term at high Mach) and
  `C_L = m g / (½ ρ V² S)` (lift balances weight). Heavier means
  more lift, which means more induced drag. That is why **mass matters so much**.
* `m g sin(γ)` is the climb term. Climbing needs extra thrust, and descending needs less,
  down to idle.
* OpenAP then maps the thrust (as a fraction of maximum) to fuel flow with an engine
  fuel-flow curve fitted to the ICAO databank points. Thrust is limited to the idle..max
  range with a smooth clamp.

In the main estimate the acceleration term `m a` is set to 0. The only speed we have is
ground speed, whose derivative includes wind changes (for example turning from headwind to
tailwind). That noise would enter thrust directly, and because thrust cannot go below idle,
the noise biases fuel **upwards**. On the Paris–Toulouse flight including it added about 2 %.

**Inputs and simplifications.**

* **TAS ≈ ground speed** (no wind in ADS-B). A `tas_kt` column is used if present.
* **Mass.** `m0 = 0.85 × MTOW`. With `update_mass` the mass then decreases as fuel burns,
  using a *two-pass fixed point*: pass 1 computes fuel flow at constant mass, then
  `m(t) = m0 − burned(t)`, then pass 2 recomputes fuel flow with that mass. One iteration is
  enough because the correction is small (a few percent of mass changes fuel flow by a few
  percent, and the second-order effect is negligible). Mass is floored at OEW.
* **Vertical rate** is smoothed with a 30 s centred moving average before entering γ.
  Single-sample noise would otherwise toggle between idle and climb thrust.

**Thrust modes** (`thrust_mode()`):

| mode | when | fuel flow |
|---|---|---|
| taxi | ground phase | ICAO databank idle fuel flow × engines (7 % thrust) |
| takeoff | ground samples faster than 50 kt before the first airborne sample | OpenAP take-off thrust |
| airborne | flight phases | OpenAP `enroute` (physics above) |
| landing | non-ground samples after the last point > 100 ft above landing altitude | idle |

*Why not OpenAP for taxi?* The drag model is for a clean wing in flight. At 15 kt it is
meaningless; on the landing roll it even returned NaN or enormous values. ICAO idle is
the standard taxi value in emission inventories. OpenAP's own fuel curve at 7 % thrust gave
an A321 0.39 kg/s versus the databank's 0.23 kg/s.

**Integration: the trapezoidal rule.**

```
fuel = Σ ½ (ff_i + ff_{i+1}) · (t_{i+1} − t_i)
```

This is exact for a fuel flow that changes linearly between samples, and it uses the
*actual* time stamps, so it is correct across gaps too. Over a 30-minute gap it assumes the
fuel flow changes linearly from one side to the other, which is a good approximation in
cruise. Each interval is attributed to the phase of its first sample, so per-phase totals
add up exactly to the flight total.

**Plausibility check.** `check_type_plausibility()` warns if the aircraft flew above its
type's ceiling, or if the median cruise ground speed is more than 35 % away from
`Mach_cruise × speed of sound`. That catches gross type errors (a 747 tagged as a Cessna);
it cannot tell an A320 from an A321.

### emissions.py — from fuel to emissions

* **CO₂ = 3.16 × fuel, H₂O = 1.23 × fuel.** Burning 1 kg of kerosene (about C₁₂H₂₃) with
  oxygen produces about 3.16 kg of CO₂ and 1.23 kg of water. These are fixed by chemistry,
  so CO₂ is exactly as accurate as the fuel estimate.
* **NOx, CO, HC** depend on combustor temperature and pressure, i.e. on the thrust setting.
  The ICAO databank gives emission indices (g/kg fuel) only at four *sea-level* thrust
  settings. **Boeing Fuel Flow Method 2 (BFFM2)** bridges the gap:
  1. Convert the in-flight fuel flow to a *sea-level-equivalent* fuel flow using the ambient
     temperature ratio θ and pressure ratio δ: `ff_SL = ff · θ^3.8 / δ · e^{0.2 M²}`.
  2. Look up (interpolate) the emission index at that fuel flow in the databank.
  3. Correct back to altitude. For NOx: `EI_NOx = EI_SL · √(δ^1.02 / θ^3.3) · e^{H}`, with a
     humidity term H. For CO and HC: `EI = EI_SL · θ^3.3 / δ^1.02`.
  NOx rises with thrust (hot combustion), while CO and HC are highest at idle (incomplete
  combustion). The unit tests check exactly this behaviour.
* Rates (g/s) are integrated with the same trapezoidal rule and phase attribution as fuel.

### inefficiency.py — how much was wasted?

**Route inefficiency.**

```
extra % = (flown − great circle) / great circle × 100
extra fuel = extra km × (cruise fuel ÷ cruise distance)
```

* The great circle uses the **first and last airborne** points, not the airports. That way
  the metric works for any coverage window and excludes taxi routes.
* The haversine formula (sphere, R = 6,371.0088 km):
  `d = 2R · asin(√(sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2)))`. The spherical error (< 0.5 %)
  is negligible next to the other uncertainties, and the formula is easy to check by hand.
* The extra distance is costed at **cruise** fuel per km, which is conservative. Extra
  distance flown low down (vectoring, holding) costs more per km.
* **Circular flights.** If the end points are less than 100 km apart (sightseeing, training,
  round trips) the percentage is meaningless, because it divides by almost zero. It is
  reported as "n/a (circular)" instead. TRA051 is in the sample to exercise exactly this.

**Descent inefficiency.**

1. **Window.** The floor (end) is the first sample of the final run below *landing altitude
   + 3,000 ft* that lasts until landing. The top of descent (start) is the last sample within
   200 ft of the highest altitude reached after the last climb before the floor. For a normal
   flight that is the end of cruise. For a sightseeing flight it is the top of the *final*
   descent only. The first version simply used "end of last cruise", and for TRA051 that
   window swallowed three hours of sightseeing, which is why the definition changed.
   *Why stop at 3,000 ft?* Below about 3,000 ft, aircraft fly a short level segment to
   intercept the ILS glide slope. That segment is procedural, not inefficient.
2. **Level-offs.** These are runs where |vertical rate| < 300 ft/min for at least 20 s. This
   is the harmonised EUROCONTROL/FAA definition used in vertical flight efficiency studies.
   (A 60 s minimum missed a real 40 s level-off at 5,000 ft on the Paris–Toulouse flight,
   which led to switching to the published definition.)
3. **The idealised continuous descent (CDO).** For a fair comparison the ideal must cover
   the **same ground distance** between the **same start and end altitudes**:
   * stay at the top-of-descent altitude, then descend at a constant 3° flight-path angle
     (≈ 318 ft per nautical mile) so as to arrive exactly at the window's end altitude;
   * if 3° does not fit in the distance (the actual descent was steeper), use a constant
     gradient over the whole distance;
   * fly the **actual calibrated airspeed** at each distance, converted to TAS at the ideal
     altitude and capped at cruise Mach. Pilots fly CAS, so the comparison changes only the
     vertical profile, not the speed schedule;
   * compute both fuel figures with the same OpenAP model, the same mass (the mass at top of
     descent) and the **same treatment of acceleration**. Here the acceleration term *is*
     included (60 s smoothed TAS derivative): a descent is an energy-management manoeuvre,
     and a decelerating aircraft can fly a shallower path at idle. Without it, a normal
     deceleration from 250 to 180 kt on a shallow approach looked like powered flight, and
     the "saving" was inflated to 11–17 % of descent fuel on four flights that had almost no
     level-offs. Because both profiles are treated identically,
     the wind noise in the acceleration largely cancels in the difference.
4. **Saving** = actual − ideal. A flight that already flies a continuous 3° descent gets a
   saving close to zero; the unit test checks this (< 2 %). The saving captures level-offs
   *and* shallow powered segments, so it is an upper bound on what a CDO could save given
   the model, not a measured fact.

### validate.py — can we trust the numbers?

**Check 1: measured fuel flow.** `traffic` ships an A320 flight-data-recorder sample with the
real fuel flow and gross weight every second. The fuel model runs under three input
scenarios that peel the error apart:

| scenario | what it isolates |
|---|---|
| recorded mass + TAS from recorded CAS | error of the OpenAP model itself |
| recorded mass + TAS = ground speed | + the no-wind simplification |
| 0.85 × MTOW + TAS = ground speed | the pipeline's default assumptions |

The result: OpenAP alone overestimates total fuel by 3.5 % (10.5 % mean per-sample error,
correlation 0.97). Ground speed instead of TAS adds 2 points. The default assumptions end
at +2.2 %, but only because the assumed mass is lighter than the real one, which offsets
the other errors. Say this clearly in an interview: **a good total can hide compensating
errors, which is why the scenarios are separated.**

**Check 2: ICAO Carbon Emissions Calculator.** The calculator gives CO₂ per economy
passenger and fuel per journey for a city pair. Values are typed by hand into
`data/reference/icao_reference.csv`, and the pipeline reports the % difference, or
"awaiting ICAO value" when a row is empty. Expected reasons for differences:

* ICAO allocates CO₂ to passengers using load factors and removes a cargo share; we use
  seats × 0.80 and allocate everything to passengers;
* ICAO averages over all aircraft on the route; we model the actual type;
* ICAO uses great circle + a fixed correction; we use the actual track;
* our 0.85 × MTOW mass assumption, plus OpenAP and wind errors;
* ICAO uses a standard LTO cycle for taxi; we use what ADS-B saw.

### pipeline.py, cli.py, plotting.py and the dashboard

* `process_flight()` chains the stages and returns a `FlightResult`. If a stage cannot run
  (fewer than 10 points, unknown type), the flight gets a **status** such as
  `skipped: cannot model aircraft type 'C172'` and the batch continues. One bad file must
  not kill a 10,000-flight run.
* `run_pipeline()` writes the CSV tables, per-flight parquet time series (which the dashboard
  can reuse), `run_config.json`, and `summary.md`. The README copies its tables from
  `summary.md` with `scripts/update_readme.py`, so the README cannot drift from the code.
* **Figures** use one fixed colour per phase everywhere. The four flight phases take the first
  four slots of a colour-blind-checked categorical palette, and ground is a neutral grey.
  Two different measures (e.g. % extra distance and kg saved) are drawn as **two panels,
  never on two y-axes**, because a dual axis invents a visual correlation. Map backgrounds
  use Natural Earth boundaries bundled with the package, so nothing needs the network.
* The **Streamlit dashboard** recomputes results with `st.cache_resource` (keyed on the
  input folder and the two assumption sliders). Moving the mass slider shows directly how
  much the headline number depends on that assumption.

---

## 4. Formula sheet

| Quantity | Formula | Where |
|---|---|---|
| Great-circle distance | `2R asin √(sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2))`, R = 6,371.0088 km | `geo.haversine_km` |
| Implied speed (spike test) | `distance / Δt`, in kt | `clean.detect_position_spikes` |
| Vertical rate from altitude | `dh/dt × 60` (ft/min), central differences | `clean.derive_vertical_rate_fpm` |
| Flight-path angle | `γ = atan(VS / TAS)` | OpenAP `FuelFlow.enroute` |
| Required thrust | `T = D + m g sin γ + m a` | OpenAP `FuelFlow.enroute` |
| Drag | `D = ½ ρ V² S (C_D0 + k C_L²)`, `C_L = m g / (½ ρ V² S)` | OpenAP `Drag` |
| Initial mass | `m₀ = 0.85 × MTOW` | `fuel.estimate_fuel_flow` |
| Mass update | `m(t) = max(m₀ − ∫ff dt, OEW)` | `fuel.estimate_fuel_flow` |
| Fuel (trapezoid) | `Σ ½ (ff_i + ff_{i+1}) Δt_i` | `fuel.interval_amounts` |
| CO₂, H₂O | `3.16 × fuel`, `1.23 × fuel` | OpenAP `Emission` |
| NOx (BFFM2) | sea-level-equivalent ff → ICAO EI → altitude correction | OpenAP `Emission.nox` |
| Route inefficiency | `(flown − GC) / GC × 100` | `inefficiency.route_inefficiency` |
| Extra fuel | `extra km × cruise fuel / cruise km` | `inefficiency.route_inefficiency` |
| 3° descent gradient | `tan 3° / 0.0003048 ≈ 172 ft/km ≈ 318 ft/NM` | `inefficiency.ideal_descent_altitude` |
| Ideal descent length | `(H₀ − H₁) × 0.0003048 / tan 3°` km | `inefficiency.ideal_descent_altitude` |
| Per-passenger CO₂ | `CO₂ / (seats × load factor)` | `validate.compare_with_icao` |

---

## 5. Testing strategy

The tests do not compare against "whatever the code produced last time". They use
**synthetic trajectories with answers you can derive on paper**:

* flights due east along the **equator**, where one degree of longitude is exactly
  2πR/360 km, so distances and speeds are known exactly;
* **constant** and **linear** fuel flows, where the trapezoidal rule must give the exact
  integral, including across a deliberately inserted gap;
* **injected faults**: a 220 km position spike, an altitude spike, a 2,000 kt ground speed, a
  25,000 ft/min vertical rate, duplicates, a 110 s and a 610 s gap. Each test checks that
  the right number of points is removed or nulled, and that the good neighbours survive;
* **label sequences** for segment merging (`climb, level, climb` → climb);
* a **perfect 3° descent**, where the CDO saving must be ≈ 0, and a **stepped descent** with
  a 6-minute level-off at FL150, which must be detected with the right altitude and must
  show a positive saving;
* a flight **without a type code**, which must be skipped, or modelled with a warning when a
  fallback is configured;
* a synthetic **flight-data recording generated by the model itself**, where the validation
  must report ≈ 0 % error.

Coverage is 98 %; CI enforces at least 70 %.

---

## 6. What I would do next

1. **Wind.** Get TAS from ERA5 reanalysis winds, or from Mode S EHS (BDS 5,0/6,0) where
   available. That removes the largest per-segment error after mass.
2. **Mass estimation.** Estimate take-off mass from the climb profile (energy rate vs thrust),
   or from stage length with a fuel-planning model, instead of a fixed MTOW fraction.
3. **Uncertainty propagation.** Monte Carlo over mass, wind and model error to report fuel as
   a range, not a single number.
4. **Airport-based metrics.** Measure the great circle airport-to-airport and use EUROCONTROL's
   additional time in terminal airspace (ASMA) and taxi-out time.
5. **Scale.** Run on a month of OpenSky data with parallel processing per flight. The pipeline
   is already per-flight and stateless.
6. **Non-CO₂ effects.** Contrail formation (OpenAP has a contrail module) matters as much as
   CO₂ for aviation's climate impact.

---

## 7. Likely interview questions

**Why is mass the biggest uncertainty, and how big is it?**
Lift must equal weight, and induced drag grows with lift squared over speed squared. The
sensitivity table in the README shows that ±0.15 of MTOW changes total fuel by roughly
±10–15 %. That is larger than the OpenAP model error (+3.5 % on the A320 check).

**Why not just use the ICAO calculator?**
It gives a route average for a typical aircraft, per passenger. It cannot tell you that a
*specific* flight held for 20 minutes, flew 49.5 % further than the great circle, or levelled
off in the descent. This pipeline is flight-specific and phase-resolved. The ICAO figure is
used as a sanity check, not as the answer.

**How do you know your fuel model is right?**
I do not assume it is. It is checked against measured fuel flow from a flight-data recorder,
with the error split by source: model, wind and mass. The model alone is within 3.5 % on total
fuel for that A320 flight. Other types are not verified here, so I say so.

**What happens when there is no data over the ocean?**
Gaps up to 5 minutes are interpolated. Longer gaps are not filled with fake points: they are
reported (153 minutes for EDW24), distance across them is a straight line (a lower bound), and
fuel is integrated assuming fuel flow changes linearly from one side to the other.

**How do you detect a level-off, and why that threshold?**
|vertical rate| < 300 ft/min for at least 20 s, on a 30 s smoothed vertical rate, inside the
window from top of descent to 3,000 ft above the runway. That is the harmonised
EUROCONTROL/FAA definition. Using a published definition makes results comparable with
other studies.

**Your CDO saving: is it real?**
It is a model-based estimate and an upper bound. It compares the actual vertical profile with
an idealised 3° continuous descent over the same distance, at the same calibrated airspeed and
mass, with the same fuel model. It does not account for ATC constraints that made the
level-off necessary. A sanity test shows that a perfect continuous descent gives a saving of
about zero.

**Why trapezoidal integration rather than summing ff × Δt?**
It is exact for linear changes, it is second-order accurate rather than first-order, and with
real timestamps it handles irregular sampling and gaps correctly.

**Why OpenAP and not BADA?**
BADA (EUROCONTROL) is more accurate but licensed and cannot be redistributed. OpenAP is open,
peer-reviewed and covers the common airliners, so anyone can reproduce the results.

**What would break if you ran this on 100,000 flights?**
Nothing structural: each flight is processed independently, and failures are isolated per
flight. Missing type codes would be the main issue at scale (ADS-B has none), so an aircraft
database join would be added to ingestion, which the `--typecode` option already anticipates.
