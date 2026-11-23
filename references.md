# References and parameter provenance

This list records the sources used to ground the starter profile. A source supports only the specifications described below; it does not validate the entire simulator.

## Radar hardware and antenna reference

1. **Texas Instruments — IWR6843 product page and datasheet entry**  
   https://www.ti.com/product/IWR6843  
   Supports the 60–64 GHz operating band, 3 TX / 4 RX architecture, 4 GHz continuous bandwidth, and the published receiver-noise-figure / phase-noise specifications shown on the product page. Check the precise datasheet revision before translating these into a hardware calibration.

2. **Texas Instruments — IWR6843ISK-ODS evaluation module**  
   https://www.ti.com/tool/IWR6843ISK-ODS  
   Supports the ODS EVM reference and its stated 120° azimuth / 120° elevation field of view, 3 TX and 4 RX.

3. **Texas Instruments — 60 GHz mmWave Sensor EVMs User's Guide (SWRU546)**  
   https://www.ti.com/lit/ug/swru546e/swru546e.pdf  
   Describes the ODS antenna arrangement and 4 × 3 virtual antenna positions, short-range people-detection context, and EVM-level antenna/output details. The starter pack's 4 × 3 *uniform half-wavelength* coordinates remain an idealized replacement, not extracted board geometry.

## Building materials and 60 GHz propagation

4. **ITU-R Recommendation P.2040-4 (September 2025)**  
   https://www.itu.int/rec/R-REC-P.2040-4-202509-I/en  
   Current in-force ITU recommendation on effects of building materials and structures on radio-wave propagation from 1 MHz to 450 GHz. Use the current official edition when refining material models.

5. **S. Y. Jun et al., NIST — “Penetration Loss at 60 GHz for Indoor-to-Indoor and Outdoor-to-Indoor Mobile Scenarios”**  
   https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=929131  
   Reports material complex-permittivity values at 60.5 GHz for air, metal, wood, plasterboard and glass, and demonstrates that penetration loss depends strongly on material construction and incidence angle. The numeric values in `material_db.yaml` for wood/plasterboard/glass/metal are from its 60.5 GHz table.

6. **ITU-R P.2040 material model parameters (previous edition table)**  
   https://www.itu.int/rec/R-REC-P.2040  
   The initial concrete conductivity power-law coefficients in `material_db.yaml` are tagged as previous-edition model parameters and must be checked against P.2040-4 before hardware-grade claims.

## RF ray tracing and target scattering

7. **NVIDIA Sionna RT — Radar Cross Section API**  
   https://nvlabs.github.io/sionna/rt/api/rcs.html  
   Useful reference for directional point-scatterer radar cross section and bistatic TX-scatterer-RX path formulation. Its documented supported interactions and limitations must be checked before using it as a substitute engine.

8. **NVIDIA Sionna RT — RCS tutorial**  
   https://nvlabs.github.io/sionna/rt/tutorials/RCS.html  
   Demonstrates how a target can be represented with scattering points and how target geometry can occlude propagation paths. It is an optional later backend, not a required dependency for tonight.

## Reference use policy

- Do not cite a source for a parameter it does not provide.
- Keep manufacturer values separate from board-level and configuration-specific values.
- Record exact source revision/date in future calibration reports.
- Material properties vary with moisture, thickness, composition, roughness, polarization and incidence angle.
- Human RCS is not a universal scalar. The range in `radar_config.yaml` is a sensitivity prior, not a published measurement.
