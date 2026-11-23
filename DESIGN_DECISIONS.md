# Design decisions for the first build

## Why one ODS-class radar?

The first build uses one 60 GHz 3-TX/4-RX ODS-style radar reference, benchmarked against the TI IWR6843ISK-ODS because it offers a documented 2D MIMO starting point and wide azimuth/elevation coverage without introducing a second radar, synchronization problem or external capture board. The software must keep this profile replaceable by another vendor device or a custom PCB; it is not a vendor lock-in. This minimizes project complexity and is a practical starting point, not a proven minimum-price bill of materials. The physical EVM is a legacy reference and procurement availability/pricing must be checked separately before buying.

## Why not choose the standard long-range ISK antenna?

The ODS reference has a much wider elevation field of view (120° vs 30° listed for the standard ISK), which is more relevant to a room scene with a person standing, sitting and lying on the floor. The trade-off is lower directivity/range emphasis and relatively sparse 12-channel angular sampling.

## Why not add a second radar tonight?

A second radar adds timing, coordinate calibration, fusion and cost. It should only be added after the single-sensor simulator and validation pipeline work.

## Why not use a deep-learning fall model tonight?

The priority is to make the scene-to-RF-to-ADC-to-DSP chain trustworthy. A rule-based baseline and labelled events are enough for the first end-to-end simulation. Train a classifier only after the synthetic data generator has passed validation and a group-wise held-out split is available.

## Why the visible avatar is separate from the internal skeleton

The 17-joint pose is a compact animation and collision/control representation. The user-facing Scene panel renders a complete rigged human mesh. The radar pipeline receives body scattering proxies generated from the pose, not the visible mesh vertices unless the scattering model explicitly samples them. The ground-truth mesh must never be copied into radar detections.

## Important 3D-imaging limitation

A 12-virtual-channel array produces sparse, aperture-limited angular information. The point cloud is not a camera image. Any occupancy/voxel output is a radar-derived estimate with missing/uncertain regions marked. Dense camera-like 3D reconstruction is a later research milestone requiring more measured spatial information, array aperture, bandwidth, multiple viewpoints or stronger priors.
