# Third-party notices

This project keeps third-party implementations in reproducible external
checkouts. Exact repository revisions are recorded in
`config/upstreams.lock.json`; their source is not relicensed as part of this
project.

## LingBot-Map

- Source: https://github.com/robbyant/lingbot-map
- Locked commit: `7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`
- License: Apache License 2.0
- Local license after fetch: `third_party/lingbot-map/LICENSE.txt`

The integration imports the official model and demo helper functions. Local
code only serializes model outputs into the navigation artifact contract.

## Segment Anything Model 3 (SAM3 / SAM3.1)

- Source: https://github.com/facebookresearch/sam3
- Locked commit: `46957e47805eaa273f4aa7bbbd25a88bca9108ce`
- License: SAM License (not Apache-2.0)
- Local license after fetch: `third_party/sam3/LICENSE`

SAM3 checkpoints are distributed separately and require the user to request
access from the official model host. Repository-source availability does not
grant checkpoint access or replace the checkpoint terms.

The integration calls the official video predictor session API. It does not
copy or modify the SAM3 detector/tracker architecture.

## Navigation2

- Source: https://github.com/ros-navigation/navigation2
- Locked Humble commit: `3c3db59d6969d8ecee8e68468693d006397f4a0c`
- License: Apache License 2.0

Nav2 is consumed as the official ROS 2 Humble package. This project only sends
reviewed `NavigateToPose` goals and consumes action feedback/results.

## Habitat-Sim

- Source: https://github.com/facebookresearch/habitat-sim
- Locked tag/commit: `v0.3.3` / `acbe6f4922e68145e401e55c30f9dfea460a3f24`
- License: MIT

Habitat is restricted to data capture, simulation, display, and evaluation.
Its depth, semantic ground truth, pose ground truth, and navmesh are prohibited
from the formal LingBot mapping, place generation, and navigation-control data
paths.

## TurtleBot3 Simulations

- Source: https://github.com/ROBOTIS-GIT/turtlebot3_simulations
- Locked Humble commit: `a35a56c8b04877dc89772b598084d8ce648a9023`
- License: Apache License 2.0

The formal simulation launch reuses the official Gazebo differential-drive
robot, sensors, and world launch rather than implementing a trajectory player.
