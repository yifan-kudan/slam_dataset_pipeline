# FAST-LIO source changes this pipeline expects

`laserMapping.patch` is the diff this pipeline is run against, taken from the
FAST_LIO_ROS2 working tree. Four changes: three about the *saved map*, and one --
`common.imu_queue_size` -- that does change the estimate, because without it the estimate
silently depends on how fast the bag is played.

* **`publish.map_period_ms`** (new parameter, default 1000 ms — unchanged behaviour).
  The timer that appends the current scan to `pcl_wait_pub`, i.e. the cloud `/map_save`
  writes out, fires once a second while the LiDAR runs at 100 ms, so the saved map keeps
  one scan in ten. Lower it to densify the map; the estimate is unaffected.
* **`publish.map_publish_en`** (new parameter, default `true` — unchanged behaviour).
  `publish_map()` does two things: it appends the current scan to `pcl_wait_pub`, which is
  what `/map_save` writes, and it publishes the *whole accumulated cloud* on `/Laser_map`.
  The second cost grows with the square of the run length — Hilti `site2_robot_1`
  accumulates 28.2 M points, so at `map_period_ms: 370` its last messages are ~900 MB
  each, every 370 ms. That starves the node's own callback thread, the reader stops
  draining the shared-memory port, and the best-effort cloud writer then drops: 2906 of
  6994 frames, with the loss ramping up in step with the cloud. Setting this false keeps
  the accumulation (so the saved map is identical) and skips the publication, which has no
  subscriber in a headless batch run. The Hilti configs set it false.
* **`common.imu_queue_size`** (new parameter, default 10 — unchanged behaviour).
  Upstream subscribes to `/imu` with a bare `10`, i.e. reliable `KeepLast(10)`. That is
  25 ms of buffer at the Hilti handheld rig's 400 Hz, on the same single-threaded executor
  as a LiDAR callback that takes 4–8 ms per scan with spikes — so the queue overflows and
  the reader loses IMU samples, with no log line anywhere. The signature is that the
  **estimate depends on playback rate even at 100 % cloud delivery, and only on the 400 Hz
  rig**: at `-r 0.3` the same bags scored 13x, 3.5x and 2.3x better while the 200 Hz robot
  was identical to three significant figures. With `imu_queue_size: 2000`, `-r 1.0`
  reproduces the `-r 0.3` result — on `site1_handheld_5` to four significant figures
  (27.5 / 40.4 cm, rel 1.188 % either way). `imu_cbk` only pushes onto a deque, so depth
  costs memory and nothing else. The Hilti configs set 2000.
* **A comment, deliberately no code change, on the cloud subscription's QoS.** The
  ~50 % cloud loss on this host is writer-side (Fast DDS shared memory), so forcing
  `reliable` here would paper over it while diverging from a real LiDAR driver, which
  publishes best-effort. The fix belongs in the transport:
  `config/dds/fastdds_shm_big.xml`, which `src/batch_fastlio.py` exports for the node
  and the bag player alike.

Regenerate with:

    git -C <FAST_LIO_ROS2> diff src/laserMapping.cpp > scripts/fastlio/laserMapping.patch
