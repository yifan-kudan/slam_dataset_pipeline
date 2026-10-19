// Dump a built .kdm's leaf voxels to CSV so the map can be plotted without a GUI.
//
// The release's build_map viewer is the only thing that renders a .kdm, and it needs a window.
// For reports and headless inspection what is actually wanted is the geometry the localizer
// matches against - the plane voxels - as plain numbers.
//
//   map_dump <map.kdm> [--planes-only] [--stride N]
//
// Writes CSV to stdout: x,y,z,nx,ny,nz,size,type,confidence
//   type: 0 = empty (point/line), 1 = plane, 2 = cluster
//   plane voxels carry the fitted plane position and normal; for the other types the voxel
//   centre is written and the normal is zero.
//
// Run via scripts/mapping/map_dump.sh inside the mapping container.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include <kd/mapping/localizer_map.hpp>
#include <kd/mapping/plane_map.hpp>

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << "usage: map_dump <map.kdm> [--planes-only] [--stride N]" << std::endl;
    return 1;
  }
  const std::string path = argv[1];
  bool planes_only = false;
  size_t stride = 1;
  for (int i = 2; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--planes-only") {
      planes_only = true;
    } else if (flag == "--stride" && i + 1 < argc) {
      stride = static_cast<size_t>(std::atoi(argv[++i]));
      if (stride == 0) stride = 1;
    } else {
      std::cerr << "unknown argument: " << flag << std::endl;
      return 1;
    }
  }

  try {
    const kd::LocalizerMap map = kd::LocalizerMap::load(path);
    const kd::PlaneMap& plane_map = map.plane_map;

    std::printf("x,y,z,nx,ny,nz,size,type,confidence\n");
    size_t index = 0;
    plane_map.for_each_leaf(plane_map.root(), [&](kd::PlaneMap::ConstHandle leaf) {
      const kd::PlaneVoxel& voxel = *leaf;
      const bool is_plane = voxel.geom.type() == kd::GeometryEstimate::Type::Plane;
      if (planes_only && !is_plane) {
        return;
      }
      if (index++ % stride != 0) {
        return;
      }
      int type = 0;
      switch (voxel.geom.type()) {
        case kd::GeometryEstimate::Type::Empty: type = 0; break;
        case kd::GeometryEstimate::Type::Plane: type = 1; break;
        case kd::GeometryEstimate::Type::Cluster: type = 2; break;
      }
      // position()/normal() throw unless the estimate carries them; fall back to the voxel centre.
      kd::Vector3f p;
      kd::Vector3f n = kd::Vector3f::Zero();
      if (type == 0) {
        const auto c = leaf.center();
        p = kd::Vector3f(
          static_cast<float>(c.x()), static_cast<float>(c.y()), static_cast<float>(c.z()));
      } else {
        p = voxel.geom.position();
        if (is_plane) {
          n = voxel.geom.normal();
        }
      }
      std::printf(
        "%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.3f,%d,%.3f\n", p.x(), p.y(), p.z(), n.x(), n.y(), n.z(),
        leaf.size(), type, voxel.confidence);
    });
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
