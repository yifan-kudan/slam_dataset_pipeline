// Headless localizer-map builder for `main` (.kdm file version 8).
//
// The shipped build_map is a Polyscope GUI: its only CLI is --point-cloud / --localizer-map
// to PRELOAD an asset, there is no --output, so a batch build needs this wrapper. The whole
// pipeline is exported by the shared libraries, and this mirrors exactly what the GUI's
// "Build localizer map" button does (build_map.cpp, start_plane_and_bbs_build):
//
//   map.persistent_map = PointDistributionMap::construct(points, voxel_size)
//   map.plane_map      = StaticPlaneMap::construct(persistent_map, submap_depth, config)
//   map.bbs3d_map      = Bbs3dMap::construct(bbs3d_config, persistent_map)
//   map.save(path, overwrite)
//
// This differs from the v1.1.1 builder (build_map_headless.cpp) in a way that matters: on
// v1.1.1 the plane and BBS maps were each built from the RAW POINTS. On main both are built
// from the PointDistributionMap, which is itself voxelised at --voxel-size. So --voxel-size
// now sets the resolution of everything downstream, and the .kdm carries the persistent map
// as a fourth section.
//
// persistent_map_identity is deliberately left empty (id "", generation 0), which is what the
// GUI produces for a map built from a point cloud. An id is only assigned when the map is
// paired with a LifelongMap (.kdllm).
//
// Defaults are main's library defaults, NOT the v1.1.1 campaign's values. Two differ and both
// change the map: voxel_size 0.25 -> 0.2, and plane sigma 0.01 -> 0.03. On the v1.1.1 pipeline
// loosening the plane sigma to 0.05 dropped the peak ray inlier ratio from 0.93 to 0.42, so
// --plane-sigma is worth sweeping -- but the geometry pipeline changed too (persistent map,
// max_geometry_voxel_size, treat_unobserved_as_empty), so that result may not transfer.
// Build with defaults first; sweep second.
//
// Compile and run via build_map_headless_v8.sh.

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <kd/geometry/voxel_filter.hpp>
#include <kd/mapping/bbs3d_map.hpp>
#include <kd/mapping/localizer_map.hpp>
#include <kd/mapping/point_distribution_map.hpp>
#include <kd/mapping/static_plane_map.hpp>

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>

namespace
{

struct Args
{
  std::string input;
  std::string output;
  double downsample = 0.;  // 0 disables
  double voxel_size = 0.2;
  int submap_depth = 6;
  float plane_sigma = 0.03f;
  float line_sigma = 0.03f;
  float volume_z_value = 3.f;
  double max_geometry_voxel_size = 2.;
  bool treat_unobserved_as_empty = true;
  int bbs_min_level_voxel_factor = 2;
  int bbs_max_level = 6;
  bool overwrite = false;
  bool skip_bbs = false;
};

void usage()
{
  std::cout
    << "usage: build_map_headless_v8 --input <cloud> --output <map.kdm> [options]\n\n"
       "  --input <path>                 .las / .laz / .pcd / .ply cloud (driver inferred)\n"
       "  --output <path>                output localizer map (.kdm, file version 8)\n"
       "  --downsample <m>               voxel downsample of input points (0 = off, default 0)\n"
       "  --voxel-size <m>               persistent-map voxel size (default 0.2)\n"
       "  --submap-depth <n>             plane submap depth; submap = voxel*2^n (default 6)\n"
       "  --plane-sigma <m>              plane fit tolerance (default 0.03)\n"
       "  --line-sigma <m>               line fit tolerance (default 0.03)\n"
       "  --volume-z-value <z>           volume z-value (default 3)\n"
       "  --max-geometry-voxel-size <m>  subdivide lines/planes above this (default 2)\n"
       "  --no-unobserved-as-empty       disable treat_unobserved_as_empty (default on)\n"
       "  --bbs-voxel-factor <n>         BBS finest res = voxel-size * n (default 2)\n"
       "  --bbs-max-level <n>            BBS pyramid levels above finest (default 6)\n"
       "  --skip-bbs                     plane map only (kd_relocalizer will refuse it)\n"
       "  --overwrite                    overwrite an existing output file\n";
}

bool parse(int argc, char** argv, Args& args)
{
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + flag);
      }
      return argv[++i];
    };
    if (flag == "--input") {
      args.input = value();
    } else if (flag == "--output") {
      args.output = value();
    } else if (flag == "--downsample") {
      args.downsample = std::stod(value());
    } else if (flag == "--voxel-size") {
      args.voxel_size = std::stod(value());
    } else if (flag == "--submap-depth") {
      args.submap_depth = std::stoi(value());
    } else if (flag == "--plane-sigma") {
      args.plane_sigma = std::stof(value());
    } else if (flag == "--line-sigma") {
      args.line_sigma = std::stof(value());
    } else if (flag == "--volume-z-value") {
      args.volume_z_value = std::stof(value());
    } else if (flag == "--max-geometry-voxel-size") {
      args.max_geometry_voxel_size = std::stod(value());
    } else if (flag == "--no-unobserved-as-empty") {
      args.treat_unobserved_as_empty = false;
    } else if (flag == "--bbs-voxel-factor") {
      args.bbs_min_level_voxel_factor = std::stoi(value());
    } else if (flag == "--bbs-max-level") {
      args.bbs_max_level = std::stoi(value());
    } else if (flag == "--skip-bbs") {
      args.skip_bbs = true;
    } else if (flag == "--overwrite") {
      args.overwrite = true;
    } else if (flag == "-h" || flag == "--help") {
      usage();
      return false;
    } else {
      throw std::runtime_error("unknown argument: " + flag);
    }
  }
  if (args.input.empty() || args.output.empty()) {
    usage();
    throw std::runtime_error("--input and --output are required");
  }
  return true;
}

std::vector<kd::Vector3f> read_points(const std::string& path)
{
  pdal::StageFactory factory;
  const std::string driver = factory.inferReaderDriver(path);
  if (driver.empty()) {
    throw std::runtime_error("no PDAL reader for " + path);
  }
  pdal::Stage* reader = factory.createStage(driver);
  if (reader == nullptr) {
    throw std::runtime_error("failed to create " + driver);
  }
  pdal::Options options;
  options.add("filename", path);
  reader->setOptions(options);

  pdal::PointTable table;
  reader->prepare(table);
  const pdal::PointViewSet views = reader->execute(table);

  std::vector<kd::Vector3f> points;
  for (const auto& view : views) {
    points.reserve(points.size() + view->size());
    for (pdal::PointId i = 0; i < view->size(); ++i) {
      points.emplace_back(
        static_cast<float>(view->getFieldAs<double>(pdal::Dimension::Id::X, i)),
        static_cast<float>(view->getFieldAs<double>(pdal::Dimension::Id::Y, i)),
        static_cast<float>(view->getFieldAs<double>(pdal::Dimension::Id::Z, i)));
    }
  }
  std::cout << "  driver: " << driver << std::endl;
  return points;
}

void print_extent(const std::vector<kd::Vector3f>& points)
{
  if (points.empty()) {
    return;
  }
  kd::Vector3f lo = points.front();
  kd::Vector3f hi = points.front();
  for (const auto& p : points) {
    lo = lo.cwiseMin(p);
    hi = hi.cwiseMax(p);
  }
  const kd::Vector3f size = hi - lo;
  const std::ios::fmtflags flags = std::cout.flags();
  const std::streamsize precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(1) << "  extent: " << size.x() << " x " << size.y()
            << " x " << size.z() << " m  (min " << lo.x() << ", " << lo.y() << ", " << lo.z()
            << ")" << std::endl;
  std::cout.flags(flags);
  std::cout.precision(precision);
}

}  // namespace

int main(int argc, char** argv)
{
  Args args;
  try {
    if (!parse(argc, argv, args)) {
      return 0;
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }

  try {
    std::cout << "reading " << args.input << std::endl;
    std::vector<kd::Vector3f> points = read_points(args.input);
    if (points.empty()) {
      throw std::runtime_error("input cloud has no points");
    }
    std::cout << "  points: " << points.size() << std::endl;
    print_extent(points);

    if (args.downsample > 0.) {
      kd::VoxelFilter filter;
      filter.calculate(points, args.downsample);
      filter.filter_inplace(points);
      std::cout << "downsampled at " << args.downsample << " m -> " << points.size() << " points"
                << std::endl;
      if (points.empty()) {
        throw std::runtime_error("downsampling removed every point");
      }
    }

    kd::LocalizerMap map;

    // 1. Persistent map. Everything below is built FROM this, not from the raw points,
    //    so --voxel-size sets the resolution of the whole map.
    std::cout << "building persistent map (voxel " << args.voxel_size << " m)" << std::endl;
    map.persistent_map = kd::PointDistributionMap::construct(points, args.voxel_size);
    std::cout << "  occupied voxels: " << map.persistent_map.voxel_count() << std::endl;

    // 2. Plane map. submap width = voxel_size * 2^submap_depth.
    kd::StaticPlaneMap::ConstructConfig config;
    config.geom_tolerances.plane_sigma = args.plane_sigma;
    config.geom_tolerances.line_sigma = args.line_sigma;
    config.geom_tolerances.volume_z_value = args.volume_z_value;
    config.max_geometry_voxel_size = args.max_geometry_voxel_size;
    config.treat_unobserved_as_empty = args.treat_unobserved_as_empty;
    std::cout << "building plane map (submap depth " << args.submap_depth << " -> submap "
              << args.voxel_size * (1 << args.submap_depth) << " m, plane sigma "
              << args.plane_sigma << ", max geometry voxel " << args.max_geometry_voxel_size
              << ", unobserved as empty " << (args.treat_unobserved_as_empty ? "on" : "off") << ")"
              << std::endl;
    map.plane_map =
      kd::StaticPlaneMap::construct(map.persistent_map, args.submap_depth, config);
    std::cout << "  submaps: " << map.plane_map.submaps().size() << std::endl;

    // 3. BBS pyramid for global relocalization.
    if (!args.skip_bbs) {
      kd::Bbs3dMap::ConstructConfig bbs_config;
      bbs_config.min_level_voxel_factor = args.bbs_min_level_voxel_factor;
      bbs_config.max_level = args.bbs_max_level;
      const double finest = args.voxel_size * args.bbs_min_level_voxel_factor;
      std::cout << "building 3D-BBS map (finest " << finest << " m, max level "
                << bbs_config.max_level << ", coarsest " << finest * (1 << bbs_config.max_level)
                << " m)" << std::endl;
      map.bbs3d_map = kd::Bbs3dMap::construct(bbs_config, map.persistent_map);
      std::cout << "  top level: " << map.bbs3d_map.top_level()
                << ", occupied voxels at level 0: " << map.bbs3d_map.occupied_points(0).size()
                << std::endl;
    } else {
      std::cout << "skipping 3D-BBS map (kd_relocalizer_node will reject this map)" << std::endl;
    }

    // persistent_map_identity left empty: matches what the GUI writes for a cloud-built map.
    map.save(args.output, args.overwrite);
    std::cout << "wrote " << args.output << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
