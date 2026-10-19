// Headless map builder for the kd-localizer-v2.0.0 release (.kdm file version 9).
//
// 2.0 splits the map in two. The persistent point distribution now lives in a lifelong map
// (.kdllm) and the localizer map (.kdm) carries only the identity, plane map and 3D-BBS map
// derived from it. The shipped tools cover every step except the first: build_map (GUI) is
// the only way to make a .kdllm from points, and build_localizer_map (CLI) turns a .kdllm into
// a .kdm. This wrapper does both headlessly, in the order the GUI does them:
//
//   distribution = PointDistributionMap::construct(points, voxel_size)   "Build lifelong map"
//   lifelong     = LifelongMap::construct(distribution, config)          (mints a UUID, gen 0)
//   localizer    = lifelong.to_localizer_map()                           "Build localizer map"
//   localizer.save(kdm); lifelong.save(kdllm)                            "Save maps"
//
// to_localizer_map() is what build_localizer_map calls, and it builds the plane and BBS maps
// from the same distribution with the construct config stored in the lifelong map, so the two
// files always share one identity and one set of build parameters. The .kdllm is kept because
// 2.0's update_lifelong_map applies a run's session data (.kds) to it, not to the .kdm.
//
// Differences from build_map_headless_v8.cpp (main at 2026-08-25): LocalizerMap lost its
// persistent_map section, and the identity is no longer left empty -- LifelongMap::construct
// assigns one, exactly as the GUI does for a map built from a point cloud.
//
// Defaults are the 2.0 library / GUI defaults, the same values the v8 builder used: voxel 0.2,
// submap depth 6, plane and line sigma 0.03, BBS finest level 2x voxel, 6 levels above it.
// The output layout matches the GUI's default save: <dir>/localizer_map.kdm and
// <dir>/localizer_map_lifelong.kdllm.
//
// Compile and run via build_map_headless_v9.sh.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <kd/geometry/voxel_filter.hpp>
#include <kd/map_maintenance/lifelong_map.hpp>
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
  std::string lifelong;  // empty -> <output stem>_lifelong.kdllm, as the GUI names it
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
};

void usage()
{
  std::cout
    << "usage: build_map_headless_v9 --input <cloud> --output <map.kdm> [options]\n\n"
       "  --input <path>                 .las / .laz / .pcd / .ply cloud (driver inferred)\n"
       "  --output <path>                output localizer map (.kdm, file version 9)\n"
       "  --lifelong <path>              output lifelong map (default <stem>_lifelong.kdllm)\n"
       "  --downsample <m>               voxel downsample of input points (0 = off, default 0)\n"
       "  --voxel-size <m>               distribution-map voxel size (default 0.2)\n"
       "  --submap-depth <n>             plane submap depth; submap = voxel*2^n (default 6)\n"
       "  --plane-sigma <m>              plane fit tolerance (default 0.03)\n"
       "  --line-sigma <m>               line fit tolerance (default 0.03)\n"
       "  --volume-z-value <z>           volume z-value (default 3)\n"
       "  --max-geometry-voxel-size <m>  subdivide lines/planes above this (default 2)\n"
       "  --no-unobserved-as-empty       disable treat_unobserved_as_empty (default on)\n"
       "  --bbs-voxel-factor <n>         BBS finest res = voxel-size * n (default 2)\n"
       "  --bbs-max-level <n>            BBS pyramid levels above finest (default 6)\n"
       "  --overwrite                    overwrite existing output files\n";
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
    } else if (flag == "--lifelong") {
      args.lifelong = value();
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
  if (args.lifelong.empty()) {
    const std::filesystem::path out(args.output);
    args.lifelong = (out.parent_path() / (out.stem().string() + "_lifelong.kdllm")).string();
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

double seconds_since(std::chrono::steady_clock::time_point t0)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
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
    // Refuse up front rather than after minutes of building: save() checks too, but only at
    // the end, and the .kdm is written before the .kdllm.
    if (!args.overwrite) {
      for (const auto& p : {args.output, args.lifelong}) {
        if (std::filesystem::exists(p)) {
          throw std::runtime_error(p + " exists (pass --overwrite)");
        }
      }
    }

    auto t0 = std::chrono::steady_clock::now();
    std::cout << "reading " << args.input << std::endl;
    std::vector<kd::Vector3f> points = read_points(args.input);
    if (points.empty()) {
      throw std::runtime_error("input cloud has no points");
    }
    std::cout << "  points: " << points.size() << "  (" << seconds_since(t0) << " s)"
              << std::endl;
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

    // 1. Point distribution -> lifelong map. Everything below is built FROM the distribution,
    //    not from the raw points, so --voxel-size sets the resolution of the whole map.
    t0 = std::chrono::steady_clock::now();
    std::cout << "building point distribution map (voxel " << args.voxel_size << " m)"
              << std::endl;
    const kd::PointDistributionMap distribution =
      kd::PointDistributionMap::construct(points, args.voxel_size);
    std::cout << "  occupied voxels: " << distribution.voxel_count() << "  ("
              << seconds_since(t0) << " s)" << std::endl;
    points.clear();
    points.shrink_to_fit();

    kd::LifelongMap::Config config;
    config.localizer_construct.submap_depth = static_cast<size_t>(args.submap_depth);
    auto& plane = config.localizer_construct.plane_map_config;
    plane.geom_tolerances.plane_sigma = args.plane_sigma;
    plane.geom_tolerances.line_sigma = args.line_sigma;
    plane.geom_tolerances.volume_z_value = args.volume_z_value;
    plane.max_geometry_voxel_size = args.max_geometry_voxel_size;
    plane.treat_unobserved_as_empty = args.treat_unobserved_as_empty;
    auto& bbs = config.localizer_construct.bbs3d_map_config;
    bbs.min_level_voxel_factor = args.bbs_min_level_voxel_factor;
    bbs.max_level = args.bbs_max_level;

    const kd::LifelongMap lifelong = kd::LifelongMap::construct(distribution, config);
    std::cout << "lifelong map id " << lifelong.id << " generation "
              << lifelong.persistent_map_generation << std::endl;

    // 2. Plane map + 3D-BBS, exactly as build_localizer_map derives them.
    const double finest = args.voxel_size * args.bbs_min_level_voxel_factor;
    std::cout << "building localizer map (submap " << args.voxel_size * (1 << args.submap_depth)
              << " m, plane sigma " << args.plane_sigma << ", max geometry voxel "
              << args.max_geometry_voxel_size << ", unobserved as empty "
              << (args.treat_unobserved_as_empty ? "on" : "off") << "; BBS finest " << finest
              << " m, max level " << args.bbs_max_level << ", coarsest "
              << finest * (1 << args.bbs_max_level) << " m)" << std::endl;
    t0 = std::chrono::steady_clock::now();
    const kd::LocalizerMap localizer = lifelong.to_localizer_map();
    std::cout << "  BBS top level: " << localizer.bbs3d_map.top_level()
              << ", BBS occupied voxels at level 0: " << localizer.bbs3d_map.occupied_points(0).size()
              << "  (" << seconds_since(t0) << " s)" << std::endl;

    localizer.save(args.output, args.overwrite);
    std::cout << "wrote " << args.output << std::endl;
    lifelong.save(args.lifelong, args.overwrite);
    std::cout << "wrote " << args.lifelong << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
