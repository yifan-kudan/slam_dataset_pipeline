// Headless localizer-map builder for the v1.1.0 release.
//
// build_map in the release is a Polyscope GUI with no CLI (its --help prints nothing), but
// the whole build pipeline is exported by the shipped shared libraries:
//
//   PlaneMap::construct(ConstructConfig, points)   -> tracking plane map
//   Bbs3dMap::construct(ConstructConfig, points)   -> global relocalization voxel pyramid
//   LocalizerMap{plane, bbs}.save(path, overwrite) -> the .kdm both nodes take as `map`
//
// This reproduces that sequence without a window. Points are read with PDAL, whose driver
// is inferred from the extension - so .las, .pcd and .ply (including the vertex-only PLYs
// that crash the GUI's Assimp mesh loader) all work as input.
//
// Build and run via build_map_headless.sh, which compiles it inside the mapping container.

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <kd/geometry/voxel_filter.hpp>
#include <kd/mapping/bbs3d_map.hpp>
#include <kd/mapping/localizer_map.hpp>
#include <kd/mapping/plane_map.hpp>

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
  float downsample = 0.f;  // 0 disables
  float voxel_size = 0.25f;
  float max_plane_sigma = 0.01f;
  float min_axis_sigma = 0.02f;
  double bbs_min_level_res = 0.f;  // 0 -> voxel_size * bbs_voxel_factor
  int bbs_voxel_factor = 2;
  int bbs_max_level = 7;
  bool overwrite = false;
  bool skip_bbs = false;
};

void usage()
{
  std::cout
    << "usage: build_map_headless --input <cloud> --output <map.kdm> [options]\n\n"
       "  --input <path>              .las / .pcd / .ply point cloud (driver inferred)\n"
       "  --output <path>             output localizer map (.kdm)\n"
       "  --downsample <m>            voxel downsample of input points (0 = off, default 0)\n"
       "  --voxel-size <m>            plane map min voxel size (default 0.25)\n"
       "  --max-plane-sigma <m>       plane fit tolerance (default 0.01)\n"
       "  --min-axis-sigma <m>        minimum axis sigma (default 0.02)\n"
       "  --bbs-voxel-factor <n>      BBS finest res = voxel-size * n (default 2)\n"
       "  --bbs-min-level-res <m>     BBS finest resolution, overrides --bbs-voxel-factor\n"
       "  --bbs-max-level <n>         BBS pyramid levels above finest (default 7)\n"
       "  --skip-bbs                  plane map only (relocalizer will refuse the map)\n"
       "  --overwrite                 overwrite an existing output file\n";
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
      args.downsample = std::stof(value());
    } else if (flag == "--voxel-size") {
      args.voxel_size = std::stof(value());
    } else if (flag == "--max-plane-sigma") {
      args.max_plane_sigma = std::stof(value());
    } else if (flag == "--min-axis-sigma") {
      args.min_axis_sigma = std::stof(value());
    } else if (flag == "--bbs-voxel-factor") {
      args.bbs_voxel_factor = std::stoi(value());
    } else if (flag == "--bbs-min-level-res") {
      args.bbs_min_level_res = std::stod(value());
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

    if (args.downsample > 0.f) {
      kd::VoxelFilter filter;
      filter.calculate(points, args.downsample);
      filter.filter_inplace(points);
      std::cout << "downsampled at " << args.downsample << " m -> " << points.size() << " points"
                << std::endl;
      if (points.empty()) {
        throw std::runtime_error("downsampling removed every point");
      }
    }

    kd::PlaneMap::ConstructConfig plane_config;
    plane_config.min_voxel_size = args.voxel_size;
    plane_config.max_plane_sigma = args.max_plane_sigma;
    plane_config.min_axis_sigma = args.min_axis_sigma;
    std::cout << "building plane map (voxel " << plane_config.min_voxel_size << " m, plane sigma "
              << plane_config.max_plane_sigma << ")" << std::endl;
    kd::LocalizerMap map;
    map.plane_map = kd::PlaneMap::construct(plane_config, points);

    if (!args.skip_bbs) {
      kd::Bbs3dMap::ConstructConfig bbs_config;
      bbs_config.min_level_res = args.bbs_min_level_res > 0.
                                   ? args.bbs_min_level_res
                                   : args.voxel_size * args.bbs_voxel_factor;
      bbs_config.max_level = args.bbs_max_level;
      const double coarsest = bbs_config.min_level_res * (1 << bbs_config.max_level);
      std::cout << "building 3D-BBS map (finest " << bbs_config.min_level_res << " m, max level "
                << bbs_config.max_level << ", coarsest " << coarsest << " m)" << std::endl;
      map.bbs3d_map = kd::Bbs3dMap::construct(bbs_config, points);
      std::cout << "  top level: " << map.bbs3d_map.top_level() << ", occupied voxels at level 0: "
                << map.bbs3d_map.occupied_points(0).size() << std::endl;
    } else {
      std::cout << "skipping 3D-BBS map (kd_relocalizer_node will reject this map)" << std::endl;
    }

    map.save(args.output, args.overwrite);
    std::cout << "wrote " << args.output << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
