// ----------------------------------------------------------------------------
//
//    (Work In Progress) Converter for WotS 4 files (Characters and Maps).
//
// ----------------------------------------------------------------------------

#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstddef>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#endif

// NIF importer.
#include "nif/niflib.h"
#include "nif/gen/Header.h"
#include "nif/obj/NiNode.h"
#include "nif/obj/NiMesh.h"
#include "nif/obj/NiSwitchNode.h"
#include "nif/obj/NiPointLight.h"
#include "nif/obj/NiDirectionalLight.h"
#include "nif/obj/NiMeshModifier.h"
#include "nif/obj/NiSkinningMeshModifier.h"
#include "nif/obj/NiDataStream.h"
#include "nif/obj/NiSourceTexture.h"
#include "nif/obj/NiPixelData.h"
#include "nif/obj/NiStringExtraData.h"
#include "nif/obj/NiIntegerExtraData.h"
#include "nif/obj/NiTexturingProperty.h"
#include "nif/obj/NiAlphaProperty.h"
#include "nif/obj/NiMaterialProperty.h"
#include "nif/obj/NiStencilProperty.h"
#include "nif/obj/NiVertexColorProperty.h"
#include "nif/obj/NiSpecularProperty.h"

// Convert float16 to float32.
#include "half/half.hpp"
using half_float::half;

// Decompress DXT1/DXT5.
#define S3TC_IMPLEMENTATION
#include "s3tc/s3tc.h"

// Save DDS.
#include "dds.h"

// Read keyframe files (animations).
#include "kf.h"

// STB image.
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

// glTF exporter.
#define TINYGLTF_NOEXCEPTION
#define TINYGLTF_IMPLEMENTATION
#include "tinygltf/tiny_gltf.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// ----------------------------------------------------------------------------

#define DOJIMA_LOG(x) std::cerr << x << std::endl

#ifndef NDEBUG
#define DOJIMA_QUIET_LOG(x)  DOJIMA_LOG(x)
#else
#define DOJIMA_QUIET_LOG(x)  if (static bool b=false; !b && (b=true)) DOJIMA_LOG("(quiet log) " << x)
#endif

// ----------------------------------------------------------------------------

namespace {

// Display NIF header information at loading.
static constexpr bool kDisplayHeader = true;

static constexpr bool kShowDebugRefArborescence = false;

// Output external PNG for debugging.
static constexpr bool kDebugOutputPNG = false; //
static constexpr bool kDebugOutputDDS = false; //

// Output name for accessors.
static constexpr bool kNameAccessors = true;

// glTF output options.
static constexpr bool kGLTFEmbedImages = true;
static constexpr bool kGLTFEmbedBuffers = true;
static constexpr bool kGLTFPrettyPrint = true;
static constexpr bool kGLTFWriteBinary = false;

// Keep every LOD of the characters parts (see --all-lods), instead of only the
// highest quality one.
static bool gKeepAllLODs = false;

// Keyframe files (*.kf) whose sequences are exported as glTF animations.
static std::vector<std::string> gAnimFilenames;

// Sampling rate (frames per second) of the exported animations.
static float gAnimFps = 30.0f;

// Keep the movement accumulated by the root of the animations (see --root-motion),
// instead of animating them in place.
static bool gKeepRootMotion = false;

// OS specific path separator.
static constexpr char kSystemPathSeparator{
#if defined(WIN32) || defined(_WIN32)
  '\\'
#else
  '/'
#endif
};

enum class BufferId : uint8_t {
  INDEX,
  VERTEX,
  // IMAGE, //

  kCount
};

// NIF node keys of interest.
static std::string const kNiDataStreamKey{ "NiDataStream" };
static std::string const kNiMeshKey{ "NiMesh" };
static std::string const kNiPixelDataKey{ "NiPixelData" };
// static std::string const kNiSourceTextureKey{ "NiSourceTexture" };

uint32_t GetPrimitiveType(Niflib::MeshPrimitiveType primType) {
  switch (primType) {
    case Niflib::MESH_PRIMITIVE_TRIANGLES:
    return TINYGLTF_MODE_TRIANGLES;

    case Niflib::MESH_PRIMITIVE_TRISTRIPS:
    return TINYGLTF_MODE_TRIANGLE_STRIP;

    case Niflib::MESH_PRIMITIVE_LINESTRIPS:
    return TINYGLTF_MODE_LINE_STRIP;

    case Niflib::MESH_PRIMITIVE_POINTS:
    return TINYGLTF_MODE_POINTS;

    default:
      DOJIMA_QUIET_LOG( "[Warning] " << __FUNCTION__ << " : primitive type " << primType << " is not supported." );
    return TINYGLTF_MODE_POINTS;
  };
}

// ----------------------------------------------------------------------------
//
// NIF component formats encode their layout in the enum value :
//    0x00CCSSTT  with CC = number of components, SS = bytes per component and
//    TT = the data type (see Niflib::ComponentFormat).
//
// ----------------------------------------------------------------------------

inline int FormatNumComponents(Niflib::ComponentFormat f) {
  return (static_cast<int>(f) >> 16) & 0xff;
}

inline size_t FormatByteSize(Niflib::ComponentFormat f) {
  return static_cast<size_t>(FormatNumComponents(f)) *
         static_cast<size_t>((static_cast<int>(f) >> 8) & 0xff);
}

// Decode one element of a vertex attribute to (at most 4) floats.
// Returns false for unsupported formats.
bool DecodeFormat(Niflib::ComponentFormat f, uint8_t const* src, float out[4]) {
  int const n = FormatNumComponents(f);
  int const type = static_cast<int>(f) & 0xff;
  if (n < 1 || n > 4) {
    return false;
  }

  auto load = [&](auto const* p, auto fn) {
    using T = std::remove_cv_t<std::remove_pointer_t<decltype(p)>>;
    for (int i = 0; i < n; ++i) {
      T v;
      std::memcpy(&v, src + i * sizeof(T), sizeof(T));
      out[i] = fn(v);
    }
    return true;
  };

  if (type >= 0x01 && type <= 0x04) return load((int8_t const*)0, [](int8_t v) { return float(v); });
  if (type >= 0x05 && type <= 0x08) return load((uint8_t const*)0, [](uint8_t v) { return float(v); });
  if (type >= 0x09 && type <= 0x0C) return load((int8_t const*)0, [](int8_t v) { return std::max(float(v) / 127.0f, -1.0f); });
  if (type >= 0x0D && type <= 0x10) return load((uint8_t const*)0, [](uint8_t v) { return float(v) / 255.0f; });
  if (type >= 0x11 && type <= 0x14) return load((int16_t const*)0, [](int16_t v) { return float(v); });
  if (type >= 0x15 && type <= 0x18) return load((uint16_t const*)0, [](uint16_t v) { return float(v); });
  if (type >= 0x19 && type <= 0x1C) return load((int16_t const*)0, [](int16_t v) { return std::max(float(v) / 32767.0f, -1.0f); });
  if (type >= 0x1D && type <= 0x20) return load((uint16_t const*)0, [](uint16_t v) { return float(v) / 65535.0f; });
  if (type >= 0x31 && type <= 0x34) return load((half const*)0, [](half v) { return static_cast<float>(v); });
  if (type >= 0x35 && type <= 0x38) return load((float const*)0, [](float v) { return v; });
  return false;
}

// ----------------------------------------------------------------------------
//  Minimal affine transform helpers (column-major 4x4, as glTF).
// ----------------------------------------------------------------------------

using Mat4 = std::array<double, 16>;

Mat4 Identity() {
  return { 1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1 };
}

Mat4 Mul(Mat4 const& a, Mat4 const& b) {
  Mat4 r{};
  for (int c = 0; c < 4; ++c) {
    for (int row = 0; row < 4; ++row) {
      double v = 0.0;
      for (int k = 0; k < 4; ++k) {
        v += a[k * 4 + row] * b[c * 4 + k];
      }
      r[c * 4 + row] = v;
    }
  }
  return r;
}

// Inverse of an affine transform.
Mat4 AffineInverse(Mat4 const& m) {
  double const a = m[0], b = m[4], c = m[8];
  double const d = m[1], e = m[5], f = m[9];
  double const g = m[2], h = m[6], i = m[10];
  double const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  double const inv = (std::abs(det) > 1e-30) ? 1.0 / det : 0.0;

  // (row, column) = inverse of the upper 3x3.
  double r[3][3] = {
    { (e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv },
    { (f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv },
    { (d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv },
  };
  Mat4 o = Identity();
  for (int col = 0; col < 3; ++col) {
    for (int row = 0; row < 3; ++row) {
      o[col * 4 + row] = r[row][col];
    }
  }
  for (int row = 0; row < 3; ++row) {
    o[12 + row] = -(r[row][0] * m[12] + r[row][1] * m[13] + r[row][2] * m[14]);
  }
  return o;
}

Mat4 FromTRS(tinygltf::Node const& n) {
  Mat4 m = Identity();
  double x = 0, y = 0, z = 0, w = 1;
  if (n.rotation.size() == 4) { x = n.rotation[0]; y = n.rotation[1]; z = n.rotation[2]; w = n.rotation[3]; }
  double sx = 1, sy = 1, sz = 1;
  if (n.scale.size() == 3) { sx = n.scale[0]; sy = n.scale[1]; sz = n.scale[2]; }
  m[0]  = (1 - 2 * (y * y + z * z)) * sx;  m[1]  = (2 * (x * y + z * w)) * sx;      m[2]  = (2 * (x * z - y * w)) * sx;
  m[4]  = (2 * (x * y - z * w)) * sy;      m[5]  = (1 - 2 * (x * x + z * z)) * sy;  m[6]  = (2 * (y * z + x * w)) * sy;
  m[8]  = (2 * (x * z + y * w)) * sz;      m[9]  = (2 * (y * z - x * w)) * sz;      m[10] = (1 - 2 * (x * x + y * y)) * sz;
  if (n.translation.size() == 3) { m[12] = n.translation[0]; m[13] = n.translation[1]; m[14] = n.translation[2]; }
  return m;
}

// Decompose an affine transform to the node's translation / rotation / scale.
void SetTRS(Mat4 const& m, tinygltf::Node *n) {
  n->translation = { m[12], m[13], m[14] };

  double s[3];
  for (int c = 0; c < 3; ++c) {
    s[c] = std::sqrt(m[c*4]*m[c*4] + m[c*4+1]*m[c*4+1] + m[c*4+2]*m[c*4+2]);
    if (s[c] < 1e-12) s[c] = 1.0;
  }
  n->scale = { s[0], s[1], s[2] };

  double const r00 = m[0] / s[0], r10 = m[1] / s[0], r20 = m[2] / s[0];
  double const r01 = m[4] / s[1], r11 = m[5] / s[1], r21 = m[6] / s[1];
  double const r02 = m[8] / s[2], r12 = m[9] / s[2], r22 = m[10] / s[2];

  double x, y, z, w;
  if (double const tr = r00 + r11 + r22; tr > 0.0) {
    double const k = std::sqrt(tr + 1.0) * 2.0;
    w = 0.25 * k; x = (r21 - r12) / k; y = (r02 - r20) / k; z = (r10 - r01) / k;
  } else if ((r00 > r11) && (r00 > r22)) {
    double const k = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
    w = (r21 - r12) / k; x = 0.25 * k; y = (r01 + r10) / k; z = (r02 + r20) / k;
  } else if (r11 > r22) {
    double const k = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
    w = (r02 - r20) / k; x = (r01 + r10) / k; y = 0.25 * k; z = (r12 + r21) / k;
  } else {
    double const k = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
    w = (r10 - r01) / k; x = (r02 + r20) / k; y = (r12 + r21) / k; z = 0.25 * k;
  }
  double const l = std::sqrt(x*x + y*y + z*z + w*w);
  n->rotation = { x / l, y / l, z / l, w / l };
}

// Returns the level of a "LV<n>_" prefixed name (LV1 is the best), or -1.
int ParseLODLevel(std::string const& name) {
  if ((name.size() > 3) && (name[0] == 'L') && (name[1] == 'V') && std::isdigit(static_cast<unsigned char>(name[2]))) {
    size_t pos = 2;
    int level = 0;
    while ((pos < name.size()) && std::isdigit(static_cast<unsigned char>(name[pos]))) {
      level = level * 10 + (name[pos++] - '0');
    }
    if ((pos < name.size()) && (name[pos] == '_')) {
      return level;
    }
  }
  return -1;
}

// Basename of a path (either separator style), without its extension.
std::string FileStem(std::string const& path) {
  size_t const start = path.find_last_of("/\\");
  std::string name = path.substr((start == std::string::npos) ? 0 : start + 1);
  if (size_t const dot = name.find_last_of('.'); dot != std::string::npos) {
    name.erase(dot);
  }
  return name;
}

// Returns true if the path has a ".kf" extension.
bool IsKeyframeFile(std::string const& path) {
  if (path.size() < 3u) {
    return false;
  }
  std::string ext = path.substr(path.size() - 3u);
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
  return ext == ".kf";
}

std::vector<float> MakeSampleTimes(float duration, float fps) {
  std::vector<float> times{ 0.0f };
  if (duration <= 1e-6f) {
    return times;
  }
  for (size_t i = 1; ; ++i) {
    float const t = static_cast<float>(i) / fps;
    if (t >= duration - 1e-4f) {
      break;
    }
    times.push_back(t);
  }
  times.push_back(duration);
  return times;
}

char const* CycleTypeName(kf::CycleType cycle) {
  switch (cycle) {
    case kf::CycleType::kLoop:    return "loop";
    case kf::CycleType::kReverse: return "reverse";
    case kf::CycleType::kClamp:   return "clamp";
  }
  return "unknown";
}

// Convert a NiAlphaProperty to a glTF alpha mode.
//
// WotS4 draws characters' hair and lashes in two passes : an alpha tested one
// (GEQUAL 255 : the opaque texels) and an alpha blended one (LESS 255 : the
// remaining ones). Skin & clothes only enable the blending.
void SetAlphaMode(Niflib::NiAlphaProperty const& prop, tinygltf::Material *material) {
  using AP = Niflib::NiAlphaProperty;
  bool const blend = prop.GetBlendState();
  bool const test = prop.GetTestState();
  auto const func = prop.GetTestFunc();
  double const threshold = prop.GetTestThreshold() / 255.0;

#if 1
  switch (func) {
    case AP::TF_LESS:
      material->alphaMode = "BLEND";
    break;

    case AP::TF_GREATER:
      // (some body parts have this flag set for no clear reasons, we might discard them)
    case AP::TF_GEQUAL:
      material->alphaMode = "MASK";
      material->alphaCutoff = threshold;
    break;

    case AP::TF_ALWAYS:
    default:
      material->alphaMode = "OPAQUE";
    break;
  }
#else
  if (test && (func == AP::TF_GEQUAL)) {
    material->alphaMode = "MASK";
    material->alphaCutoff = threshold;
  } else if (test && (func == AP::TF_GREATER)) {
    material->alphaMode = "MASK";
    material->alphaCutoff = std::min(threshold + 1.0 / 255.0, 1.0); // (strictly greater)
  } else if (blend || (test && ((func == AP::TF_LESS) || (func == AP::TF_LEQUAL)))) {
    material->alphaMode = "BLEND";
  } else {
    material->alphaMode = "OPAQUE";
  }
#endif
}

} // namespace ""

// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  std::string nifFilename;
  bool badArguments = false;
  for (int i = 1; i < argc; ++i) {
    std::string const arg(argv[i]);
    if (arg == "--all-lods") {
      gKeepAllLODs = true;
    } else if (arg == "--root-motion") {
      gKeepRootMotion = true;
    } else if ((arg == "--anim-fps") && (i + 1 < argc)) {
      gAnimFps = static_cast<float>(std::atof(argv[++i]));
    } else if (arg == "--anim") {
      if (i + 1 < argc) {
        gAnimFilenames.push_back(argv[++i]);
      } else {
        badArguments = true;
      }
    } else if (IsKeyframeFile(arg)) {
      gAnimFilenames.push_back(arg);
    } else if (arg.compare(0, 2, "--") == 0) {
      DOJIMA_LOG( "[Error] unknown option : " << arg );
      badArguments = true;
    } else {
      nifFilename = arg;
    }
  }
  if (!(gAnimFps >= 1.0f) || !(gAnimFps <= 240.0f)) {
    DOJIMA_LOG( "[Error] --anim-fps must be between 1 and 240." );
    badArguments = true;
  }
  if (nifFilename.empty() || badArguments) {
    DOJIMA_LOG( "Usage : " << std::endl
                << argv[0] << " [options] nif_file [kf_file...]" << std::endl
                << "  --all-lods      : keep every LOD / skinning variant of the meshes (default : only the best one)." << std::endl
                << "  --anim kf_file  : export the animations of a keyframe file along the model" << std::endl
                << "                    (a '.kf' file given as argument is also exported)." << std::endl
                << "  --anim-fps n    : sampling rate of the exported animations (default : 30)." << std::endl
                << "  --root-motion   : keep the movement accumulated by the animations' root node" << std::endl
                << "                    (default : the animations are played in place)." );
    return EXIT_FAILURE;
  }

  // Create an outputs directory for the transformed NIF file.
  std::string const outputDirectory{
    [](std::string const& s) -> std::string {
      return "_outputs" +
             std::string(1, kSystemPathSeparator) +
             FileStem(s) +
             std::string(1, kSystemPathSeparator);
    }(nifFilename)
  };
  std::filesystem::create_directories(outputDirectory);

  // --------------

  // Retrieve blocks / structure informations from the NIF header.
  Niflib::Header nifHeader = Niflib::ReadHeader(nifFilename);

  // Block type names.
  auto const &blockTypes = nifHeader.getBlockTypes();

  // Type index of each blocks.
  auto const &blockTypeIndex = nifHeader.getBlockTypeIndex();

  // Number of block of each type.
  std::vector<size_t> blockTypeCount(blockTypes.size(), 0);

  // List of node indices from their types.
  std::unordered_map< std::string, std::vector<size_t> > typeToListIndices;
  {
    int current_index = 0;
    for (auto const bti : blockTypeIndex) {
      ++blockTypeCount[bti];
      typeToListIndices[blockTypes[bti]].push_back(current_index++);
    }
  }

  // Block type name to base index.
  std::unordered_map<std::string, size_t> typeToIndex;
  for (size_t i = 0; i < blockTypes.size(); ++i) {
    typeToIndex[blockTypes[i]] = i;
  }

  // Indices to differents datastreams types.
  std::vector<size_t> dataStreamIndices;
  for (auto const& [key, indices] : typeToListIndices) {
    if (key.find(kNiDataStreamKey) != std::string::npos) {
      dataStreamIndices.insert(dataStreamIndices.end(), indices.cbegin(), indices.cend());
    }
  }

  //  Return the number of block for a given block name.
  auto getNumBlocks = [&](std::string const& key) -> size_t {
    auto it = typeToIndex.find(key);
    auto const count = (it == typeToIndex.end()) ? 0 : blockTypeCount.at( it->second );
    return count;
  };

  // Return the list of node indices for a given block name.
  auto getTypeListIndices = [&](std::string const& key) -> std::vector<size_t> {
    auto it = typeToListIndices.find(key);
    return (it == typeToListIndices.end()) ? std::vector<size_t>() : it->second;
  };

  // Display header informations.
  if constexpr (kDisplayHeader) {
    std::cerr <<
      " * Header string : "  << nifHeader.getHeaderString() << std::endl <<
      " * Version : "        << std::hex << "0x" << nifHeader.getVersion() << std::dec << std::endl <<
      " * Endianness : "     << ((nifHeader.getEndianType() == Niflib::ENDIAN_BIG) ? "big" : "little") << std::endl <<
    "";

    std::cerr << " * Block types (" << blockTypes.size() << ") :" << std::endl;
    int bti = 0;
    for (auto const &bt : blockTypes) {
      std::cerr << "      + " << bti << " " << bt << " [" << blockTypeCount[bti] << "], \n";
      ++bti;
    }
  }

  if (nifHeader.getVersion() == 0x1e10003) {
    std::cerr << " * Theorical mesh count : " << getNumBlocks("NiColorExtraData") / 3 << std::endl;

  }

  // --------------

  //
  // WotS4 has 3 data streams per Mesh :
  //    * USAGE_VERTEX_INDEX  : (INDEX)
  //    * USAGE_VERTEX        : (Texcoord, Position, Normal, Binormal, Tangent, BlendIndice, BlendWeight)
  //    * USAGE_USER          : (BONE_PALETTE)
  //
  //  Mesh processing :
  //
  // 1) We create 2 buffers for indices and vertices (we don't use BONE_PALETTE for now).
  //     [ better approach : use one buffer per LODs ]
  //
  // 2) For each NiMesh we create :
  //    * 2 buffer_views for indices and vertices (one per used datastreams).
  //
  // 3) For each mesh's submeshes we create :
  //    * 1 accessors for indices.
  //    * 7 accessors for vertices.
  //
  // Note that "Binormal" are not supported by glTF 2.0 as they can be retrieved
  // from normal + tangent but to avoid further buffer reparameterization we
  // provide them anyways.
  //

  auto nifList = Niflib::ReadNifList(nifFilename);

  /* Show the refs arborescence structure of the Nif file. */
  if constexpr(kShowDebugRefArborescence)
  {
    std::function<void(std::string, Niflib::NiObject*)> showRefs = [&](std::string prefix, auto n) {
      auto ptrs = n->GetRefs();
      for (auto pp : ptrs) {
        DOJIMA_LOG(prefix << "   > " << pp->GetType().GetTypeName()  << " " );
        showRefs(prefix + "  ", pp);
      }
    };

    for (size_t i = 0; i < nifList.size(); ++i) {
      auto node = nifList[i];
      auto ptr = Niflib::DynamicCast<Niflib::NiAVObject>(node);

      DOJIMA_LOG( i << " "  << node->GetType().GetTypeName() );

      if (ptr) {
        showRefs(" ", ptr);
      }
    }

    // return -1;
  }

  // Determine the bytesize of the global Index & Vertex buffers.
  size_t indicesBytesize = 0;
  size_t verticesBytesize = 0;
  for (auto index : dataStreamIndices) {
    auto nifDataStream = Niflib::StaticCast<Niflib::NiDataStream>(nifList[index]);
    auto const usage = nifDataStream->GetUsage();
    size_t const bytesize = nifDataStream->GetNumBytes();

    if (usage == Niflib::USAGE_VERTEX_INDEX) {
      indicesBytesize += bytesize;
    } else if (usage == Niflib::USAGE_VERTEX) {
      verticesBytesize += bytesize; //
    }
  }

  // Note :
  //  When the buffer contains *only* half floating point values its size
  //  should theorically be doubled for converting it to float32.
  //  We use this only as a preallocation scheme, but as for WotS4
  //  the biggest files are float16 maps anyways we always allocate enough.
  verticesBytesize *= 2;

  // *Theorical* uncompressed total bytelength for images.
  size_t constexpr kRawPixelsDefaultSize{ 4 * 1024 * 1024 }; //
  size_t const kNumTextures{ getNumBlocks(kNiPixelDataKey) };
  // size_t const imagesBytesize = kRawPixelsDefaultSize * kNumTextures;

  // Create the base glTF buffers.
  std::vector<tinygltf::Buffer> Buffers(static_cast<size_t>(BufferId::kCount));

  auto &IndexBuffer = Buffers[(int)BufferId::INDEX];
  IndexBuffer.name = "Indices";
  IndexBuffer.data.reserve(indicesBytesize);

  auto &VertexBuffer = Buffers[(int)BufferId::VERTEX];
  VertexBuffer.name = "Vertices";
  VertexBuffer.data.reserve(verticesBytesize);

  // (wip)
  // auto &ImageBuffer = Buffers[(int)BufferId::IMAGE];
  // ImageBuffer.name = "Images";
  // ImageBuffer.data.reserve(imagesBytesize); //

  // Retrieve the number of NIF mesh nodes.
  size_t const kNiMeshCount{ getNumBlocks(kNiMeshKey) };

  // Meshes buffers (only the kept ones, see skippedMeshes).
  std::vector<tinygltf::Mesh> Meshes;
  Meshes.reserve( kNiMeshCount );

  // BufferViews and Accessors are appended as they get created.
  std::vector<tinygltf::BufferView> BufferViews;
  std::vector<tinygltf::Accessor> Accessors;

  // [lambda] Append bytes to a buffer (4 bytes aligned), create the matching
  // bufferView and return its index.
  auto addBufferView{
    [&](BufferId bufferId, void const* bytes, size_t bytesize, int target) -> int {
      auto &buffer = Buffers[static_cast<size_t>(bufferId)];
      buffer.data.resize((buffer.data.size() + 3u) & ~size_t(3));

      tinygltf::BufferView bv;
      bv.buffer = static_cast<int>(bufferId);
      bv.byteOffset = buffer.data.size();
      bv.byteLength = bytesize;
      bv.target = target;

      auto const* src = static_cast<uint8_t const*>(bytes);
      buffer.data.insert(buffer.data.end(), src, src + bytesize);
      BufferViews.push_back(bv);
      return static_cast<int>(BufferViews.size()) - 1;
    }
  };

  // [lambda] Create a tightly packed float attribute accessor from a vector of
  // floats, and returns its index.
  auto addFloatAccessor{
    [&](std::string const& name, std::vector<float> const& values, int type, int numComps, bool bounds) -> int {
      size_t const count = values.size() / numComps;

      tinygltf::Accessor acc;
      acc.name = kNameAccessors ? name : "";
      acc.bufferView = addBufferView(BufferId::VERTEX, values.data(), values.size() * sizeof(float), TINYGLTF_TARGET_ARRAY_BUFFER);
      acc.byteOffset = 0;
      acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
      acc.type = type;
      acc.count = count;

      if (bounds && (count > 0)) {
        acc.minValues.assign(numComps,  std::numeric_limits<double>::max());
        acc.maxValues.assign(numComps, -std::numeric_limits<double>::max());
        for (size_t i = 0; i < count; ++i) {
          for (int c = 0; c < numComps; ++c) {
            double const v = values[i * numComps + c];
            acc.minValues[c] = std::min(acc.minValues[c], v);
            acc.maxValues[c] = std::max(acc.maxValues[c], v);
          }
        }
      }
      Accessors.push_back(acc);
      return static_cast<int>(Accessors.size()) - 1;
    }
  };

  // Materials buffer.
  // Depends on textures, MaterialData & NiMaterialProperty.
  // (one per mesh, sharing the mesh index)
  std::vector<tinygltf::Material> Materials;
  Materials.reserve( kNiMeshCount );

  // [ TODO ]
  // Samplers buffer.
  std::vector<tinygltf::Sampler> Samplers( 1 ); // (0 : default, repeat)

  // Returns the sampler matching a NIF clamp mode, created on demand.
  std::unordered_map<int, int> mapClampToSampler;
  auto getSampler{
    [&](Niflib::TexClampMode mode) -> int {
      if (mode == Niflib::WRAP_S_WRAP_T) {
        return 0;
      }
      if (auto it = mapClampToSampler.find(mode); it != mapClampToSampler.end()) {
        return it->second;
      }
      bool const clampS = (mode == Niflib::CLAMP_S_CLAMP_T) || (mode == Niflib::CLAMP_S_WRAP_T);
      bool const clampT = (mode == Niflib::CLAMP_S_CLAMP_T) || (mode == Niflib::WRAP_S_CLAMP_T);
      tinygltf::Sampler sampler;
      sampler.wrapS = clampS ? TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE : TINYGLTF_TEXTURE_WRAP_REPEAT;
      sampler.wrapT = clampT ? TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE : TINYGLTF_TEXTURE_WRAP_REPEAT;
      Samplers.push_back(sampler);
      return mapClampToSampler[mode] = static_cast<int>(Samplers.size()) - 1;
    }
  };
  // std::map< uint16_t, size_t > mapTexDescFlagToSamplerIndex;

  // -- Texture extraction helpers.

  // Textures buffers.
  // (There are approximatively 4 textures per mesh part or 1 per NiPixelData, but we don't
  // use all of them).
  std::vector<tinygltf::Image> Images;
  std::vector<tinygltf::Texture> Textures;
  Images.reserve( kNumTextures );
  Textures.reserve( kNumTextures );

  // Link a texture's path/name to its Image-Texture index.
  std::unordered_map< std::string, size_t > mapTextureNameToIndex;

  // Buffer to convert DXT compressed texture to raw pixels.
  std::vector<uint8_t> rawPixels(kRawPixelsDefaultSize);

  // Helper to build and uncompress DDS from compressed bytes.
  DDS dds;

  // [lambda] Extract an internal DDS texture and set texture image buffers.
  auto processInternalTexture{
    [&](Niflib::Ref<Niflib::NiPixelData> const& nifPixelData, std::string const& texFilename, size_t* texIndex) -> size_t {
      // Check the texture has not been loaded yet.
      if (auto it = mapTextureNameToIndex.find(texFilename); it != mapTextureNameToIndex.end()) {
        // DOJIMA_QUIET_LOG(texFilename << " already loaded.");
        *texIndex = it->second;
        return true;
      }

      auto const fmt = nifPixelData->GetPixelFormat();
      if ((Niflib::PX_FMT_DXT1 != fmt) &&
          (Niflib::PX_FMT_DXT5 != fmt) &&
          (Niflib::PX_FMT_DXT5_ALT != fmt)) {
        DOJIMA_QUIET_LOG( texFilename << " [" << fmt << "] : non DXT1 / DXT5 textures are not supported yet." );
        return false;
      }
      auto decompressDXT = (Niflib::PX_FMT_DXT1 == fmt) ? s3tc::DecompressDXT1
                                                        : s3tc::DecompressDXT5;

      // Add a new texture data.
      Images.push_back( tinygltf::Image() );
      auto &img = Images.back();

      Textures.push_back( tinygltf::Texture() );
      auto &tex = Textures.back();

      // Update the Texture LUT with the new image/texture index using its full name.
      *texIndex = Images.size() - 1;
      mapTextureNameToIndex[texFilename] = *texIndex;

      // Extract the texture's basename, removing its path and extension.
      auto const texName{ FileStem(texFilename) };

      img.name = texName;
      img.width = static_cast<uint32_t>(nifPixelData->GetWidth());
      img.height = static_cast<uint32_t>(nifPixelData->GetHeight());
      img.component = 4;
      img.bits = 8;
      img.pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;

      // [note] Decompressing DXT to PNG might lose alpha channel information.

      // Transform the DDS compressed data to PNG for gltf 2.0 compatibility.
      {
        // (all mipmaps are flatten on the first level)
        auto const& pixelBytes = nifPixelData->GetMipMaps()[0];

        // [optional] Reconstruct the DDS file from data.
        if (kDebugOutputDDS) {
          uint32_t const levels = nifPixelData->GetNumMipMaps(); //
          uint32_t const mask[4]{
            0x000000ff, // nifPixelData->GetRedMask(),
            0x0000ff00, // nifPixelData->GetGreenMask(),
            0x00ff0000, // nifPixelData->GetBlueMask(),
            0xff000000, // nifPixelData->GetAlphaMask(),
          };
          dds.build(levels, img.width, img.height, fmt, mask, pixelBytes);
          dds.save(outputDirectory + img.name + ".dds");
        }

        // Decompress the internal DDS texture.
        size_t const rowStride = img.component * img.width;
        rawPixels.resize( rowStride * img.height );
        decompressDXT(img.width, img.height, pixelBytes.data(), (uint32_t *)rawPixels.data());

        std::string const uri{outputDirectory + img.name + ".png"};

        // Save image as PNG.
        if constexpr (kDebugOutputPNG) {
          stbi_write_png(uri.c_str(), img.width, img.height, img.component, rawPixels.data(), rowStride);
        }

        {
          // (internal)
          img.uri = uri; //
          img.image.insert(img.image.begin(), rawPixels.begin(), rawPixels.end());
        }
      }

      // Set the texture to the textureMap.
      tex.sampler = 0; //
      tex.source = *texIndex;

      // [ TODO ]
      // Handle texture transforms.

      return true;
    }
  }; // -- end processInternalTexture

  // [lambda] Wrapper for processInternalTexture using Niflib::TexDesc.
  // When the texture is available the texIndex is set and the function return true.
  auto processTexDesc{
    [&](Niflib::TexDesc const& texDesc, size_t* texIndex) -> bool {
      auto srcTex = texDesc.source;
      if (nullptr == srcTex) {
        return false;
      }
      if (srcTex->IsTextureExternal()) {
        DOJIMA_QUIET_LOG( "[ Warning ] External texture loading is not supported." );
        return false;
      }
      bool const isNew = (mapTextureNameToIndex.count(srcTex->GetTextureFileName()) == 0);
      bool const ok = processInternalTexture(srcTex->GetPixelData(), srcTex->GetTextureFileName(), texIndex);
      if (ok && isNew) {
        // (textures are shared by name, so the first use decides of the wrapping)
        Textures[*texIndex].sampler = getSampler(texDesc.clampMode);
      }
      return ok;
    }
  };

  // -- Meshes

  // Associate a NIF mesh to its glTF mesh / material index.
  // (names are *not* unique within a file, so they cannot be used as keys)
  std::unordered_map<Niflib::NiObject const*, size_t> mapMeshToId;

  // LODs.
  //
  // WotS4 characters embed several copies of each skinned part under a
  // NiSwitchNode, as children named "LV1_*", "LV3_*"... (LV1 being the best),
  // which only differ by the number of bones influencing their vertices
  // (NiSkinningLODController). Exported all together they overlap each others
  // so, unless asked otherwise, only the best level of each switch is kept.
  std::unordered_set<Niflib::NiObject const*> skippedNodes;
  std::unordered_set<Niflib::NiObject const*> skippedMeshes;
  if (!gKeepAllLODs) {
    std::function<void(Niflib::NiAVObject const*)> skipSubtree{
      [&](Niflib::NiAVObject const* n) -> void {
        skippedNodes.insert(n);
        if (Niflib::DynamicCast<Niflib::NiMesh>(n)) {
          skippedMeshes.insert(n);
        }
        if (auto nifNode = Niflib::DynamicCast<Niflib::NiNode>(n); nifNode) {
          for (auto const& child : nifNode->GetChildren()) {
            skipSubtree(child);
          }
        }
      }
    };
    for (auto const& obj : nifList) {
      auto const switchNode = Niflib::DynamicCast<Niflib::NiSwitchNode>(obj);
      if (!switchNode) {
        continue;
      }
      int bestLevel = std::numeric_limits<int>::max();
      for (auto const& child : switchNode->GetChildren()) {
        if (int const level = ParseLODLevel(child->GetName()); level >= 0) {
          bestLevel = std::min(bestLevel, level);
        }
      }
      for (auto const& child : switchNode->GetChildren()) {
        if (int const level = ParseLODLevel(child->GetName()); (level >= 0) && (level != bestLevel)) {
          skipSubtree(child);
        }
      }
    }
    if (!skippedMeshes.empty()) {
      DOJIMA_LOG( "Skipped " << skippedMeshes.size() << " alternate LOD meshes (use --all-lods to keep them)." );
    }
  }

  auto const& meshIndices = getTypeListIndices( kNiMeshKey );
  for (size_t const meshListIndex : meshIndices) {
    auto const nifMesh = Niflib::StaticCast<Niflib::NiMesh>(nifList[meshListIndex]);
    if (skippedMeshes.count(nifMesh) > 0) {
      continue;
    }
    auto const primType = GetPrimitiveType(nifMesh->GetPrimitiveType());

    Meshes.emplace_back();
    Materials.emplace_back();
    size_t const mesh_id = Meshes.size() - 1;
    auto &mesh = Meshes.back();

    // Name.
    mesh.name = nifMesh->GetName();
    mapMeshToId[nifMesh] = mesh_id;

    // ------------------------
    // Material.
    //
    //  WotS NiMesh have the following properties :
    //    * NiMaterialProperty (with non PBR values),
    //    * NiAlphaProperty
    //    * NiSpecularProperty
    //    * NiVertexColorProperty
    //    * NiTexturingProperty, with a varying number of NiSourceTexture, each with a NiPixelData :
    //          + a Base texture (FMT_DXT3), seems useless.
    //          + a Detail Texture (FMT_DXT1), a RGB(A) mask.
    //          + a Normal Texture (FMT_DXT1).
    //          + a Gloss Texture (FMT_DXT1).
    //
    //  ( After each root's "part" node, a NiPixelData is defined containing the
    //   diffuse texture (DXT1) for this part's mesh )
    //
    size_t const meshMaterialIndex = mesh_id; // (one material per mesh)
    auto &material = Materials.back();
    material.name = mesh.name + "_mat_" + std::to_string(meshMaterialIndex); //

    // The NIF material is a Blinn-Phong one, it has no metal and the glTF
    // default (fully metallic) would render everything black.
    material.pbrMetallicRoughness.metallicFactor = 0.0;
    material.pbrMetallicRoughness.roughnessFactor = 1.0;

    //
    // [TODO]
    // Prefer using the specular or SpecularGlossiness GLTF extensions, rather
    // than pbrMetallicRoughness. See :
    //   https://kcoley.github.io/glTF/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/
    //

    // if (auto prop = nifMesh->GetPropertyByType(Niflib::NiSpecularProperty::TYPE); prop) {
    //   DOJIMA_QUIET_LOG( material.name << ": A specular property was detected but not used." );
    // }
    // if (auto prop = nifMesh->GetPropertyByType(Niflib::NiVertexColorProperty::TYPE); prop) {
    //   DOJIMA_QUIET_LOG( material.name << ": A vertex color property was detected but not used." );
    // }

    // Basic / non-PBR parameters.
    if (auto prop = nifMesh->GetPropertyByType(Niflib::NiMaterialProperty::TYPE); prop) {
      auto matProp = Niflib::StaticCast<Niflib::NiMaterialProperty>(prop);

      Niflib::Color3 const& emissive{ matProp->GetEmissiveColor() };
      material.emissiveFactor = { emissive.r, emissive.g, emissive.b };

      // Technically not for metal roughness PBR but well.
      Niflib::Color3 const& diffuse{ matProp->GetDiffuseColor() };
      material.pbrMetallicRoughness.baseColorFactor = { diffuse.r, diffuse.g, diffuse.b, matProp->GetTransparency() };

      // Keep the Blinn-Phong values for users who want to rebuild a specular shading.
      Niflib::Color3 const& specular{ matProp->GetSpecularColor() };
      material.extras = tinygltf::Value(tinygltf::Value::Object{
        { "specularColor", tinygltf::Value(tinygltf::Value::Array{
            tinygltf::Value(specular.r), tinygltf::Value(specular.g), tinygltf::Value(specular.b) }) },
        { "glossiness", tinygltf::Value(static_cast<double>(matProp->GetGlossiness())) },
      });
    }

    // Alpha Testing & Blending.
    if (auto prop = nifMesh->GetPropertyByType(Niflib::NiAlphaProperty::TYPE); prop) {
      auto alphaProp = Niflib::StaticCast<Niflib::NiAlphaProperty>(prop);

      SetAlphaMode(*alphaProp, &material);
    }

    // Check for double sided geometry.
    if (auto prop = nifMesh->GetPropertyByType(Niflib::NiStencilProperty::TYPE); prop) {
      auto stencilProp = Niflib::StaticCast<Niflib::NiStencilProperty>(prop);
      material.doubleSided = (Niflib::DRAW_BOTH == stencilProp->GetFaceDrawMode());
    }

    // Image / Texture.
    if (auto prop = nifMesh->GetPropertyByType(Niflib::NiTexturingProperty::TYPE); prop) {
      auto texProp = Niflib::StaticCast<Niflib::NiTexturingProperty>(prop);
      size_t texIndex;

      // (I don't know where could be the diffuse texture in WotS4 files !)
      // (intenal texture are fjust empty black transparent stuff, textures are elsewhere)
      if (auto texDesc = texProp->GetBaseTexture(); processTexDesc(texDesc, &texIndex)) {
        // DOJIMA_QUIET_LOG( ">>> " << texDesc.source->GetTextureFileName());
        material.pbrMetallicRoughness.baseColorTexture.index = texIndex;
      }

      // Normal Texture.
      if (processTexDesc(texProp->GetNormalTexture(), &texIndex)) {
        material.normalTexture.index = texIndex;
      }

      // "Specular" Texture.
      // processTexDesc(texProp->GetGlossTexture(), &texIndex);

      // Some RGBA mask.
      if (auto texDesc = texProp->GetDetailTexture(); processTexDesc(texDesc, &texIndex)) {
        // material.pbrMetallicRoughness.baseColorTexture.index = texIndex; //
      }
    }
    // ------------------------

    // Primitives / submeshes.
    uint32_t const nsubmeshes = nifMesh->GetNumSubmeshes();
    mesh.primitives.resize(nsubmeshes);
    for (auto &prim : mesh.primitives) {
      prim.mode = primType;
      prim.material = meshMaterialIndex; // (same as current niMesh)
    }

    // BONE_PALETTE : for each submesh, the list of skeleton bones (as indices to
    // the skinning modifier's bones) its vertices' joints refer to.
    Niflib::MeshData const* paletteData = nullptr;
    for (auto const& md : nifMesh->GetMeshDatas()) {
      if ((md.stream->GetUsage() == Niflib::USAGE_USER) && !md.componentSemantics.empty() &&
          (std::string(md.componentSemantics[0].name) == "BONE_PALETTE")) {
        paletteData = &md;
        break;
      }
    }

    // Returns the skeleton bone index of a submesh's palette entry.
    auto remapJoint = [&](uint32_t submesh_id, uint16_t paletteIndex) -> uint16_t {
      if (!paletteData) {
        return paletteIndex;
      }
      auto const& region = paletteData->stream->GetRegions()[paletteData->submeshToRegionMap[submesh_id]];
      if (paletteIndex >= region.numIndices) {
        DOJIMA_QUIET_LOG( "[Warning] " << mesh.name << " : joint index out of its bone palette." );
        return 0;
      }
      uint16_t bone;
      std::memcpy(&bone, paletteData->stream->GetData().data() + (region.startIndex + paletteIndex) * sizeof(uint16_t), sizeof(uint16_t));
      return bone;
    };

    for (auto const& md : nifMesh->GetMeshDatas()) {
      auto const& stream = md.stream;
      auto const usage = stream->GetUsage();

      // Bypass unused meshData (USAGE_USER holds the BONE_PALETTE).
      if ((usage != Niflib::USAGE_VERTEX_INDEX) && (usage != Niflib::USAGE_VERTEX)) {
        continue;
      }

      auto const& data = stream->GetData();
      auto const& formats = stream->GetComponentFormats();
      auto const& regions = stream->GetRegions();

      // -- Indices : the stream is copied as is, each submesh has an accessor
      //    to its region.
      if (usage == Niflib::USAGE_VERTEX_INDEX) {
        int const bufferView = addBufferView(BufferId::INDEX, data.data(), data.size(), TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER);

        size_t const byteSize = FormatByteSize(formats[0]);
        int const componentType = (byteSize == sizeof(uint32_t)) ? TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT
                                                                  : TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
        for (uint32_t submesh_id = 0; submesh_id < nsubmeshes; ++submesh_id) {
          auto const& region = regions[md.submeshToRegionMap[submesh_id]];

          tinygltf::Accessor acc;
          acc.name = kNameAccessors ? "INDEX" : "";
          acc.bufferView = bufferView;
          acc.byteOffset = region.startIndex * byteSize;
          acc.componentType = componentType;
          acc.type = TINYGLTF_TYPE_SCALAR;
          acc.count = region.numIndices;

          Accessors.push_back(acc);
          mesh.primitives[submesh_id].indices = static_cast<int>(Accessors.size()) - 1;
        }
        continue;
      }

      // -- Vertices : attributes are decoded to float32 (NIF streams can be
      //    interleaved, half-precision, or have a different layout than the one
      //    glTF mandates) and repacked as one tight accessor per attribute and
      //    per submesh.
      size_t vertexStride = 0;
      for (auto const format : formats) {
        vertexStride += FormatByteSize(format);
      }

      for (uint32_t submesh_id = 0; submesh_id < nsubmeshes; ++submesh_id) {
        auto &prim = mesh.primitives[submesh_id];
        auto const& region = regions[md.submeshToRegionMap[submesh_id]];

        size_t const count = region.numIndices;
        size_t const regionOffset = static_cast<size_t>(region.startIndex) * vertexStride;
        if (regionOffset + count * vertexStride > data.size()) {
          DOJIMA_QUIET_LOG( "[Warning] " << mesh.name << " : vertex region exceeds its data stream." );
          continue;
        }

        // Decode each component of the stream.
        struct Attribute_t {
          std::vector<float> values; // always 4 floats per vertex.
          int numComps = 0;
        };
        std::unordered_map<std::string, Attribute_t> attributes;

        size_t componentOffset = 0;
        for (int comp_id = 0; comp_id < md.numComponents; ++comp_id) {
          auto const format = formats[comp_id];
          size_t const byteSize = FormatByteSize(format);

          std::string key{ md.componentSemantics[comp_id].name };
          if (size_t const pos = key.rfind("_BP"); pos != std::string::npos) {
            key.erase(pos); // (bind pose variants)
          }
          key += std::to_string(md.componentSemantics[comp_id].index);

          Attribute_t attr;
          attr.numComps = FormatNumComponents(format);
          attr.values.assign(count * 4u, 0.0f);

          bool ok = true;
          for (size_t i = 0; ok && (i < count); ++i) {
            ok = DecodeFormat(format, data.data() + regionOffset + i * vertexStride + componentOffset, &attr.values[i * 4u]);
          }
          if (ok) {
            attributes[key] = std::move(attr);
          } else {
            DOJIMA_QUIET_LOG( "[Warning] " << key << " : unsupported format " << format << "." );
          }
          componentOffset += byteSize;
        }

        // Returns the attribute repacked with `n` components, or nullptr.
        auto getAttribute = [&](std::string const& key, int n, std::vector<float> *out) -> bool {
          auto it = attributes.find(key);
          if (it == attributes.end()) {
            return false;
          }
          out->resize(count * n);
          for (size_t i = 0; i < count; ++i) {
            for (int c = 0; c < n; ++c) {
              (*out)[i * n + c] = it->second.values[i * 4u + c];
            }
          }
          return true;
        };

        std::vector<float> v;

        if (getAttribute("POSITION0", 3, &v)) {
          prim.attributes["POSITION"] = addFloatAccessor("POSITION", v, TINYGLTF_TYPE_VEC3, 3, true);
        }
        if (getAttribute("NORMAL0", 3, &v)) {
          prim.attributes["NORMAL"] = addFloatAccessor("NORMAL", v, TINYGLTF_TYPE_VEC3, 3, false);
        }
        for (int set = 0; set < 8; ++set) {
          if (getAttribute("TEXCOORD" + std::to_string(set), 2, &v)) {
            auto const name = "TEXCOORD_" + std::to_string(set);
            prim.attributes[name] = addFloatAccessor(name, v, TINYGLTF_TYPE_VEC2, 2, false);
          }
        }

        // glTF expects a VEC4 tangent (xyz + handedness) and no binormal :
        // the binormal is only used to recover the handedness.
        if (std::vector<float> n, t, b; getAttribute("NORMAL0", 3, &n) && getAttribute("TANGENT0", 3, &t)) {
          bool const hasBinormal = getAttribute("BINORMAL0", 3, &b);
          v.assign(count * 4u, 1.0f);
          for (size_t i = 0; i < count; ++i) {
            float const* N = &n[i * 3u];
            float const* T = &t[i * 3u];
            v[i * 4u + 0] = T[0];
            v[i * 4u + 1] = T[1];
            v[i * 4u + 2] = T[2];
            if (hasBinormal) {
              float const* B = &b[i * 3u];
              float const cx = N[1] * T[2] - N[2] * T[1];
              float const cy = N[2] * T[0] - N[0] * T[2];
              float const cz = N[0] * T[1] - N[1] * T[0];
              v[i * 4u + 3] = (cx * B[0] + cy * B[1] + cz * B[2] < 0.0f) ? -1.0f : 1.0f;
            }
          }
          prim.attributes["TANGENT"] = addFloatAccessor("TANGENT", v, TINYGLTF_TYPE_VEC4, 4, false);
        }

        // Skinning attributes.
        //
        // NIF stores 3 weights per vertex (the 4th is implicit) and the joint
        // indices refer to the submesh BONE_PALETTE, which we resolve here to
        // indices in the skin's joints (the skinning modifier's bones).
        if (auto itJ = attributes.find("BLENDINDICES0"), itW = attributes.find("BLENDWEIGHT0");
            (itJ != attributes.end()) && (itW != attributes.end())) {
          std::vector<uint16_t> joints(count * 4u);
          std::vector<float> weights(count * 4u);

          for (size_t i = 0; i < count; ++i) {
            float const* j = &itJ->second.values[i * 4u];
            float const* w = &itW->second.values[i * 4u];

            float sum = 0.0f;
            float ww[4];
            for (int c = 0; c < 4; ++c) {
              ww[c] = (c < itW->second.numComps) ? std::max(w[c], 0.0f) : 0.0f;
              sum += ww[c];
            }
            if (itW->second.numComps == 3) {
              ww[3] = std::max(1.0f - sum, 0.0f);
              sum += ww[3];
            }
            // Snap residual weights (e.g. from the implicit 4th one) to zero, as
            // glTF requires unused joints to have a null weight and a null index.
            constexpr float kMinWeight = 1e-5f;
            sum = 0.0f;
            for (int c = 0; c < 4; ++c) {
              ww[c] = (ww[c] < kMinWeight) ? 0.0f : ww[c];
              sum += ww[c];
            }
            for (int c = 0; c < 4; ++c) {
              ww[c] = (sum > 0.0f) ? ww[c] / sum : ((c == 0) ? 1.0f : 0.0f);
              weights[i * 4u + c] = ww[c];
              joints[i * 4u + c] = (ww[c] > 0.0f) ? remapJoint(submesh_id, static_cast<uint16_t>(j[c])) : uint16_t(0);
            }
          }

          tinygltf::Accessor accJ;
          accJ.name = kNameAccessors ? "JOINTS_0" : "";
          accJ.bufferView = addBufferView(BufferId::VERTEX, joints.data(), joints.size() * sizeof(uint16_t), TINYGLTF_TARGET_ARRAY_BUFFER);
          accJ.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
          accJ.type = TINYGLTF_TYPE_VEC4;
          accJ.count = count;
          Accessors.push_back(accJ);
          prim.attributes["JOINTS_0"] = static_cast<int>(Accessors.size()) - 1;

          prim.attributes["WEIGHTS_0"] = addFloatAccessor("WEIGHTS_0", weights, TINYGLTF_TYPE_VEC4, 4, false);
        }
      } // -- end submeshes --
    } // -- end meshdatas --

    // TODO : NiSkinningMeshModifier.
    //mesh.weights;

    // TODO : extra datas (NiFloatExtraData, NiColorExtraData).
    //mesh.extra;
  } // -- end meshes --

  // --------------

  // -- Node hierarchy.

  std::vector<tinygltf::Node> Nodes{};
  tinygltf::Node *root_node_ptr = nullptr;

  // Detects if the NIF file is a pack.
  uint32_t numSubParts = 0u;
  if (auto firstNode = Niflib::DynamicCast<Niflib::NiIntegerExtraData>(nifList[0]); firstNode) {
    numSubParts = firstNode->GetData() / 2u;
    DOJIMA_QUIET_LOG( "Detected Nif subparts : " << numSubParts );
  }

  // (only use a common root on NifPack)
  bool kUseRootNode = (numSubParts > 0u);

  // Determine the total number of hierarchical nodes.
  size_t totalNodeCount = kUseRootNode ? 1 : 0;
  for (auto const& n : nifList) {
    auto ptr = Niflib::DynamicCast<Niflib::NiAVObject>(n);
    totalNodeCount += (ptr != nullptr) ? 1 : 0;
    // (lights are held by an additional child node)
    totalNodeCount += (Niflib::DynamicCast<Niflib::NiPointLight>(n) || Niflib::DynamicCast<Niflib::NiDirectionalLight>(n)) ? 1 : 0;
  }

  Nodes.reserve(totalNodeCount);

  // NIF object to glTF node index, and the nodes holding a mesh (skins are
  // created once the whole hierarchy is known, as bones might be visited after
  // the meshes they influence).
  std::vector<tinygltf::Light> Lights;
  std::unordered_map<Niflib::NiAVObject const*, int> mapNifToNode;
  std::vector<std::pair<int, Niflib::Ref<Niflib::NiMesh>>> meshNodes;

  if (kUseRootNode) {
    Nodes.push_back({});
    root_node_ptr = &Nodes[0];
    root_node_ptr->name = "root";
  }


  // -- Shared skeleton.
  //
  // In a pack each part holds its own copy of the skeleton, and often only a
  // partial one (the head part has no spine for instance) although the skinning
  // data (bind poses) always refer to the complete rig. Rig nodes (the bones
  // and their ancestors) are therefore merged by name, so that all the skins of
  // a character share a single skeleton, which is also what the animations
  // are targeting.
  std::unordered_set<Niflib::NiAVObject const*> rigNodes;
  for (auto const& obj : nifList) {
    auto const nifMesh = Niflib::DynamicCast<Niflib::NiMesh>(obj);
    if (!nifMesh || (skippedMeshes.count(nifMesh) > 0)) {
      continue;
    }
    for (auto const& modifier : nifMesh->GetModifiers()) {
      auto const skinModifier = Niflib::DynamicCast<Niflib::NiSkinningMeshModifier>(modifier);
      if (!skinModifier) {
        continue;
      }
      for (Niflib::NiAVObject const* bone : skinModifier->GetBones()) {
        for (Niflib::NiAVObject const* n = bone; n; n = n->GetParent()) {
          rigNodes.insert(n);
        }
      }
    }
  }
  std::unordered_map<std::string, int> mapRigNameToNode;

  // The parts' rigs are partial : a bone missing its real parent is often
  // attached to a farther ancestor ("main" for instance). When copies of a bone
  // disagree we therefore keep the deepest parent, which is the real one.
  std::unordered_map<std::string, std::pair<std::string, int>> rigBestParent;
  for (auto const& obj : nifList) {
    auto const nifNode = Niflib::DynamicCast<Niflib::NiAVObject>(obj);
    if (!nifNode || (rigNodes.count(nifNode) == 0)) {
      continue;
    }
    Niflib::NiAVObject* parent = nifNode->GetParent();
    int depth = 0;
    for (Niflib::NiAVObject const* n = parent; n; n = n->GetParent()) {
      ++depth;
    }
    auto &best = rigBestParent[nifNode->GetName()];
    if (parent && (depth > best.second)) {
      best = { parent->GetName(), depth };
    }
  }
  std::unordered_map<int, int> rigFallbackParent; // (node index to the index of its traversal parent)

  // [lambda] recursive function to fill the node hierarchy.
  std::function<void(tinygltf::Node *const, Niflib::NiAVObject *const, size_t *)> fillNodes{
    [&](tinygltf::Node *const parent, Niflib::NiAVObject *const nifAVO, size_t *mainTexId) -> void {
      // Alternate LODs are not exported.
      if (skippedNodes.count(nifAVO) > 0) {
        return;
      }

      // Rig nodes already created by a previous part are reused.
      bool const isRigNode = (rigNodes.count(nifAVO) > 0);
      int nodeIndex = -1;
      if (isRigNode) {
        if (auto it = mapRigNameToNode.find(nifAVO->GetName()); it != mapRigNameToNode.end()) {
          nodeIndex = it->second;
        }
      }

      if (nodeIndex < 0) {
        // Get a reference to a new node.
        Nodes.push_back( tinygltf::Node() );
        nodeIndex = static_cast<int>(Nodes.size()) - 1;
        auto &node = Nodes.back();

        if (isRigNode) {
          // (rig nodes are linked to their parent once all parts are processed)
          mapRigNameToNode[nifAVO->GetName()] = nodeIndex;
          rigFallbackParent[nodeIndex] = parent ? static_cast<int>(parent - Nodes.data()) : -1;
        } else if (parent) {
          // Set it as its parent's children.
          parent->children.push_back(nodeIndex);
        }

        node.name = nifAVO->GetName();

        auto const& qRotation = nifAVO->GetLocalRotation().AsQuaternion();
        node.rotation = { (double)qRotation.x, (double)qRotation.y, (double)qRotation.z, (double)qRotation.w };

        auto const fScale = static_cast<double>(nifAVO->GetLocalScale());
        node.scale = { fScale, fScale, fScale };

        auto const &vTranslation = nifAVO->GetLocalTranslation();
        node.translation = { (double)vTranslation.x, (double)vTranslation.y, (double)vTranslation.z };

        if (auto nifMesh = Niflib::DynamicCast<Niflib::NiMesh>(nifAVO); nifMesh) {
          size_t const meshId = mapMeshToId.at(nifMesh);
          node.mesh = static_cast<int>(meshId);
          meshNodes.emplace_back(nodeIndex, nifMesh);

          if (mainTexId) {
            auto &mat = Materials.at(meshId);
            mat.pbrMetallicRoughness.baseColorTexture.index = *mainTexId; //
          }
        }

        // Lights (glTF has no ambient light, which is dropped).
        // NIF directional lights shine along their local +X, glTF's along -Z :
        // the light is held by a child node applying the fixed rotation.
        if (auto nifLight = Niflib::DynamicCast<Niflib::NiLight>(nifAVO); nifLight) {
          bool const isDirectional = (nullptr != Niflib::DynamicCast<Niflib::NiDirectionalLight>(nifAVO));
          bool const isPoint = (nullptr != Niflib::DynamicCast<Niflib::NiPointLight>(nifAVO));
          if (isDirectional || isPoint) {
            Niflib::Color3 const c = nifLight->GetDiffuseColor();
            tinygltf::Light light;
            light.name = node.name;
            light.type = isDirectional ? "directional" : "point";
            light.color = { c.r, c.g, c.b };
            light.intensity = nifLight->GetDimmer();
            Lights.push_back(light);

            tinygltf::Node lightNode;
            lightNode.name = node.name + "_light";
            if (isDirectional) {
              lightNode.rotation = { 0.0, -0.70710678118654752, 0.0, 0.70710678118654752 };
            }
            lightNode.extensions["KHR_lights_punctual"] = tinygltf::Value(tinygltf::Value::Object{
              { "light", tinygltf::Value(static_cast<int>(Lights.size()) - 1) } });
            Nodes.push_back(lightNode);
            Nodes[nodeIndex].children.push_back(static_cast<int>(Nodes.size()) - 1);
          }
        }

        // [ TODO ] Detect meshLOD node (nifSwitchNode).
        // if (auto nifSwitchNode = Niflib::DynamicCast<Niflib::NiSwitchNode>(nifAVO); nifSwitchNode) {}

        // [ TODO ] Handle extra datas.
        // auto const& listXD = nifAVO->GetExtraData();
        // node.extras;
      }
      mapNifToNode[nifAVO] = nodeIndex;

      // Continue processing if the nif object is a pure node (and not only a leaf).
      if (auto nifNode = Niflib::DynamicCast<Niflib::NiNode>(nifAVO); nifNode) {
        for (auto &child : nifNode->GetChildren()) {
          fillNodes(&Nodes[nodeIndex], child, mainTexId);
        }
      }
    }
  };

  // (on NIF's "Characters Pack")
  //
  //  The first node is a "NiIntegerExtraData" and contains the number of
  //   "NiStringExtraData" on the upper root.
  //
  // Then for each NiStringExtraData, if the following object has a consecutive index
  // it contains the path for that part's data and is alternatively a NiNode or a NiPixelData.
  //
  // The PixelData representing the DDS DXT1/5 diffuse texture for the mesh.
  //
  if (bool const bNIFPack = (numSubParts > 0u); bNIFPack) {
    // -- NIF Pack.
    DOJIMA_QUIET_LOG(">> nif pack");

    struct PartIndices_t {
      int path_id = -1; // index to a NiStringExtraData containing a path.
      int data_id = -1; // index to either a NiNode or a NiPixelData.
    };

    std::vector<PartIndices_t> nodeParts;
    nodeParts.reserve(numSubParts);

    std::vector<PartIndices_t> pixelParts;
    pixelParts.reserve(numSubParts);

    // Retrieve each part indices.
    for (int i = 0; i < (int)nifList.size(); ++i) {
      auto n0 = nifList[i];
      if (auto nifString = Niflib::DynamicCast<Niflib::NiStringExtraData>(n0); nifString) {
        auto n1 = nifList[i+1];
        if (auto nifNode = Niflib::DynamicCast<Niflib::NiNode>(n1); nifNode && (nifNode->GetName() == "SceneNode")) {
          nodeParts.push_back( {i, ++i} );
          continue;
        }
        if (auto nifPixelData = Niflib::DynamicCast<Niflib::NiPixelData>(n1); nifPixelData) {
          pixelParts.push_back( {i, ++i} );
          continue;
        }
      }
    }

    // Parts are processed from the one with the most complete rig, so shared
    // rig nodes get their parent from the full skeleton and the other parts
    // only add what they define on top of it (e.g. facial bones).
    std::function<size_t(Niflib::NiAVObject const*)> countRigNodes{
      [&](Niflib::NiAVObject const* n) -> size_t {
        size_t count = rigNodes.count(n);
        if (auto nifNode = Niflib::DynamicCast<Niflib::NiNode>(n); nifNode) {
          for (auto &child : nifNode->GetChildren()) {
            count += countRigNodes(child);
          }
        }
        return count;
      }
    };
    std::vector<size_t> partOrder(nodeParts.size());
    std::vector<size_t> partRigSize(nodeParts.size());
    for (size_t i = 0; i < nodeParts.size(); ++i) {
      partOrder[i] = i;
      partRigSize[i] = countRigNodes(Niflib::DynamicCast<Niflib::NiAVObject>(nifList[nodeParts[i].data_id]));
    }
    std::stable_sort(partOrder.begin(), partOrder.end(), [&](size_t a, size_t b) {
      return partRigSize[a] > partRigSize[b];
    });

    // Append the node part to the nodes hierarchy.
    for (size_t const i : partOrder) {
      if (i >= pixelParts.size()) {
        DOJIMA_LOG( "[Warning] part " << i << " has no matching texture." );
      }
      // Geometry.
      auto const nodeId = nodeParts[i].data_id;
      auto upperNode = Niflib::StaticCast<Niflib::NiNode>(nifList[nodeId]);

      // ---------------
      /* XXX Questionnable hack to bypass 2 'useless' nodes XXX */
      if constexpr (false)
      {
        if (auto child1 = Niflib::DynamicCast<Niflib::NiNode>(upperNode->GetChildren().at(0)); child1) {
          // (This bypass its sibbling node, probably useful for rigging)
          upperNode = Niflib::DynamicCast<Niflib::NiNode>(child1->GetChildren().at(1));
        }
      }
      // ---------------

      // ---------------
      // Load side albedo texture on Characters (can fails elsewhere).
      size_t texIndex = 0;
      bool hasTexture = false;
      if (i < pixelParts.size()) {
        auto const pixelPart = pixelParts[i];
        auto texPathNode = Niflib::StaticCast<Niflib::NiStringExtraData>(nifList.at(pixelPart.path_id));
        auto texPixelNode = Niflib::StaticCast<Niflib::NiPixelData>(nifList.at(pixelPart.data_id));
        hasTexture = processInternalTexture(texPixelNode, texPathNode->GetData(), &texIndex);
      }

      if (hasTexture) {
        fillNodes(root_node_ptr, upperNode, &texIndex);

        // [ TODO ]
        // Associate each meshes material diffuse texture with the one defined by
        // their following upper texPixelNode (check with ID).
        // auto &pbr = material.pbrMetallicRoughness;
        // pbr.baseColorTextureindex = -1;  // required.
        // pbr.baseColorTexture.texCoord; // (texcoord index)
      } else {
        fillNodes(root_node_ptr, upperNode, nullptr);
      }

    }
  } else if (auto firstNode = Niflib::DynamicCast<Niflib::NiNode>(nifList[0]); firstNode) {
    // -- NIF Part.

    // [wip] Retrieve side source textures, not found within mesh nodes.
    // auto const& sourceTexList = getTypeListIndices( "NiSourceTexture" );
    // for (size_t i = 0; i < sourceTexList.size(); ++i) {
    //   size_t texIndex;
    //   auto srcTexNode = Niflib::StaticCast<Niflib::NiSourceTexture>(nifList.at(sourceTexList[i]));
    //   processInternalTexture(srcTexNode->GetPixelData(), srcTexNode->GetTextureFileName(), &texIndex);
    // }

    fillNodes(nullptr, firstNode, nullptr);
  } else if (auto firstNode = Niflib::DynamicCast<Niflib::NiPixelData>(nifList[0]); firstNode) {
    // -- NIF PixelData.
    // WotS4 PixelData-only NIF files are encoded as BIG_ENDIAN and supposedly from a
    // version unsupported by Niflib so we can't read them as is.
    // Consequently we never reach this branch as Niflib::ReadHeader will crash before.
  }

  // --------------

  // std::cerr << "Nodes usage : " << Nodes.size() << " / " << Nodes.capacity()
  //           << std::endl;

  // -- Link the shared rig nodes to their (deepest) parent.
  {
    std::vector<std::pair<int, std::string>> rigNodeList;
    for (auto const& [name, index] : mapRigNameToNode) {
      rigNodeList.emplace_back(index, name);
    }
    std::sort(rigNodeList.begin(), rigNodeList.end()); // (deterministic order)

    for (auto const& [index, name] : rigNodeList) {
      int parentIndex = rigFallbackParent[index];
      if (auto it = rigBestParent.find(name); (it != rigBestParent.end()) && !it->second.first.empty()) {
        if (auto pit = mapRigNameToNode.find(it->second.first); (pit != mapRigNameToNode.end()) && (pit->second != index)) {
          parentIndex = pit->second;
        }
      }
      if (parentIndex >= 0) {
        Nodes[parentIndex].children.push_back(index);
      }
    }
  }

  // -- Skins.

  std::vector<tinygltf::Skin> Skins;
  std::unordered_map<int, Mat4> bindWorld; // (joint node index to its bind pose world transform)
  for (auto const& [nodeIndex, nifMesh] : meshNodes) {
    for (auto const& modifier : nifMesh->GetModifiers()) {
      auto const skinModifier = Niflib::DynamicCast<Niflib::NiSkinningMeshModifier>(modifier);
      if (!skinModifier) {
        continue;
      }

      tinygltf::Skin skin;
      skin.name = nifMesh->GetName() + "_skin";

      if (auto it = mapNifToNode.find(skinModifier->GetSkeletonRoot()); it != mapNifToNode.end()) {
        skin.skeleton = it->second;
      }

      // The NIF bone transforms go from the skin (bind pose) space to each bone
      // space, which are glTF's inverse bind matrices. (column-major 4x4)
      // Note : NIF rotation matrices are stored transposed relatively to glTF.
      auto const& bones = skinModifier->GetBones();
      auto const& boneTransforms = skinModifier->GetBoneTransforms();
      std::vector<float> inverseBindMatrices;
      inverseBindMatrices.reserve(bones.size() * 16u);

      bool valid = true;
      for (size_t b = 0; b < bones.size(); ++b) {
        auto it = mapNifToNode.find(bones[b]);
        if (it == mapNifToNode.end()) {
          DOJIMA_QUIET_LOG( "[Warning] " << nifMesh->GetName() << " : a skin bone is not part of the hierarchy." );
          valid = false;
          break;
        }
        skin.joints.push_back(it->second);

        Niflib::SkinTransform const& tr = boneTransforms[b];
        Niflib::Matrix33 rot = tr.rotation;
        for (int col = 0; col < 3; ++col) {
          for (int row = 0; row < 3; ++row) {
            inverseBindMatrices.push_back(rot[col][row] * tr.scale);
          }
          inverseBindMatrices.push_back(0.0f);
        }
        inverseBindMatrices.insert(inverseBindMatrices.end(), { tr.translation.x, tr.translation.y, tr.translation.z, 1.0f });

        if (bindWorld.count(it->second) == 0) {
          Mat4 ibm;
          std::copy(inverseBindMatrices.end() - 16, inverseBindMatrices.end(), ibm.begin());
          bindWorld[it->second] = AffineInverse(ibm);
        }
      }
      if (!valid || bones.empty()) {
        break;
      }

      tinygltf::Accessor acc;
      acc.name = kNameAccessors ? "INVERSE_BIND_MATRICES" : "";
      acc.bufferView = addBufferView(BufferId::VERTEX, inverseBindMatrices.data(), inverseBindMatrices.size() * sizeof(float), 0);
      acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
      acc.type = TINYGLTF_TYPE_MAT4;
      acc.count = bones.size();
      Accessors.push_back(acc);
      skin.inverseBindMatrices = static_cast<int>(Accessors.size()) - 1;

      Skins.push_back(skin);
      Nodes[nodeIndex].skin = static_cast<int>(Skins.size()) - 1;
      break; // (a mesh has a single skin)
    }
  }

  // -- Rest pose.
  //
  // The nodes of the partial rigs do not hold the real bind pose, so the rest
  // pose of the bones is rebuilt from the skinning data : the skeleton then
  // matches the inverse bind matrices, and the meshes appear unposed.
  if (!Nodes.empty()) {
    std::function<void(int, Mat4 const&)> bakeRestPose{
      [&](int index, Mat4 const& parentWorld) -> void {
        auto &node = Nodes[index];
        Mat4 world;
        if (auto it = bindWorld.find(index); it != bindWorld.end()) {
          world = it->second;
          SetTRS(Mul(AffineInverse(parentWorld), world), &node);
        } else {
          world = Mul(parentWorld, FromTRS(node));
        }
        for (int child : node.children) {
          bakeRestPose(child, world);
        }
      }
    };
    bakeRestPose(0, Identity());
  }

  // -- Animations.
  //
  // Each sequence of the keyframe files is exported as a glTF animation. Its
  // evaluators (one per animated node, as B-splines or keyframes) are resampled
  // at a fixed rate to translation / rotation / scale channels. The nodes are
  // targeted by name, on the shared rig when it has the node, as parts hold
  // their own copies of the skeleton.
  std::vector<tinygltf::Animation> Animations;
  if (!gAnimFilenames.empty() && !Nodes.empty()) {
    std::unordered_map<std::string, std::vector<int>> mapNameToNodes;
    for (size_t i = 0; i < Nodes.size(); ++i) {
      mapNameToNodes[Nodes[i].name].push_back(static_cast<int>(i));
    }
    auto findTargets = [&](std::string const& name) -> std::vector<int> {
      if (auto it = mapRigNameToNode.find(name); it != mapRigNameToNode.end()) {
        return { it->second };
      }
      if (auto it = mapNameToNodes.find(name); it != mapNameToNodes.end()) {
        return it->second;
      }
      return {};
    };

    // [lambda] Create an animation accessor (not bound to a buffer view target).
    auto addAnimAccessor{
      [&](std::string const& name, std::vector<float> const& values, int type, size_t numComps, bool bounds) -> int {
        size_t const count = values.size() / numComps;

        tinygltf::Accessor acc;
        acc.name = kNameAccessors ? name : "";
        acc.bufferView = addBufferView(BufferId::VERTEX, values.data(), values.size() * sizeof(float), 0);
        acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        acc.type = type;
        acc.count = count;
        if (bounds && (count > 0u)) {
          acc.minValues.assign(numComps,  std::numeric_limits<double>::max());
          acc.maxValues.assign(numComps, -std::numeric_limits<double>::max());
          for (size_t i = 0; i < count; ++i) {
            for (size_t c = 0; c < numComps; ++c) {
              double const v = values[i * numComps + c];
              acc.minValues[c] = std::min(acc.minValues[c], v);
              acc.maxValues[c] = std::max(acc.maxValues[c], v);
            }
          }
        }
        Accessors.push_back(acc);
        return static_cast<int>(Accessors.size()) - 1;
      }
    };

    for (auto const& kfFilename : gAnimFilenames) {
      kf::File kfFile;
      if (std::string error; !kf::Load(kfFilename, &kfFile, &error)) {
        DOJIMA_LOG( "[Error] " << kfFilename << " : " << error );
        continue;
      }

      size_t numChannels = 0;
      size_t numPlayed = 0;
      std::unordered_set<std::string> missingTargets;
      std::unordered_set<std::string> unsupportedEvaluators;

      for (auto const& seq : kfFile.sequences) {
        tinygltf::Animation anim;
        anim.name = seq.name.empty() ? FileStem(kfFilename) : seq.name;

        std::vector<float> const times = MakeSampleTimes(seq.duration, gAnimFps);
        int timeAccessor = -1; // (created along the first channel)

        // Each (node, path) can be targeted only once by an animation.
        std::unordered_set<std::string> targeted;

        // [lambda] Add a sampler (once) and the channels targeting the nodes with it.
        auto addChannels = [&](std::vector<int> const& nodes, char const* path, std::vector<float> const& values,
                               int type, size_t numComps, auto const& isRestValue) -> void {
          int sampler = -1;
          for (int const nodeIndex : nodes) {
            if (isRestValue(Nodes[nodeIndex])) {
              continue;
            }
            if (!targeted.insert(std::to_string(nodeIndex) + path).second) {
              DOJIMA_QUIET_LOG( "[Warning] " << anim.name << " : '" << Nodes[nodeIndex].name << "' has several " << path << " channels." );
              continue;
            }
            if (sampler < 0) {
              if (timeAccessor < 0) {
                timeAccessor = addAnimAccessor("TIME", times, TINYGLTF_TYPE_SCALAR, 1, true);
              }
              tinygltf::AnimationSampler s;
              s.input = timeAccessor;
              s.output = addAnimAccessor(path, values, type, numComps, false);
              s.interpolation = "LINEAR";
              anim.samplers.push_back(s);
              sampler = static_cast<int>(anim.samplers.size()) - 1;
            }
            tinygltf::AnimationChannel channel;
            channel.sampler = sampler;
            channel.target_node = nodeIndex;
            channel.target_path = path;
            anim.channels.push_back(channel);
          }
        };

        // [lambda] Sample a channel at every time, false if it is not animated.
        auto sampleChannel = [&](kf::Channel const& c, std::vector<float> *values) -> bool {
          size_t const nc = static_cast<size_t>(c.numComps);
          values->assign(times.size() * nc, 0.0f);
          for (size_t i = 0; i < times.size(); ++i) {
            if (!kf::Sample(c, times[i], &(*values)[i * nc])) {
              return false;
            }
          }
          return true;
        };

        constexpr float kEpsilon = 1e-5f;

        for (auto const& evaluator : seq.evaluators) {
          auto const targets = findTargets(evaluator.target);
          if (targets.empty()) {
            missingTargets.insert(evaluator.target);
            continue;
          }
          bool const isAccumRoot = (evaluator.target == seq.accumRoot);
          std::vector<float> values;

          // Translation.
          if (sampleChannel(evaluator.translation, &values)) {
            if (isAccumRoot && !gKeepRootMotion) {
              for (size_t axis = 0; axis < 3u; ++axis) {
                if (seq.accumFlags & (static_cast<uint32_t>(kf::kAccumXTrans) << axis)) {
                  for (size_t i = 1; i < times.size(); ++i) {
                    values[i * 3u + axis] = values[axis];
                  }
                }
              }
            }
            addChannels(targets, "translation", values, TINYGLTF_TYPE_VEC3, 3u, [&](tinygltf::Node const& node) {
              for (size_t i = 0; i < values.size(); ++i) {
                double const rest = (node.translation.size() == 3u) ? node.translation[i % 3u] : 0.0;
                if (std::abs(static_cast<double>(values[i]) - rest) > static_cast<double>(kEpsilon)) {
                  return false;
                }
              }
              return true;
            });
          }

          // Rotation (kf quaternions are w, x, y, z, glTF's x, y, z, w).
          if (sampleChannel(evaluator.rotation, &values)) {
            // Keep the quaternions in the same hemisphere, so the interpolation
            // always takes the shortest path.
            for (size_t i = 1; i < times.size(); ++i) {
              float const* p = &values[(i - 1u) * 4u];
              float *q = &values[i * 4u];
              if (p[0]*q[0] + p[1]*q[1] + p[2]*q[2] + p[3]*q[3] < 0.0f) {
                for (int c = 0; c < 4; ++c) q[c] = -q[c];
              }
            }
            for (size_t i = 0; i < times.size(); ++i) {
              float *q = &values[i * 4u];
              std::rotate(q, q + 1, q + 4);
            }
            addChannels(targets, "rotation", values, TINYGLTF_TYPE_VEC4, 4u, [&](tinygltf::Node const& node) {
              if (node.rotation.size() != 4u) {
                return false;
              }
              for (size_t i = 0; i < times.size(); ++i) {
                double dot = 0.0;
                for (size_t c = 0; c < 4u; ++c) {
                  dot += static_cast<double>(values[i * 4u + c]) * node.rotation[c];
                }
                if (1.0 - std::abs(dot) > 1e-6) {
                  return false;
                }
              }
              return true;
            });
          }

          // Scale (uniform).
          if (sampleChannel(evaluator.scale, &values)) {
            std::vector<float> scales(values.size() * 3u);
            for (size_t i = 0; i < values.size(); ++i) {
              scales[i * 3u + 0u] = scales[i * 3u + 1u] = scales[i * 3u + 2u] = values[i];
            }
            addChannels(targets, "scale", scales, TINYGLTF_TYPE_VEC3, 3u, [&](tinygltf::Node const& node) {
              for (size_t i = 0; i < scales.size(); ++i) {
                double const rest = (node.scale.size() == 3u) ? node.scale[i % 3u] : 1.0;
                if (std::abs(static_cast<double>(scales[i]) - rest) > static_cast<double>(kEpsilon)) {
                  return false;
                }
              }
              return true;
            });
          }
        }
        unsupportedEvaluators.insert(seq.unsupported.begin(), seq.unsupported.end());

        if (anim.channels.empty()) {
          DOJIMA_LOG( "[Warning] " << kfFilename << " : sequence '" << anim.name << "' has no channel matching the model." );
          continue;
        }

        // Keep the playback informations of the game, as glTF has no equivalent.
        // [TODO: extract as custom file / json ?]
        tinygltf::Value::Array textKeys;
        for (auto const& key : seq.textKeys) {
          textKeys.push_back(tinygltf::Value(tinygltf::Value::Object{
            { "time", tinygltf::Value(static_cast<double>(key.time)) },
            { "text", tinygltf::Value(key.text) },
          }));
        }
        anim.extras = tinygltf::Value(tinygltf::Value::Object{
          { "cycle",       tinygltf::Value(std::string(CycleTypeName(seq.cycle))) },
          { "duration",    tinygltf::Value(static_cast<double>(seq.duration)) },
          { "frequency",   tinygltf::Value(static_cast<double>(seq.frequency)) },
          { "accumRoot",   tinygltf::Value(seq.accumRoot) },
          { "accumFlags",  tinygltf::Value(static_cast<int>(seq.accumFlags)) },
          { "rootMotion",  tinygltf::Value(gKeepRootMotion) },
          { "textKeys",    tinygltf::Value(textKeys) },
        });

        numChannels += anim.channels.size();
        ++numPlayed;
        Animations.push_back(std::move(anim));
      }

      DOJIMA_LOG( kfFilename << " : " << numPlayed << " animation(s) exported (" << numChannels << " channels, "
                  << gAnimFps << " fps" << (gKeepRootMotion ? ", with root motion" : ", in place") << ")." );
      if (!missingTargets.empty()) {
        std::string names;
        size_t shown = 0;
        for (auto const& name : missingTargets) {
          names += (shown++ < 8u) ? (name + " ") : "";
        }
        DOJIMA_LOG( "[Warning] " << missingTargets.size() << " animated node(s) not found in the model : " << names
                    << ((missingTargets.size() > 8u) ? "..." : "") );
      }
      for (auto const& type : unsupportedEvaluators) {
        DOJIMA_LOG( "[Warning] " << type << " evaluators are not supported (skipped)." );
      }
    }
  }

  tinygltf::Model data{};
  {
    data.asset.generator = "nif2gltf";
    data.asset.version = "2.0";

    data.accessors    = std::move( Accessors );
    data.animations   = std::move( Animations );
    data.buffers      = std::move( Buffers );
    data.bufferViews  = std::move( BufferViews );
    data.materials    = std::move( Materials );
    data.meshes       = std::move( Meshes );
    data.nodes        = std::move( Nodes );
    data.textures     = std::move( Textures );
    data.images       = std::move( Images );
    data.skins        = std::move( Skins );
    data.samplers     = std::move( Samplers );
    data.lights       = std::move( Lights );
    // data.cameras;

    tinygltf::Scene scene;
    scene.nodes.push_back(0);
    data.scenes.push_back(scene);
  }

  std::string const gltfFilename{
    [](std::string s, std::string const& ext) -> std::string {
      return s.replace(
        s.begin() + s.find_last_of(".") + 1,
        s.end(),
        ext
      ).substr(s.find_last_of(kSystemPathSeparator) + 1);
    }(nifFilename, kGLTFWriteBinary ? "glb" : "gltf")
  };

  if (tinygltf::TinyGLTF gltf; !gltf.WriteGltfSceneToFile(
      &data, outputDirectory + gltfFilename, kGLTFEmbedImages, kGLTFEmbedBuffers, kGLTFPrettyPrint, kGLTFWriteBinary
    ))
  {
    DOJIMA_LOG( "[Error] glTF file writing failed." );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
