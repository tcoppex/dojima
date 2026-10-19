// ----------------------------------------------------------------------------
//
//    (Work In Progress) Converter for WotS 4 files (Characters and Maps).
//
//    TODO
//      * Fix mesh attributes buffers to remove NIF's BINORMAL and
//  recompute TANGENT to be true VEC4 (this will require to change
//  bufferView and accessors too).
//
//
//  Use a glTF validator to check errors and warnings.
//    ( https://github.khronos.org/glTF-Validator/ )
//
// ----------------------------------------------------------------------------

#include <cassert>
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

} // namespace ""

// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  if (argc < 2) {
    DOJIMA_LOG( "Usage : " << std::endl << argv[0] << " nif_file" );
    return EXIT_FAILURE;
  }
  std::string const nifFilename{ argv[1] };

  // Create an outputs directory for the transformed NIF file.
  std::string const outputDirectory{
    [](std::string s) -> std::string {
      return "_outputs" +
             std::string(&kSystemPathSeparator) +
             s.replace(s.begin() + s.find_last_of("."), s.end(), "")
              .substr(s.find_last_of(kSystemPathSeparator) + 1) +
             std::string(&kSystemPathSeparator);
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

  // Meshes buffers.
  std::vector<tinygltf::Mesh> Meshes( kNiMeshCount );

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
  std::vector<tinygltf::Material> Materials( kNiMeshCount ); // (might be lower)

  // [ TODO ]
  // Samplers buffer.
  std::vector<tinygltf::Sampler> Samplers( 1 ); //
  // std::map< uint16_t, size_t > mapTexDescFlagToSamplerIndex;

  // Index to available material slot.
  size_t current_material = 0; //

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
      auto const texName{
        [](std::string const& fn, size_t pos) -> std::string {
          size_t startIndex = (pos != std::string::npos) ? pos + 1 : 0;
          return fn.substr(startIndex, fn.find_last_of('.'));
        }(texFilename, texFilename.find_last_of('/'))
      };

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
    [processInternalTexture](Niflib::TexDesc const& texDesc, size_t* texIndex) -> bool {
      auto srcTex = texDesc.source;
      if (nullptr == srcTex) {
        return false;
      }
      if (srcTex->IsTextureExternal()) {
        DOJIMA_QUIET_LOG( "[ Warning ] External texture loading is not supported." );
        return false;
      }
      // [ TODO ]
      // Retrieve sampler from texdesc flag and samplerMap.
      return processInternalTexture(srcTex->GetPixelData(), srcTex->GetTextureFileName(), texIndex);
    }
  };

  // -- Meshes

  // Associate a mesh name to its internal id.
  std::unordered_map<std::string, size_t> mapMeshNameToId;
  std::unordered_map<std::string, size_t> mapMeshNameToMaterialId;

  auto const& meshIndices = getTypeListIndices( kNiMeshKey );
  for (size_t mesh_id = 0; mesh_id < meshIndices.size(); ++mesh_id) {
    auto const nifMesh = Niflib::StaticCast<Niflib::NiMesh>(nifList[meshIndices[mesh_id]]);
    auto const primType = GetPrimitiveType(nifMesh->GetPrimitiveType());
    auto &mesh = Meshes.at(mesh_id);

    // Name.
    mesh.name = nifMesh->GetName();
    mapMeshNameToId[mesh.name] = mesh_id;

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
    size_t const meshMaterialIndex = current_material++; //
    auto &material = Materials[meshMaterialIndex];
    material.name = mesh.name + "_mat_" + std::to_string(meshMaterialIndex); //

    mapMeshNameToMaterialId[mesh.name] = meshMaterialIndex; //

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
      material.pbrMetallicRoughness.baseColorFactor = { diffuse.r, diffuse.g, diffuse.b, 1.0 };
    }

    // Alpha Testing & Blending.
    if (auto prop = nifMesh->GetPropertyByType(Niflib::NiAlphaProperty::TYPE); prop) {
      auto alphaProp = Niflib::StaticCast<Niflib::NiAlphaProperty>(prop);

      auto const alphaTestFunc = alphaProp->GetTestFunc();
      switch (alphaTestFunc) {
        case Niflib::NiAlphaProperty::TF_LESS:
          material.alphaMode = "BLEND";
        break;

        case Niflib::NiAlphaProperty::TF_GREATER:
          // (weirdly, some body parts have this flag set for no clear reasons,
          // should we discard them ?)
        case Niflib::NiAlphaProperty::TF_GEQUAL:
          material.alphaMode = "MASK";
          material.alphaCutoff = alphaProp->GetTestThreshold() / 255.0;
        break;

        case Niflib::NiAlphaProperty::TF_ALWAYS:
        default:
          material.alphaMode = "OPAQUE";
        break;
      }
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
        // indices refer to the submesh BONE_PALETTE.
        // [ TODO ] remap the joints through the bone palette once skins are exported.
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
            for (int c = 0; c < 4; ++c) {
              ww[c] = (sum > 0.0f) ? ww[c] / sum : ((c == 0) ? 1.0f : 0.0f);
              weights[i * 4u + c] = ww[c];
              // (glTF requires unused joints to have a null index)
              joints[i * 4u + c] = (ww[c] > 0.0f) ? static_cast<uint16_t>(j[c]) : uint16_t(0);
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
  }

  Nodes.reserve(totalNodeCount);

  if (kUseRootNode) {
    Nodes.push_back({});
    root_node_ptr = &Nodes[0];
    root_node_ptr->name = "root";
  }


  // [lambda] recursive function to fill the node hierarchy.
  std::function<void(tinygltf::Node *const, Niflib::NiAVObject *const, size_t *)> fillNodes{
    [&](tinygltf::Node *const parent, Niflib::NiAVObject *const nifAVO, size_t *mainTexId) -> void {
      // [ test, skip empty nodes ]
      // if (auto nifMesh = Niflib::DynamicCast<Niflib::NiMesh>(nifAVO); !nifMesh) {
      //   return;
      // }

      // Get a reference to a new node.
      Nodes.push_back( tinygltf::Node() );
      auto &node = Nodes.back();

      // Set it as its parent's children.
      if (parent) {
        int const nodeIndex = static_cast<int>(Nodes.size()) - 1;
        parent->children.push_back(nodeIndex);
      }

      node.name = nifAVO->GetName();

      // auto nnn = Niflib::DynamicCast<Niflib::NiNode>(nifAVO);
      // DOJIMA_LOG( mapNodeNameToId[node.name] // < " " << (nnn ? nnn->GetType().GetTypeName() : "")
      //                                        << " " << node.name
      //           );

      auto const& qRotation = nifAVO->GetLocalRotation().AsQuaternion();
      node.rotation = { (double)qRotation.x, (double)qRotation.y, (double)qRotation.z, (double)qRotation.w };

      auto const fScale = static_cast<double>(nifAVO->GetLocalScale());
      node.scale = { fScale, fScale, fScale };

      auto const &vTranslation = nifAVO->GetLocalTranslation();
      node.translation = { (double)vTranslation.x, (double)vTranslation.y, (double)vTranslation.z };

      if (auto nifMesh = Niflib::DynamicCast<Niflib::NiMesh>(nifAVO); nifMesh) {
        node.mesh = mapMeshNameToId[node.name];

        if (mainTexId) {
          auto &mat = Materials.at(mapMeshNameToMaterialId[node.name]);
          mat.pbrMetallicRoughness.baseColorTexture.index = *mainTexId; //
        }
      }

      // [ TODO ] Detect meshLOD node (nifSwitchNode).
      // if (auto nifSwitchNode = Niflib::DynamicCast<Niflib::NiSwitchNode>(nifAVO); nifSwitchNode) {}

      // [ TODO ] Handle extra datas.
      // auto const& listXD = nifAVO->GetExtraData();
      // node.extras;

      // Continue processing if the nif object is a pure node (and not only a leaf).
      if (auto nifNode = Niflib::DynamicCast<Niflib::NiNode>(nifAVO); nifNode) {
        if (nifNode->IsSkeletonRoot()) {
          // node.skin;
        }
        for (auto &child : nifNode->GetChildren()) {
          fillNodes(&node, child, mainTexId);
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

    // Append the node part to the nodes hierarchy.
    for (size_t i = 0; i < nodeParts.size(); ++i) {
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
      auto const pixelPart = pixelParts[i];
      auto texPathNode = Niflib::StaticCast<Niflib::NiStringExtraData>(nifList.at(pixelPart.path_id));
      auto texPixelNode = Niflib::StaticCast<Niflib::NiPixelData>(nifList.at(pixelPart.data_id));

      if (size_t texIndex; processInternalTexture(texPixelNode, texPathNode->GetData(), &texIndex)) {
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

  tinygltf::Model data{};
  {
    data.asset.generator = "nif2gltf";
    data.asset.version = "2.0";

    data.meshes = std::move( Meshes );
    data.materials = std::move( Materials );
    data.accessors = std::move( Accessors );
    data.bufferViews = std::move( BufferViews );
    data.buffers = std::move( Buffers );
    data.nodes = std::move( Nodes );

    if constexpr(true) {
      data.images = std::move( Images );
      data.textures = std::move( Textures );
      data.samplers = std::move( Samplers );
    }

    //data.skins;
    //data.animations;
    //data.cameras;
    //data.lights;

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
