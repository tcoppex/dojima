#include <cassert>
#include <cstdlib>
#include <cstddef>

#include <array>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "nif/niflib.h"

#include "nif/gen/Header.h"
#include "nif/obj/NiDataStream.h"
#include "nif/obj/NiMesh.h"
#include "nif/obj/NiNode.h"

#include "nif/obj/NiIntegerExtraData.h"
#include "nif/obj/NiStringExtraData.h"
#include "nif/obj/NiPixelData.h"

#define TINYGLTF_NOEXCEPTION
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tinygltf/tiny_gltf.h"

// ----------------------------------------------------------------------------

#define DOJIMA_LOG(x) std::cerr << x << std::endl

#ifndef NDEBUG
#define DOJIMA_QUIET_LOG(x)  DOJIMA_LOG(x)
#else
#define DOJIMA_QUIET_LOG(x)  if (static bool b=false; !b && (b=true)) DOJIMA_LOG("(quiet log) " << x)
#endif

// ----------------------------------------------------------------------------

// Display NIF header information at loading.
static constexpr bool kDisplayHeader = false;

// Force the output filename to "out.gltf".
static constexpr bool kDebugOutput = true;

// Output name for accessors.
static constexpr bool kNameAccessors = false;

// OS specific path separator.
static constexpr char kSystemPathSeparator{
#if defined(WIN32) || defined(_WIN32)
  '\\'
#else
  '/'
#endif
};

// ----------------------------------------------------------------------------

enum class BufferId : uint8_t {
  INDEX,
  VERTEX,

  kCount
};

enum class AttributeId : uint8_t {
  Position,
  Texcoord,
  Normal,
  Binormal,
  Tangent,
  Joints,
  Weights,
  
  kCount
};

// ----------------------------------------------------------------------------

// NIF node keys of interest.
std::string const kNiDataStreamKey{ "NiDataStream" };
std::string const kNiMeshKey{ "NiMesh" };

// ----------------------------------------------------------------------------

static bool SetAttribute(Niflib::SemanticData cs, size_t accessorIndex, tinygltf::Primitive *prim, bool *bNormalized) {
  std::string const name = cs.name;
  std::string const suffix{"_" + std::to_string(cs.index)};

  if ((name == "POSITION") || (name == "POSITION_BP")) {
    prim->attributes["POSITION"] = accessorIndex;
    *bNormalized = false;
  } 
  else if ((name == "NORMAL") || (name == "NORMAL_BP")) {
    prim->attributes["NORMAL"] = accessorIndex;
    *bNormalized = true;
  } 
  else if (name == "TEXCOORD") {
    prim->attributes["TEXCOORD" + suffix] = accessorIndex;
    // *bNormalized = true;
  } 
  else if (name == "BLENDINDICES") {
    prim->attributes["JOINTS" + suffix] = accessorIndex;
    *bNormalized = false;
  } 
  else if (name == "BLENDWEIGHT") {
    prim->attributes["WEIGHTS" + suffix] = accessorIndex;
    *bNormalized = true;
  } 
  else if ((name == "TANGENT") || (name == "TANGENT_BP")) {
    prim->attributes["TANGENT"] = accessorIndex;
    *bNormalized = true;
  }
  else if ((name == "BINORMAL") || (name == "BINORMAL_BP")) {
    prim->attributes["BINORMAL"] = accessorIndex;
    *bNormalized = true;
  }
  else {
    DOJIMA_QUIET_LOG( "[Warning] " << __FUNCTION__ << " : " << name << " component not used." ); 
    return false;
  }

  return true;
}

static uint32_t GetPrimitiveType(Niflib::MeshPrimitiveType primType) {
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

static size_t SetAccessorFormat(Niflib::ComponentFormat format, tinygltf::Accessor &accessor) {
  switch(format) {
    case Niflib::F_UINT16_1:
      accessor.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
      accessor.type = TINYGLTF_TYPE_SCALAR;
    return 1 * sizeof(uint16_t);

    case Niflib::F_FLOAT32_2:
      accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
      accessor.type = TINYGLTF_TYPE_VEC2;
    return 2 * sizeof(float);
    
    case Niflib::F_FLOAT32_3:
      accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
      accessor.type = TINYGLTF_TYPE_VEC3;
    return 3 * sizeof(float);

    case Niflib::F_UINT8_4:
      accessor.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
      accessor.type = TINYGLTF_TYPE_VEC4;
    return 4 * sizeof(uint8_t);

    // [ todo : convert float16 data to float32 ]
    case Niflib::F_FLOAT16_2:
    case Niflib::F_FLOAT16_4:
    default:
      DOJIMA_QUIET_LOG( "[Warning] " << __FUNCTION__ << " : unsupported format " << format << "." );
    return 0;
  };
}

// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  if (argc < 2) {
    DOJIMA_LOG( "Usage : " << std::endl << argv[0] << " nif_file" );
    return EXIT_FAILURE;
  }
  std::string const nifFilename{ argv[1] };
  
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

  // --------------

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

  // --------------

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

    std::cerr << " * Blocks type index (" << blockTypeIndex.size() << ") :" << std::endl;
    for (auto const &bti : blockTypeIndex) {
      std::cerr << bti << ", ";
    }
    std::cerr << std::endl;
  }

  // --------------

  // WotS4 has 3 data streams per Mesh :
  //    * USAGE_VERTEX_INDEX  : (INDEX)
  //    * USAGE_VERTEX        : (Texcoord, Position, Normal, Binormal, Tangent, BlendIndice, BlendWeight)
  //    * USAGE_USER          : (BONE_PALETTE)
  //
  //  Process :
  //
  // 1) We create 2 buffers for indices and vertices. 
  //       (we don't use BONE_PALETTE for now)
  //
  // 2) For each NiMesh we create :
  //    * 2 buffer_views for indices and vertices (one per used datastreams).
  //
  // 3) For each mesh's submeshes we create :
  //    * 1 accessors for indices.
  //    * 7 accessors for vertices.
  //
  // [ better approach ] use one buffer per LODs.
  //
  auto nifList = Niflib::ReadNifList(nifFilename);

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
      verticesBytesize += bytesize;
    }
  }

  // Create the base Index & Vertex buffers object.
  std::vector< tinygltf::Buffer > Buffers(static_cast<size_t>(BufferId::kCount));
  
  auto &IndexBuffer = Buffers[(int)BufferId::INDEX];
  IndexBuffer.name = "Indices";
  IndexBuffer.data.reserve(indicesBytesize);

  auto &VertexBuffer = Buffers[(int)BufferId::VERTEX];
  VertexBuffer.name = "Vertices";
  VertexBuffer.data.reserve(verticesBytesize);
  
  // Offset to current buffers' end, used to build buffer viewers.
  size_t indexBufferOffset = 0;
  size_t vertexBufferOffset = 0;

  // Retrieve the number of NIF mesh nodes.
  uint32_t const kNiMeshCount{ getNumBlocks(kNiMeshKey) };
  
  // Calculate the total number of submeshes.
  uint32_t totalSubmeshesCount{ 0 };
  {
    auto const& meshIndices = getTypeListIndices( kNiMeshKey );
    for (auto mesh_id : meshIndices) {
      Niflib::NiObject *obj = nifList[mesh_id];
      Niflib::NiMesh* nifMesh = (Niflib::NiMesh*)obj;
      totalSubmeshesCount += nifMesh->GetNumSubmeshes();
    }
  }

  // Meshes buffers.
  std::vector< tinygltf::Mesh > Meshes( kNiMeshCount );
  std::vector< tinygltf::BufferView > BufferViews( Buffers.size() * kNiMeshCount );

  // Submeshes buffers.
  size_t constexpr kNumAccessorPerSubmesh{ 1 + static_cast<size_t>(AttributeId::kCount) };
  std::vector< tinygltf::Accessor > Accessors( totalSubmeshesCount * kNumAccessorPerSubmesh );
  std::vector< tinygltf::Material > Materials( 1 ); // TODO

  // Index to available buffer slot.
  size_t current_buffer_view = 0;
  size_t current_accessor = 0;
  size_t current_material = 0;

  // --------------

  // Associate a mesh name to its internal id.
  std::unordered_map<std::string, size_t> mapMeshNameToId;

  // Setup Meshes.
  auto const& meshIndices = getTypeListIndices( kNiMeshKey );
  for (size_t mesh_id = 0; mesh_id < meshIndices.size(); ++mesh_id) {
    auto const nifMesh = Niflib::StaticCast<Niflib::NiMesh>(nifList[meshIndices[mesh_id]]);
    auto const primType = GetPrimitiveType(nifMesh->GetPrimitiveType());
    auto &mesh = Meshes.at(mesh_id);
    
    // Name.
    mesh.name = nifMesh->GetName();
    mapMeshNameToId[mesh.name] = mesh_id;

    // Primitives / submeshes.
    uint32_t const nsubmeshes = nifMesh->GetNumSubmeshes();
    mesh.primitives.resize(nsubmeshes);

    // Create a buffer_view for each used datastream.
    uint32_t numMeshdata = 0;
    for (auto const& md : nifMesh->GetMeshDatas()) {
      auto const& stream = md.stream;
      auto const& usage = stream->GetUsage();

      // Bypass unused meshData.
      if ((usage == Niflib::USAGE_SHADER_CONSTANT) ||
          (usage == Niflib::USAGE_USER)) {
        // DOJIMA_QUIET_LOG( "[Warning] A " << usage << " MeshData was not used." );
        continue;
      }

      if (++numMeshdata > static_cast<uint32_t>(BufferId::kCount)) {
        DOJIMA_QUIET_LOG( "[Warning] An exceeding amount of internal buffers has been found." );
        break;
      }

      // -- BUFFER VIEW.

      int const bufferViewIndex = current_buffer_view++;
      auto &bufferView = BufferViews.at(bufferViewIndex);

      bufferView.byteLength = stream->GetNumBytes();
      auto const& data = stream->GetData();
      auto const& formats = stream->GetComponentFormats();

      switch (usage) {
        case Niflib::USAGE_VERTEX_INDEX:
          bufferView.buffer = (int)BufferId::INDEX;
          bufferView.byteOffset = indexBufferOffset;
          bufferView.byteStride = 0;
          bufferView.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
          indexBufferOffset += bufferView.byteLength;

          IndexBuffer.data.insert(IndexBuffer.data.end(), data.cbegin(), data.cend());
        break;

        case Niflib::USAGE_VERTEX:
          bufferView.buffer = (int)BufferId::VERTEX;
          bufferView.byteOffset = vertexBufferOffset;
          bufferView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
          vertexBufferOffset += bufferView.byteLength;

          // Calculate the interleaved attributes bytestride.
          bufferView.byteStride = 0;
          for (int comp_id = 0; comp_id < md.numComponents; ++comp_id) {
            tinygltf::Accessor dummy_accessor;
            size_t const byteSize = SetAccessorFormat(formats[comp_id], dummy_accessor);
            bufferView.byteStride += byteSize;
          }

          VertexBuffer.data.insert(VertexBuffer.data.end(), data.cbegin(), data.cend());
        break;

        default:
        break;
      };

      // -- Submeshes ACCESSORs.
      
      auto const& regions = stream->GetRegions();
      
      for (uint32_t submesh_id = 0; submesh_id < nsubmeshes; ++submesh_id) {
        auto &prim = mesh.primitives[submesh_id];
        prim.mode = primType;

        // Creates accessors for each submeshes. 
        uint16_t const regionMap = md.submeshToRegionMap[submesh_id];
        auto const& region = regions[regionMap];
        
        if (usage == Niflib::USAGE_VERTEX_INDEX) {
          // -- Indices

          // Set the primitive indices accessor.
          prim.indices = current_accessor++;
          
          auto &acc = Accessors[prim.indices];
          size_t const byteSize = SetAccessorFormat(formats[0], acc);

          if (kNameAccessors) acc.name = "INDEX";
          acc.byteOffset = region.startIndex * byteSize; //
          acc.count = region.numIndices;
          acc.normalized = false;
          acc.bufferView = bufferViewIndex;
        } else {
          // -- Vertices
          
          size_t vertexBaseOffset = region.startIndex * bufferView.byteStride;
          for (int comp_id = 0; comp_id < md.numComponents; ++comp_id) {
            auto const& sem = md.componentSemantics[comp_id];
            
            // Set the primitive attribute accessor.
            bool bNormalized = false;
            SetAttribute(sem, current_accessor, &prim, &bNormalized);

            auto &acc = Accessors[current_accessor++];
            size_t const byteSize = SetAccessorFormat(formats[comp_id], acc);

            if (kNameAccessors) acc.name = sem.name;
            acc.byteOffset = vertexBaseOffset; //
            acc.count = region.numIndices;
            acc.normalized = bNormalized;
            acc.bufferView = bufferViewIndex;

            vertexBaseOffset += byteSize;
          }
        }

        // TODO : material.
        {
          // auto &mat = Materials.at(0);
          prim.material = 0;
        }

      } // -- end submeshes
    } // -- end meshdatas

    // TODO : NiSkinningMeshModifier & extra properties.
    //mesh.weights;
    //mesh.extra;
  }

  // --------------

  // Node hierarchy.

  // Determine the total number of hierarchical nodes.
  size_t totalNodeCount = 0;
  for (auto const& n : nifList) {
    auto ptr = Niflib::DynamicCast<Niflib::NiAVObject>(n);
    totalNodeCount += (ptr != nullptr) ? 1 : 0;
  }

  std::vector< tinygltf::Node > Nodes;
  Nodes.reserve(totalNodeCount);

  // Define a recursive lambda function to fill the node hierarchy.
  std::function<void(tinygltf::Node*const, Niflib::NiAVObject *const)> fillNodes;
  
  fillNodes = [&mapMeshNameToId, &Nodes, &fillNodes](tinygltf::Node *const parent, Niflib::NiAVObject *const nifAVO) -> void {
    // Get a reference to a new node.
    Nodes.push_back( tinygltf::Node() );
    auto &node = Nodes.back();
    
    // Set it as its parent's children.
    if (parent) {
      int const nodeIndex = static_cast<int>(Nodes.size() - 1);
      parent->children.push_back(nodeIndex);
    }

    node.name = nifAVO->GetName();

    auto const& qRotation = nifAVO->GetLocalRotation().AsQuaternion();
    node.rotation = { qRotation.x, qRotation.y, qRotation.z, qRotation.w };

    auto const fScale = nifAVO->GetLocalScale();
    node.scale = { fScale, fScale, fScale };

    auto const &vTranslation = nifAVO->GetLocalTranslation();
    node.translation = { vTranslation.x, vTranslation.y, vTranslation.z };

    if (auto nifMesh = Niflib::DynamicCast<Niflib::NiMesh>(nifAVO); nifMesh) {
      node.mesh = mapMeshNameToId[node.name];
    }

    // Detect meshLOD node (nifSwitchNode).
    // if (auto nifSwitchNode = Niflib::DynamicCast<Niflib::NiSwitchNode>(nifAVO); nifSwitchNode) {}

    // Handle extra datas.
    // auto const& listXD = nifAVO->GetExtraData();
    // node.extras; // TODO    

    // Continue processing if the nif object is a pure node (and not only a leaf).
    if (auto nifNode = Niflib::DynamicCast<Niflib::NiNode>(nifAVO); nifNode) {
      if (nifNode->IsSkeletonRoot()) {
        // node.skin;
      }
      for (auto &child : nifNode->GetChildren()) {
        fillNodes(&node, child);
      }
    }
  };


  /// (for Characters Pack)
  ///  The first node is a "NiIntegerExtraData" and contains the number of
  ///   "NiStringExtraData" on the upper root.
  ///
  /// Then for each NiStringExtraData, if the its following object has a consecutive index
  /// it contains the path for a part's data, and is either a NiNode or a NiPixelData.

  // Detects if the NIF file is a pack.
  uint32_t numSubParts = 0u;
  if (auto first = Niflib::DynamicCast<Niflib::NiIntegerExtraData>(nifList[0]); first) {
    numSubParts = first->GetData() / 2u;
  }
  if (numSubParts > 0u) {
    struct PartIndices_t {
      int path_id = -1; // index to a NiStringExtraData containing a path.
      int data_id = -1; // index to either a NiNode or a NiPixelData.
    };
    
    std::vector<PartIndices_t> nodeParts;
    nodeParts.reserve(numSubParts);
    
    std::vector<PartIndices_t> pixelParts;
    pixelParts.reserve(numSubParts);

    // Retrieve each parts info indices.    
    for (int i = 0; i < (int)nifList.size(); ++i) {
      auto n0 = nifList[i];
      if (auto nifString = Niflib::DynamicCast<Niflib::NiStringExtraData>(n0); nifString) {
        auto n1 = nifList[i+1];
        if (auto nifNode = Niflib::DynamicCast<Niflib::NiNode>(n1); nifNode && (nifNode->GetName() == "SceneNode")) {
          nodeParts.push_back( {i, i+1} );
          ++i; continue;
        }
        if (auto nifPixelData = Niflib::DynamicCast<Niflib::NiPixelData>(n1); nifPixelData) {
          pixelParts.push_back( {i, i+1} );
          ++i; continue;
        }
      }
    }

    // Append the node part to the nodes hierarchy.
    for (size_t i = 0; i < nodeParts.size(); ++i) {
      auto const nodeId = nodeParts[i].data_id;
      auto upperNode = Niflib::DynamicCast<Niflib::NiNode>(nifList[nodeId]);
      fillNodes(nullptr, upperNode);

      // TODO :
      //  * set mesh part path.
      //  * set pixel data.
    }
  } else if (auto upperNode = Niflib::DynamicCast<Niflib::NiNode>(nifList[0]); upperNode) {
    fillNodes(nullptr, upperNode);
  }

  // --------------

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

    //data.images;
    //data.textures;
    //data.samplers;

    //data.skins;
    //data.animations;

    //data.cameras;
    //data.lights;
    
    // ----

    tinygltf::Scene scene;
    if constexpr(true) {
      scene.nodes.push_back(0);
    } else {
      scene.nodes.resize(data.nodes.size());
      std::iota(scene.nodes.begin(), scene.nodes.end(), 0);
    }
    data.scenes.push_back(scene);
  }

  // ------

  constexpr bool bEmbedImages = true;
  constexpr bool bEmbedBuffers = true;
  constexpr bool bPrettyPrint = true;
  constexpr bool bWriteBinary = false;

  std::string const gltfFilename{
    (kDebugOutput) ? "out.gltf" :
    [](std::string s, std::string const& ext) -> std::string {
      return s.replace(
        s.begin() + s.find_last_of(".") + 1, 
        s.end(), 
        ext
      ).substr(s.find_last_of(kSystemPathSeparator) + 1);
    }(nifFilename, bWriteBinary ? "glb" : "gltf")
  };

  if (tinygltf::TinyGLTF gltf; !gltf.WriteGltfSceneToFile(
      &data, gltfFilename, bEmbedImages, bEmbedBuffers, bPrettyPrint, bWriteBinary
    )) 
  {
    DOJIMA_LOG( "[Error] glTF file writing failed." );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}