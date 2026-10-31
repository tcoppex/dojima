// ----------------------------------------------------------------------------
//
//    Reader and sampler for the WotS 4 keyframe files (*.kf).
//
// ----------------------------------------------------------------------------

#ifndef DOJIMA_KF_H_
#define DOJIMA_KF_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

namespace kf {

// ----------------------------------------------------------------------------

constexpr uint32_t kNullLink = 0xffffffffu;
constexpr uint32_t kInvalidHandle = 0xffffu;

// Sequence's accumulation flags (movement of the accumulation root which is
// meant to be applied to the whole actor instead of its animation).
enum AccumFlags : uint32_t {
  kAccumXTrans = 0x01,
  kAccumYTrans = 0x02,
  kAccumZTrans = 0x04,
  kAccumXRot   = 0x08,
  kAccumYRot   = 0x10,
  kAccumZRot   = 0x20,
};

enum class CycleType : uint32_t {
  kLoop    = 0,
  kReverse = 1,
  kClamp   = 2,
};

// Interpolation of the keys of a NiTransformData.
enum KeyType : uint32_t {
  kKeyLinear    = 1,
  kKeyQuadratic = 2,  // Bezier / Hermite, with tangents (not for rotations).
  kKeyTBC       = 3,  // Tension / Continuity / Bias, evaluated as linear.
  kKeyXYZRot    = 4,  // Euler angles, converted to quaternions at loading.
  kKeyConstant  = 5,  // Step.
};

// ----------------------------------------------------------------------------

// Animation of one component of a transform : translation (3 floats), rotation
// (4 floats, w x y z) or scale (1 float).
struct Channel {
  enum class Kind : uint8_t {
    kNone,      // Not animated, the node keeps its own value.
    kConstant,  // Always `constant`.
    kKeys,      // Keyframes.
    kSpline,    // B-spline.
  };

  Kind kind = Kind::kNone;
  int numComps = 0;
  std::array<float, 4> constant{};

  // Keyframes.
  uint32_t keyType = kKeyLinear;
  std::vector<float> times;
  std::vector<float> values;    // numComps per key.
  std::vector<float> tangents;  // 2 * numComps per key : forward, backward (quadratic keys).

  // B-spline.
  float start = 0.0f;
  float stop = 0.0f;
  uint32_t numControlPoints = 0;
  std::vector<float> control;   // numComps per control point.

  // True when the channel changes over time.
  bool IsAnimated() const {
    return (kind == Kind::kSpline) || ((kind == Kind::kKeys) && (times.size() > 1u));
  }

  bool IsRotation() const { return numComps == 4; }
};

// Animation of a node (referenced by its name).
struct Evaluator {
  std::string target;
  Channel translation;
  Channel rotation;
  Channel scale;
};

struct TextKey {
  float time = 0.0f;
  std::string text;
};

struct Sequence {
  std::string name;
  float duration = 0.0f;
  float frequency = 1.0f;
  CycleType cycle = CycleType::kLoop;
  std::string accumRoot;
  uint32_t accumFlags = 0u;
  std::vector<TextKey> textKeys;
  std::vector<Evaluator> evaluators;

  // Types of the evaluators which were not decoded.
  std::vector<std::string> unsupported;
};

struct File {
  std::vector<Sequence> sequences;
};

// ----------------------------------------------------------------------------
//  Sampling.
// ----------------------------------------------------------------------------

namespace detail {

constexpr float kInvalidFloat = -3.402823466e+38f; // (-FLT_MAX : unset pose value)

inline bool IsValidFloat(float v) {
  return v > -3.0e+38f;
}

inline void Normalize4(float *q) {
  float const l = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  if (l > 1e-12f) {
    for (int i = 0; i < 4; ++i) q[i] /= l;
  } else {
    q[0] = 1.0f; q[1] = q[2] = q[3] = 0.0f;
  }
}

// Quaternions (w, x, y, z).
inline void QuatMul(float const* a, float const* b, float *out) {
  float const w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
  float const x = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
  float const y = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
  float const z = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
  out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}

inline void QuatSlerp(float const* a, float const* b, float t, float *out) {
  float bb[4] = { b[0], b[1], b[2], b[3] };
  float d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
  if (d < 0.0f) {
    d = -d;
    for (float &v : bb) v = -v;
  }
  float wa = 1.0f - t;
  float wb = t;
  if (d < 0.9995f) {
    float const theta = std::acos(d);
    float const s = std::sin(theta);
    wa = std::sin((1.0f - t) * theta) / s;
    wb = std::sin(t * theta) / s;
  }
  for (int i = 0; i < 4; ++i) out[i] = wa * a[i] + wb * bb[i];
  Normalize4(out);
}

// Index of the key segment containing `time` (times[i] <= time < times[i+1]).
inline size_t FindSegment(std::vector<float> const& times, float time) {
  auto it = std::upper_bound(times.begin(), times.end(), time);
  size_t const i = static_cast<size_t>(std::distance(times.begin(), it));
  return (i == 0u) ? 0u : (i - 1u);
}

inline void EvalKeys(Channel const& c, float time, float *out) {
  size_t const n = c.times.size();
  size_t const nc = static_cast<size_t>(c.numComps);
  if ((n == 1u) || (time <= c.times.front())) {
    std::copy_n(c.values.begin(), nc, out);
    return;
  }
  if (time >= c.times.back()) {
    std::copy_n(c.values.begin() + static_cast<std::ptrdiff_t>((n - 1u) * nc), nc, out);
    return;
  }

  size_t const i = FindSegment(c.times, time);
  float const* v0 = &c.values[i * nc];
  float const* v1 = &c.values[(i + 1u) * nc];
  float const dt = c.times[i + 1u] - c.times[i];
  float const s = (dt > 0.0f) ? (time - c.times[i]) / dt : 0.0f;

  if (c.keyType == kKeyConstant) {
    std::copy_n(v0, nc, out);
  } else if (c.numComps == 4) {
    QuatSlerp(v0, v1, s, out);
  } else if ((c.keyType == kKeyQuadratic) && (c.tangents.size() == 2u * nc * n)) {
    // Hermite curve, tangents are already scaled by the key interval.
    float const* m0 = &c.tangents[(i * 2u + 0u) * nc];
    float const* m1 = &c.tangents[((i + 1u) * 2u + 1u) * nc];
    float const s2 = s * s;
    float const s3 = s2 * s;
    float const h00 = 2.0f * s3 - 3.0f * s2 + 1.0f;
    float const h10 = s3 - 2.0f * s2 + s;
    float const h01 = -2.0f * s3 + 3.0f * s2;
    float const h11 = s3 - s2;
    for (size_t k = 0; k < nc; ++k) {
      out[k] = h00 * v0[k] + h10 * m0[k] + h01 * v1[k] + h11 * m1[k];
    }
  } else {
    for (size_t k = 0; k < nc; ++k) {
      out[k] = v0[k] + (v1[k] - v0[k]) * s;
    }
  }
}

// Cubic (or lower when there are less control points), uniform, clamped B-spline.
inline void EvalSpline(Channel const& c, float time, float *out) {
  size_t const nc = static_cast<size_t>(c.numComps);
  int const n = static_cast<int>(c.numControlPoints);
  if (n <= 0) {
    std::fill_n(out, nc, 0.0f);
    return;
  }

  int const degree = std::min(3, n - 1);
  float const range = c.stop - c.start;
  float u = (range > 0.0f) ? std::min(std::max((time - c.start) / range, 0.0f), 1.0f) : 0.0f;
  u *= static_cast<float>(n - degree);

  // Open uniform knots : degree+1 zeros, 1...(n-degree-1), then degree+1 times (n-degree).
  auto knot = [&](int i) -> float {
    return static_cast<float>((i <= degree) ? 0 : ((i >= n) ? (n - degree) : (i - degree)));
  };

  int span = n - 1;
  if (u < static_cast<float>(n - degree)) {
    span = degree;
    while ((span < n - 1) && (knot(span + 1) <= u)) {
      ++span;
    }
  }

  // Cox - de Boor.
  float basis[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
  float left[4] = {}, right[4] = {};
  for (int j = 1; j <= degree; ++j) {
    left[j] = u - knot(span + 1 - j);
    right[j] = knot(span + j) - u;
    float saved = 0.0f;
    for (int r = 0; r < j; ++r) {
      float const den = right[r + 1] + left[j - r];
      float const t = (den != 0.0f) ? basis[r] / den : 0.0f;
      basis[r] = saved + right[r + 1] * t;
      saved = left[j - r] * t;
    }
    basis[j] = saved;
  }

  std::fill_n(out, nc, 0.0f);
  for (int j = 0; j <= degree; ++j) {
    float const* cp = &c.control[static_cast<size_t>(span - degree + j) * nc];
    for (size_t k = 0; k < nc; ++k) {
      out[k] += basis[j] * cp[k];
    }
  }
}

} // namespace detail

// Evaluates a channel at a given time (in seconds),
// returns false when the channel is not animated.
inline bool Sample(Channel const& c, float time, float *out) {
  switch (c.kind) {
    case Channel::Kind::kConstant:
      std::copy_n(c.constant.begin(), c.numComps, out);
    break;

    case Channel::Kind::kKeys:
      if (c.times.empty()) return false;
      detail::EvalKeys(c, time, out);
    break;

    case Channel::Kind::kSpline:
      if (c.control.empty()) return false;
      detail::EvalSpline(c, time, out);
    break;

    case Channel::Kind::kNone:
    default:
    return false;
  }
  if (c.IsRotation()) {
    detail::Normalize4(out);
  }
  return true;
}

// ----------------------------------------------------------------------------
//  Loading.
// ----------------------------------------------------------------------------

namespace detail {

// Bounds checked reader (little endian).
class Reader {
 public:
  Reader(uint8_t const* data, size_t size) : data_(data), size_(size) {}

  bool ok() const { return !failed_; }
  size_t pos() const { return pos_; }
  size_t remaining() const { return size_ - pos_; }

  uint8_t u8()    { uint8_t v = 0;  read(&v, sizeof(v)); return v; }
  uint16_t u16()  { uint16_t v = 0; read(&v, sizeof(v)); return v; }
  uint32_t u32()  { uint32_t v = 0; read(&v, sizeof(v)); return v; }
  int16_t s16()   { int16_t v = 0;  read(&v, sizeof(v)); return v; }
  float f32()     { float v = 0.0f; read(&v, sizeof(v)); return v; }

  std::string str(size_t length) {
    std::string s;
    if (!failed_ && (length <= remaining())) {
      s.assign(reinterpret_cast<char const*>(data_ + pos_), length);
      pos_ += length;
    } else {
      failed_ = true;
    }
    return s;
  }

  // Marks the reader as failed if there is not at least `count` elements of
  // `elementSize` bytes remaining.
  bool require(size_t count, size_t elementSize) {
    if (!failed_ && (elementSize > 0u) && (count > remaining() / elementSize)) {
      failed_ = true;
    }
    return !failed_;
  }

 private:
  void read(void *dst, size_t n) {
    if (failed_ || (n > remaining())) {
      failed_ = true;
      std::memset(dst, 0, n);
      return;
    }
    std::memcpy(dst, data_ + pos_, n);
    pos_ += n;
  }

  uint8_t const* data_;
  size_t size_;
  size_t pos_ = 0;
  bool failed_ = false;
};

struct Block {
  std::string type;
  uint8_t const* data = nullptr;
  size_t size = 0;
};

struct SplineData {
  std::vector<float> floats;
  std::vector<int16_t> shorts;
};

// Euler (XYZ) angles to quaternion, as R = Rx * Ry * Rz.
inline void EulerToQuat(float x, float y, float z, float *q) {
  float const qx[4] = { std::cos(x * 0.5f), std::sin(x * 0.5f), 0.0f, 0.0f };
  float const qy[4] = { std::cos(y * 0.5f), 0.0f, std::sin(y * 0.5f), 0.0f };
  float const qz[4] = { std::cos(z * 0.5f), 0.0f, 0.0f, std::sin(z * 0.5f) };
  float tmp[4];
  QuatMul(qx, qy, tmp);
  QuatMul(tmp, qz, q);
  Normalize4(q);
}

class Loader {
 public:
  bool Load(std::string const& filename, File *file, std::string *error) {
    error_ = error;

    std::ifstream in(filename, std::ios::binary);
    if (!in) {
      return Fail("cannot open the file");
    }
    bytes_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());

    if (!ReadHeader()) {
      return false;
    }
    for (size_t i = 0; i < blocks_.size(); ++i) {
      if (blocks_[i].type == "NiSequenceData") {
        Sequence seq;
        if (!ReadSequence(i, &seq)) {
          return false;
        }
        file->sequences.push_back(std::move(seq));
      }
    }
    if (file->sequences.empty()) {
      return Fail("no NiSequenceData found");
    }
    return true;
  }

 private:
  bool Fail(std::string const& message) {
    if (error_) {
      *error_ = message;
    }
    return false;
  }

  std::string GetString(uint32_t index) const {
    return (index < strings_.size()) ? strings_[index] : std::string();
  }

  bool ReadHeader() {
    static char const kMagic[] = "Gamebryo File Format, Version ";
    size_t const magicLength = sizeof(kMagic) - 1u;

    auto const eol = std::find(bytes_.begin(), bytes_.end(), static_cast<uint8_t>('\n'));
    if ((bytes_.size() < magicLength) || (std::memcmp(bytes_.data(), kMagic, magicLength) != 0) ||
        (eol == bytes_.end())) {
      return Fail("not a Gamebryo file");
    }
    size_t const offset = static_cast<size_t>(std::distance(bytes_.begin(), eol)) + 1u;
    std::string const version(reinterpret_cast<char const*>(bytes_.data()) + magicLength,
                              offset - 1u - magicLength);
    if (version != "30.1.0.3") {
      return Fail("unsupported version " + version + " (only 30.1.0.3 is)");
    }

    Reader r(bytes_.data() + offset, bytes_.size() - offset);
    r.u32(); // version
    if (r.u8() != 1u) {
      return Fail("big endian files are not supported");
    }
    r.u32(); // user version
    uint32_t const numBlocks = r.u32();
    r.u32(); // user version 2

    uint16_t const numTypes = r.u16();
    std::vector<std::string> types;
    for (uint16_t i = 0; r.ok() && (i < numTypes); ++i) {
      types.push_back(r.str(r.u32()));
    }

    if (!r.require(numBlocks, sizeof(uint16_t))) {
      return Fail("truncated header");
    }
    std::vector<uint16_t> typeIndices(numBlocks);
    for (auto &v : typeIndices) v = r.u16();

    if (!r.require(numBlocks, sizeof(uint32_t))) {
      return Fail("truncated header");
    }
    std::vector<uint32_t> sizes(numBlocks);
    for (auto &v : sizes) v = r.u32();

    uint32_t const numStrings = r.u32();
    r.u32(); // max string length
    for (uint32_t i = 0; r.ok() && (i < numStrings); ++i) {
      strings_.push_back(r.str(r.u32()));
    }

    uint32_t const numGroups = r.u32();
    if (!r.require(numGroups, sizeof(uint32_t))) {
      return Fail("truncated header");
    }
    for (uint32_t i = 0; i < numGroups; ++i) {
      r.u32();
    }
    if (!r.ok()) {
      return Fail("truncated header");
    }

    size_t pos = offset + r.pos();
    for (uint32_t i = 0; i < numBlocks; ++i) {
      if ((typeIndices[i] >= types.size()) || (sizes[i] > bytes_.size() - pos)) {
        return Fail("invalid block " + std::to_string(i));
      }
      blocks_.push_back({ types[typeIndices[i]], bytes_.data() + pos, sizes[i] });
      pos += sizes[i];
    }
    return true;
  }

  Block const* GetBlock(uint32_t link, char const* type) const {
    if ((link >= blocks_.size()) || (blocks_[link].type != type)) {
      return nullptr;
    }
    return &blocks_[link];
  }

  // -- Splines.

  SplineData const* GetSplineData(uint32_t link) {
    if (auto it = splineData_.find(link); it != splineData_.end()) {
      return &it->second;
    }
    Block const* b = GetBlock(link, "NiBSplineData");
    if (!b) {
      return nullptr;
    }
    Reader r(b->data, b->size);
    SplineData &d = splineData_[link];
    uint32_t const numFloats = r.u32();
    if (r.require(numFloats, sizeof(float))) {
      d.floats.resize(numFloats);
      for (auto &v : d.floats) v = r.f32();
    }
    uint32_t const numShorts = r.u32();
    if (r.require(numShorts, sizeof(int16_t))) {
      d.shorts.resize(numShorts);
      for (auto &v : d.shorts) v = r.s16();
    }
    return &d;
  }

  uint32_t GetNumControlPoints(uint32_t link) const {
    Block const* b = GetBlock(link, "NiBSplineBasisData");
    if (!b || (b->size < sizeof(uint32_t))) {
      return 0u;
    }
    uint32_t n;
    std::memcpy(&n, b->data, sizeof(n));
    return n;
  }

  // -- Transform data (keyframes).

  // Reads a sequence of keys. `numComps` is 4 for rotations, 3 for translations
  // and 1 for scales (and euler angles).
  bool ReadKeys(Reader &r, int numComps, bool isRotation, Channel *c) {
    c->numComps = numComps;
    uint32_t const numKeys = r.u32();
    if (numKeys == 0u) {
      return r.ok();
    }
    return ReadKeysBody(r, numKeys, r.u32(), numComps, isRotation, c);
  }

  // Reads the keys of a sequence once its length and type are known.
  bool ReadKeysBody(Reader &r, uint32_t numKeys, uint32_t keyType, int numComps,
                    bool isRotation, Channel *c) {
    c->numComps = numComps;
    c->keyType = keyType;

    size_t const nc = static_cast<size_t>(numComps);
    bool const hasTangents = !isRotation && (c->keyType == kKeyQuadratic);
    bool const hasTBC = (c->keyType == kKeyTBC);
    size_t const keySize = sizeof(float) * (1u + nc + (hasTangents ? 2u * nc : 0u) + (hasTBC ? 3u : 0u));
    if (!r.require(numKeys, keySize)) {
      return false;
    }

    c->times.resize(numKeys);
    c->values.resize(numKeys * nc);
    if (hasTangents) {
      c->tangents.resize(numKeys * nc * 2u);
    }
    for (uint32_t i = 0; i < numKeys; ++i) {
      c->times[i] = r.f32();
      for (size_t k = 0; k < nc; ++k) {
        c->values[i * nc + k] = r.f32();
      }
      if (hasTangents) {
        for (size_t k = 0; k < 2u * nc; ++k) {
          c->tangents[i * 2u * nc + k] = r.f32();
        }
      }
      if (hasTBC) {
        r.f32(); r.f32(); r.f32();
      }
    }
    c->kind = Channel::Kind::kKeys;
    return r.ok();
  }

  // Euler angles keys (3 separate sequences) are converted to quaternion keys.
  bool ReadEulerKeys(Reader &r, Channel *rotation) {
    Channel axis[3];
    for (auto &a : axis) {
      if (!ReadKeys(r, 1, false, &a)) {
        return false;
      }
    }

    std::vector<float> times;
    for (auto const& a : axis) {
      times.insert(times.end(), a.times.begin(), a.times.end());
    }
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());

    rotation->numComps = 4;
    rotation->keyType = kKeyLinear;
    if (times.empty()) {
      return true;
    }
    rotation->kind = Channel::Kind::kKeys;
    rotation->times = times;
    rotation->values.resize(times.size() * 4u);
    for (size_t i = 0; i < times.size(); ++i) {
      float angles[4] = { 0.0f, 0.0f, 0.0f, 0.0f }; // (4 : Sample may normalize 4 floats)
      for (int k = 0; k < 3; ++k) {
        Sample(axis[k], times[i], &angles[k]);
      }
      EulerToQuat(angles[0], angles[1], angles[2], &rotation->values[i * 4u]);
    }
    return true;
  }

  bool ReadTransformData(uint32_t link, Channel *t, Channel *rot, Channel *s) {
    Block const* b = GetBlock(link, "NiTransformData");
    if (!b) {
      return false;
    }
    Reader r(b->data, b->size);

    // Rotations (euler angles are stored as 3 separate float key sequences).
    rot->numComps = 4;
    if (uint32_t const numRotKeys = r.u32(); numRotKeys > 0u) {
      uint32_t const keyType = r.u32();
      bool const ok = (keyType == kKeyXYZRot) ? ReadEulerKeys(r, rot)
                                              : ReadKeysBody(r, numRotKeys, keyType, 4, true, rot);
      if (!ok) return false;
    }

    // Translations & scales.
    if (!ReadKeys(r, 3, false, t)) return false;
    if (!ReadKeys(r, 1, false, s)) return false;
    return r.ok();
  }

  // -- Evaluators.

  struct EvaluatorHeader {
    std::string target;
    std::string controllerType;
  };

  bool ReadPose(Reader &r, Evaluator *e) {
    float pose[8];
    for (float &v : pose) v = r.f32();

    if (IsValidFloat(pose[0])) {
      e->translation.kind = Channel::Kind::kConstant;
      e->translation.numComps = 3;
      std::copy_n(pose, 3, e->translation.constant.begin());
    }
    if (IsValidFloat(pose[3])) {
      e->rotation.kind = Channel::Kind::kConstant;
      e->rotation.numComps = 4;
      std::copy_n(pose + 3, 4, e->rotation.constant.begin());
    }
    if (IsValidFloat(pose[7])) {
      e->scale.kind = Channel::Kind::kConstant;
      e->scale.numComps = 1;
      e->scale.constant[0] = pose[7];
    }
    return r.ok();
  }

  void SetupChannels(Evaluator *e) {
    e->translation.numComps = 3;
    e->rotation.numComps = 4;
    e->scale.numComps = 1;
  }

  // Decodes the control points of a spline channel.
  bool ReadSplineChannel(SplineData const& data, uint32_t numControlPoints, uint32_t handle,
                         bool compressed, float offset, float halfRange,
                         float start, float stop, Channel *c) {
    if (handle >= kInvalidHandle) {
      return true; // (channel not animated by the spline)
    }
    size_t const nc = static_cast<size_t>(c->numComps);
    size_t const count = static_cast<size_t>(numControlPoints) * nc;
    if (compressed) {
      if (count > data.shorts.size() || handle > data.shorts.size() - count) {
        return false;
      }
    } else if (count > data.floats.size() || handle > data.floats.size() - count) {
      return false;
    }

    c->kind = Channel::Kind::kSpline;
    c->start = start;
    c->stop = stop;
    c->numControlPoints = numControlPoints;
    c->control.resize(count);
    for (size_t i = 0; i < count; ++i) {
      c->control[i] = compressed
        ? offset + static_cast<float>(data.shorts[handle + i]) * (halfRange / 32767.0f)
        : data.floats[handle + i];
    }
    return true;
  }

  // Returns false on a malformed block, `supported` is false for unknown ones.
  bool ReadEvaluator(uint32_t link, Evaluator *e, bool *supported) {
    *supported = true;
    Block const& b = blocks_[link];
    bool const isTransform = (b.type == "NiTransformEvaluator");
    bool const isConst = (b.type == "NiConstTransformEvaluator");
    bool const isSplineComp = (b.type == "NiBSplineCompTransformEvaluator");
    bool const isSpline = (b.type == "NiBSplineTransformEvaluator");
    if (!isTransform && !isConst && !isSplineComp && !isSpline) {
      *supported = false;
      return true;
    }

    Reader r(b.data, b.size);
    SetupChannels(e);

    e->target = GetString(r.u32());
    r.u32(); // property type
    r.u32(); // controller type
    r.u32(); // controller id
    r.u32(); // interpolator id
    r.u32(); // channel types (4 x u8)

    if (isTransform || isConst) {
      if (!ReadPose(r, e)) return false;
      if (isTransform) {
        uint32_t const dataLink = r.u32();
        if (dataLink != kNullLink) {
          Channel t, rot, s;
          if (!ReadTransformData(dataLink, &t, &rot, &s)) return false;

          // A channel with a single key is constant.
          auto apply = [](Channel *dst, Channel &src) {
            if (src.kind != Channel::Kind::kKeys) return;
            if (src.times.size() == 1u) {
              dst->kind = Channel::Kind::kConstant;
              std::copy_n(src.values.begin(), dst->numComps, dst->constant.begin());
            } else {
              *dst = std::move(src);
            }
          };
          apply(&e->translation, t);
          apply(&e->rotation, rot);
          apply(&e->scale, s);
        }
      }
      return r.ok();
    }

    // B-splines.
    float const start = r.f32();
    float const stop = r.f32();
    uint32_t const dataLink = r.u32();
    uint32_t const basisLink = r.u32();
    if (!ReadPose(r, e)) return false;
    uint32_t const handles[3] = { r.u32(), r.u32(), r.u32() };
    float range[6] = { 0, 0, 0, 0, 0, 0 };
    if (isSplineComp) {
      for (float &v : range) v = r.f32();
    }
    if (!r.ok()) return false;

    SplineData const* data = GetSplineData(dataLink);
    uint32_t const numControlPoints = GetNumControlPoints(basisLink);
    if (!data || (numControlPoints == 0u)) {
      return false;
    }
    Channel* const channels[3] = { &e->translation, &e->rotation, &e->scale };
    for (int i = 0; i < 3; ++i) {
      if (!ReadSplineChannel(*data, numControlPoints, handles[i], isSplineComp,
                             range[i * 2], range[i * 2 + 1], start, stop, channels[i])) {
        return false;
      }
    }
    return true;
  }

  // -- Sequence.

  bool ReadSequence(size_t index, Sequence *seq) {
    Block const& b = blocks_[index];
    std::string const where = "NiSequenceData block " + std::to_string(index) + " : ";
    Reader r(b.data, b.size);

    seq->name = GetString(r.u32());

    uint32_t const numEvaluators = r.u32();
    if (!r.require(numEvaluators, sizeof(uint32_t))) {
      return Fail(where + "truncated block");
    }
    std::vector<uint32_t> links(numEvaluators);
    for (auto &l : links) l = r.u32();

    uint32_t const textKeysLink = r.u32();
    seq->duration = r.f32();
    seq->cycle = static_cast<CycleType>(r.u32());
    seq->frequency = r.f32();
    seq->accumRoot = GetString(r.u32());
    seq->accumFlags = r.u32();
    if (!r.ok()) {
      return Fail(where + "truncated block");
    }

    if (Block const* tk = GetBlock(textKeysLink, "NiTextKeyExtraData"); tk) {
      Reader tr(tk->data, tk->size);
      tr.u32(); // name
      uint32_t const numKeys = tr.u32();
      if (tr.require(numKeys, 2u * sizeof(uint32_t))) {
        for (uint32_t i = 0; i < numKeys; ++i) {
          TextKey key;
          key.time = tr.f32();
          key.text = GetString(tr.u32());
          seq->textKeys.push_back(std::move(key));
        }
      }
    }

    for (uint32_t link : links) {
      if (link >= blocks_.size()) {
        return Fail(where + "invalid evaluator link");
      }
      Evaluator e;
      bool supported = true;
      if (!ReadEvaluator(link, &e, &supported)) {
        return Fail(where + "malformed " + blocks_[link].type + " block " + std::to_string(link));
      }
      if (supported) {
        seq->evaluators.push_back(std::move(e));
      } else {
        seq->unsupported.push_back(blocks_[link].type);
      }
    }
    return true;
  }

  std::vector<uint8_t> bytes_{};
  std::vector<std::string> strings_{};
  std::vector<Block> blocks_{};
  std::unordered_map<uint32_t, SplineData> splineData_{};
  std::string *error_{nullptr};
};

} // namespace detail

// Loads every sequence of a KF file. On failure returns false and fills `error`.
inline bool Load(std::string const& filename, File *file, std::string *error = nullptr) {
  detail::Loader loader;
  return loader.Load(filename, file, error);
}

} // namespace kf

#endif // DOJIMA_KF_H_
