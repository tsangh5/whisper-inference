#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#ifdef _WIN32
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

struct Tensor {
    const uint16_t*  data  = nullptr;  // points into the mapping, fp16 bits
    std::vector<int> shape;
    size_t           numel = 0;        // product of shape; data[0..numel) is valid
};

class MappedFile {
public:
    explicit MappedFile(const std::string& path);
    ~MappedFile();

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    // TODO: a move ctor is worth having so you can return one by value.

    const uint8_t* data() const { return base_; }
    size_t         size() const { return size_; }

private:
    const uint8_t* base_ = nullptr;
    size_t         size_ = 0;
#ifdef _WIN32
    // TODO: you need to keep BOTH handles to close them in the right order:
    //       UnmapViewOfFile(base_) -> CloseHandle(mapping_) -> CloseHandle(file_)
    void* file_    = nullptr;   // HANDLE from CreateFileA
    void* mapping_ = nullptr;   // HANDLE from CreateFileMappingA
#else
    // POSIX needs no extra state: the fd can be closed right after mmap().
#endif
};

MappedFile::MappedFile(const std::string& path) {
#ifdef _WIN32
    // TODO (Windows), in order:
    //   1. CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, ..., OPEN_EXISTING, ...)
    //   2. GetFileSizeEx  -> size_
    //   3. CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr)
    //   4. MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0) -> base_
    // Throw on any failure; GetLastError() tells you which step died.
#else
    // TODO (POSIX): open(O_RDONLY) -> fstat for size_ -> mmap(nullptr, size_,
    //   PROT_READ, MAP_PRIVATE, fd, 0) -> close(fd). The mapping outlives the fd.
#endif
}

MappedFile::~MappedFile() {
    // TODO: mirror the constructor, in reverse. Guard against partial
    //       construction (base_ may be null if a step above failed).
}

// -------------------------------------------------------------- WeightMap ---

struct Config {                 // the 36-byte header, decoded
    int32_t version   = 0;
    int32_t n_layers  = 0;
    int32_t n_heads   = 0;
    int32_t d_model   = 0;
    int32_t n_mels    = 0;
    int32_t vocab_size= 0;
    int32_t n_tensors = 0;
    int32_t align     = 0;      // payloads start on multiples of this (64)
};

class WeightMap {
public:
    explicit WeightMap(const std::string& path);

    const Config& config() const { return cfg_; }

    // Throws if absent — a missing weight is a bug, not a runtime condition.
    const Tensor& at(const std::string& name) const;

private:
    void parse();               // fills cfg_ and tensors_ from file_

    MappedFile                             file_;
    Config                                 cfg_;
    std::unordered_map<std::string, Tensor> tensors_;
};

WeightMap::WeightMap(const std::string& path) : file_(path) { parse(); }

void WeightMap::parse() {
    const uint8_t* p   = file_.data();
    const uint8_t* end = p + file_.size();

    // A tiny cursor keeps the parse readable and is the natural place to put
    // the "would this read run past `end`?" check exactly once.
    // TODO: write these two helpers.
    //   int32_t take_i32();                       // read 4 LE bytes, advance
    //   const uint8_t* take_bytes(size_t n);      // return p, advance by n
    // Both must refuse to walk past `end` — a truncated file is the most
    // likely corruption and you want it caught here, not as a wild pointer.

    // --- header ---
    // TODO: check the first 4 bytes are 'W','S','P','R' and bail loudly if not.
    //       Then read the 8 int32s in the order weight_conversion.py writes
    //       them: version, n_layers, n_heads, d_model, n_mels, vocab,
    //       n_tensors, align. Reject a version you do not understand (expect 2).

    // --- tensors ---
    // for (int i = 0; i < cfg_.n_tensors; ++i) {
    //     int32_t name_len = take_i32();
    //     std::string name(reinterpret_cast<const char*>(take_bytes(name_len)), name_len);
    //
    //     int32_t ndim = take_i32();
    //     shape: ndim int32s; numel = product of them
    //         TODO: guard against overflow / absurd ndim before trusting numel.
    //
    //     SKIP PADDING before the payload. The writer inserted
    //     (-offset % align) zero bytes so the payload lands on the grid:
    //         size_t off = size_t(p - file_.data());
    //         p += (cfg_.align - off % cfg_.align) % cfg_.align;
    //
    //     const uint8_t* raw = take_bytes(numel * 2);   // 2 bytes per fp16
    //
    //     >>> Now assert (raw - file_.data()) % cfg_.align == 0. It holds for
    //     >>> all 479 tensors. Keep the assert: it is what catches a future
    //     >>> writer change that reintroduces the skew.
    //     t.data = reinterpret_cast<const uint16_t*>(raw);
    //
    //     tensors_.emplace(std::move(name), std::move(t));
    // }

    // TODO: after the loop, assert p == end. Zero trailing bytes is a strong
    //       signal the framing never drifted (the Python reader confirms this).
    (void)p; (void)end;
}

const Tensor& WeightMap::at(const std::string& name) const {
    // TODO: find, throw std::out_of_range with the name in the message if missing.
    static Tensor dummy;
    return dummy;
}

// ------------------------------------------------------------------ usage ---
//
//   WeightMap w("whisper_small.bin");
//   const Tensor& q = w.at("encoder.blocks.0.attn.query.weight");  // (768, 768)
//   // q.data is live for as long as `w` is, and holds fp16 bits.
//
// Turning one of those into a float, when you need it:
//   _cvtsh_ss(q.data[i])                  // x86 F16C intrinsic, <immintrin.h>
//   or a 20-line manual bit unpack if you would rather not depend on F16C.
