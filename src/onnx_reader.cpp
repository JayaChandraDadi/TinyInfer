#include "tinyinfer/onnx_reader.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tinyinfer {

std::string shape_str(const Shape& s) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < s.size(); ++i) os << (i ? ", " : "") << s[i];
    os << "]";
    return os.str();
}

namespace {

// ---------------------------------------------------------------------------
// Protobuf wire-format cursor
// ---------------------------------------------------------------------------
enum WireType : uint32_t { kVarint = 0, kFixed64 = 1, kLen = 2, kFixed32 = 5 };

class Reader {
public:
    Reader(const char* p, size_t n) : p_(reinterpret_cast<const uint8_t*>(p)), end_(p_ + n) {}

    bool done() const { return p_ >= end_; }

    uint64_t varint() {
        uint64_t v = 0;
        int shift = 0;
        while (true) {
            if (p_ >= end_) throw std::runtime_error("onnx: truncated varint");
            uint8_t b = *p_++;
            v |= static_cast<uint64_t>(b & 0x7F) << shift;
            if (!(b & 0x80)) break;
            shift += 7;
            if (shift > 63) throw std::runtime_error("onnx: varint too long");
        }
        return v;
    }

    void tag(uint32_t& field, uint32_t& wire) {
        uint64_t t = varint();
        field = static_cast<uint32_t>(t >> 3);
        wire = static_cast<uint32_t>(t & 7);
    }

    Reader sub() {
        uint64_t n = varint();
        check(n);
        Reader r(reinterpret_cast<const char*>(p_), n);
        p_ += n;
        return r;
    }

    std::string bytes() {
        uint64_t n = varint();
        check(n);
        std::string s(reinterpret_cast<const char*>(p_), n);
        p_ += n;
        return s;
    }

    float fixed32_float() {
        check(4);
        float f;
        std::memcpy(&f, p_, 4);
        p_ += 4;
        return f;
    }

    void skip(uint32_t wire) {
        switch (wire) {
            case kVarint: varint(); break;
            case kFixed64: check(8); p_ += 8; break;
            case kLen: { uint64_t n = varint(); check(n); p_ += n; break; }
            case kFixed32: check(4); p_ += 4; break;
            default: throw std::runtime_error("onnx: unsupported wire type " + std::to_string(wire));
        }
    }

private:
    void check(uint64_t n) const {
        if (static_cast<uint64_t>(end_ - p_) < n) throw std::runtime_error("onnx: truncated field");
    }
    const uint8_t* p_;
    const uint8_t* end_;
};

// Repeated int64 fields may be packed (wire type 2) or not (wire type 0).
void read_int64s(Reader& r, uint32_t wire, std::vector<int64_t>& out) {
    if (wire == kLen) {
        Reader s = r.sub();
        while (!s.done()) out.push_back(static_cast<int64_t>(s.varint()));
    } else {
        out.push_back(static_cast<int64_t>(r.varint()));
    }
}

void read_floats(Reader& r, uint32_t wire, std::vector<float>& out) {
    if (wire == kLen) {
        Reader s = r.sub();
        while (!s.done()) out.push_back(s.fixed32_float());
    } else {
        out.push_back(r.fixed32_float());
    }
}

// ONNX TensorProto.DataType values we care about.
enum DType { kFloat = 1, kInt32 = 6, kInt64 = 7, kDouble = 11 };

Tensor parse_tensor(Reader r, std::string* name_out) {
    Shape dims;
    int64_t dtype = 0;
    std::vector<float> fdata;
    std::vector<int64_t> idata;
    std::string raw;
    std::string name;

    while (!r.done()) {
        uint32_t f, w;
        r.tag(f, w);
        switch (f) {
            case 1: read_int64s(r, w, dims); break;
            case 2: dtype = static_cast<int64_t>(r.varint()); break;
            case 4: read_floats(r, w, fdata); break;
            case 5: read_int64s(r, w, idata); break;   // int32_data (varint-encoded)
            case 7: read_int64s(r, w, idata); break;   // int64_data
            case 8: name = r.bytes(); break;
            case 9: raw = r.bytes(); break;
            case 14: {
                int64_t loc = static_cast<int64_t>(r.varint());
                if (loc == 1) throw std::runtime_error("onnx: external tensor data is not supported");
                break;
            }
            default: r.skip(w);
        }
    }
    if (name_out) *name_out = name;

    const int64_t n = numel(dims);
    if (dtype == kFloat) {
        if (!raw.empty()) {
            if (static_cast<int64_t>(raw.size()) != n * 4) throw std::runtime_error("onnx: bad raw float size for " + name);
            fdata.resize(static_cast<size_t>(n));
            std::memcpy(fdata.data(), raw.data(), raw.size());
        }
        return Tensor(dims, std::move(fdata));
    }
    if (dtype == kDouble) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            double d;
            std::memcpy(&d, raw.data() + i * 8, 8);
            v[static_cast<size_t>(i)] = static_cast<float>(d);
        }
        return Tensor(dims, std::move(v));
    }
    if (dtype == kInt64 || dtype == kInt32) {
        if (!raw.empty()) {
            const size_t width = dtype == kInt64 ? 8 : 4;
            idata.resize(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                if (width == 8) {
                    int64_t v;
                    std::memcpy(&v, raw.data() + i * 8, 8);
                    idata[static_cast<size_t>(i)] = v;
                } else {
                    int32_t v;
                    std::memcpy(&v, raw.data() + i * 4, 4);
                    idata[static_cast<size_t>(i)] = v;
                }
            }
        }
        return Tensor::ints(dims, std::move(idata));
    }
    throw std::runtime_error("onnx: unsupported tensor dtype " + std::to_string(dtype) + " for " + name);
}

Attribute parse_attribute(Reader r, std::string& name) {
    Attribute a;
    while (!r.done()) {
        uint32_t f, w;
        r.tag(f, w);
        switch (f) {
            case 1: name = r.bytes(); break;
            case 2: a.f = r.fixed32_float(); a.kind = Attribute::Kind::Float; break;
            case 3: a.i = static_cast<int64_t>(r.varint()); a.kind = Attribute::Kind::Int; break;
            case 4: a.s = r.bytes(); a.kind = Attribute::Kind::String; break;
            case 5: a.t = parse_tensor(r.sub(), nullptr); a.kind = Attribute::Kind::Tensor; break;
            case 7: read_floats(r, w, a.floats); a.kind = Attribute::Kind::Floats; break;
            case 8: read_int64s(r, w, a.ints); a.kind = Attribute::Kind::Ints; break;
            default: r.skip(w);
        }
    }
    return a;
}

Node parse_node(Reader r) {
    Node n;
    while (!r.done()) {
        uint32_t f, w;
        r.tag(f, w);
        switch (f) {
            case 1: n.inputs.push_back(r.bytes()); break;
            case 2: n.outputs.push_back(r.bytes()); break;
            case 3: n.name = r.bytes(); break;
            case 4: n.op_type = r.bytes(); break;
            case 5: {
                std::string an;
                Attribute a = parse_attribute(r.sub(), an);
                n.attrs[an] = std::move(a);
                break;
            }
            default: r.skip(w);
        }
    }
    return n;
}

Shape parse_shape(Reader r) {  // TensorShapeProto
    Shape s;
    while (!r.done()) {
        uint32_t f, w;
        r.tag(f, w);
        if (f == 1) {  // Dimension
            Reader d = r.sub();
            int64_t v = -1;
            while (!d.done()) {
                uint32_t df, dw;
                d.tag(df, dw);
                if (df == 1) v = static_cast<int64_t>(d.varint());
                else d.skip(dw);
            }
            s.push_back(v);
        } else {
            r.skip(w);
        }
    }
    return s;
}

ValueInfo parse_value_info(Reader r) {
    ValueInfo vi;
    while (!r.done()) {
        uint32_t f, w;
        r.tag(f, w);
        if (f == 1) {
            vi.name = r.bytes();
        } else if (f == 2) {  // TypeProto
            Reader t = r.sub();
            while (!t.done()) {
                uint32_t tf, tw;
                t.tag(tf, tw);
                if (tf == 1) {  // tensor_type
                    Reader tt = t.sub();
                    while (!tt.done()) {
                        uint32_t ttf, ttw;
                        tt.tag(ttf, ttw);
                        if (ttf == 2) vi.shape = parse_shape(tt.sub());
                        else tt.skip(ttw);
                    }
                } else {
                    t.skip(tw);
                }
            }
        } else {
            r.skip(w);
        }
    }
    return vi;
}

void parse_graph(Reader r, ModelProto& m) {
    std::vector<ValueInfo> raw_inputs;
    while (!r.done()) {
        uint32_t f, w;
        r.tag(f, w);
        switch (f) {
            case 1: m.nodes.push_back(parse_node(r.sub())); break;
            case 5: {
                std::string name;
                Tensor t = parse_tensor(r.sub(), &name);
                m.initializers[name] = std::move(t);
                break;
            }
            case 11: raw_inputs.push_back(parse_value_info(r.sub())); break;
            case 12: m.outputs.push_back(parse_value_info(r.sub())); break;
            default: r.skip(w);
        }
    }
    // Older exporters list initializers as graph inputs too; only keep real inputs.
    for (auto& vi : raw_inputs)
        if (!m.initializers.count(vi.name)) m.inputs.push_back(std::move(vi));
}

}  // namespace

int64_t Node::get_int(const std::string& k, int64_t def) const {
    auto it = attrs.find(k);
    return it == attrs.end() ? def : it->second.i;
}
float Node::get_float(const std::string& k, float def) const {
    auto it = attrs.find(k);
    return it == attrs.end() ? def : it->second.f;
}
std::string Node::get_string(const std::string& k, const std::string& def) const {
    auto it = attrs.find(k);
    return it == attrs.end() ? def : it->second.s;
}
std::vector<int64_t> Node::get_ints(const std::string& k, std::vector<int64_t> def) const {
    auto it = attrs.find(k);
    return it == attrs.end() ? def : it->second.ints;
}

ModelProto parse_onnx(const std::string& bytes) {
    ModelProto m;
    Reader r(bytes.data(), bytes.size());
    while (!r.done()) {
        uint32_t f, w;
        r.tag(f, w);
        switch (f) {
            case 1: m.ir_version = static_cast<int64_t>(r.varint()); break;
            case 2: m.producer = r.bytes(); break;
            case 7: parse_graph(r.sub(), m); break;
            case 8: {  // OperatorSetIdProto
                Reader o = r.sub();
                std::string domain;
                int64_t version = 0;
                while (!o.done()) {
                    uint32_t of, ow;
                    o.tag(of, ow);
                    if (of == 1) domain = o.bytes();
                    else if (of == 2) version = static_cast<int64_t>(o.varint());
                    else o.skip(ow);
                }
                if (domain.empty() || domain == "ai.onnx") m.opset = version;
                break;
            }
            default: r.skip(w);
        }
    }
    if (m.nodes.empty()) throw std::runtime_error("onnx: model has no graph nodes");
    return m;
}

ModelProto load_onnx(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("onnx: cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse_onnx(ss.str());
}

}  // namespace tinyinfer
