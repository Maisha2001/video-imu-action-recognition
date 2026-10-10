#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) throw std::runtime_error("Usage: infer MODEL INPUTS.bin OUTPUT.bin [REPEATS]");
        int repeats = argc == 5 ? std::stoi(argv[4]) : 1;
        if (repeats < 1 || repeats > 1000) throw std::runtime_error("Repeats must be 1..1000");
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "activity");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        std::filesystem::path model_path(argv[1]);
        Ort::Session session(env, model_path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> names;
        std::vector<std::vector<int64_t>> shapes;
        std::vector<size_t> sizes;
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
            auto type = session.GetInputTypeInfo(i);
            auto info = type.GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("Inputs must be float32");
            auto shape = info.GetShape();
            size_t size = 1;
            for (auto dimension : shape) {
                if (dimension < 1 || dimension > 1000000 || size > 10000000 / static_cast<size_t>(dimension))
                    throw std::runtime_error("Unsupported input shape");
                size *= static_cast<size_t>(dimension);
            }
            shapes.push_back(shape);
            sizes.push_back(size);
        }
        if (names.empty() || names.size() > 3 || session.GetOutputCount() != 1) throw std::runtime_error("Unsupported graph contract");
        size_t stride = std::accumulate(sizes.begin(), sizes.end(), size_t{0});
        std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
        auto length = input.tellg();
        if (!input || length < 8 || length > 512 * 1024 * 1024) throw std::runtime_error("Input file must be 8 bytes to 512 MiB");
        input.seekg(0);
        char magic[4];
        uint32_t count;
        input.read(magic, 4);
        input.read(reinterpret_cast<char*>(&count), 4);
        if (std::string(magic, 4) != "VIM1" || count < 1 || count > 4096 || count * repeats > 100000 ||
                uint64_t(length) != 8 + uint64_t(count) * stride * sizeof(float)) throw std::runtime_error("Invalid input header or length");
        std::vector<float> data(count * stride);
        input.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
        if (!input || !std::all_of(data.begin(), data.end(), [](float v){ return std::isfinite(v); })) throw std::runtime_error("Nonfinite or truncated inputs");
        std::vector<const char*> input_names;
        for (const auto& name : names) input_names.push_back(name.c_str());
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        const char* output_names[] = {output_name.get()};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto run = [&](size_t row) {
            std::vector<Ort::Value> tensors;
            size_t offset = row * stride;
            for (size_t i = 0; i < sizes.size(); ++i) {
                if (names[i] == "available") {
                    if (sizes[i] != 2 || (data[offset] != 0 && data[offset] != 1) ||
                            (data[offset + 1] != 0 && data[offset + 1] != 1) || data[offset] + data[offset + 1] < 1)
                        throw std::runtime_error("Invalid sensor mask");
                }
                tensors.emplace_back(Ort::Value::CreateTensor<float>(memory, data.data() + offset, sizes[i], shapes[i].data(), shapes[i].size()));
                offset += sizes[i];
            }
            return session.Run(Ort::RunOptions{nullptr}, input_names.data(), tensors.data(), tensors.size(), output_names, 1);
        };
        for (int i = 0; i < 5; ++i) run(0);
        std::ofstream output(argv[3], std::ios::binary);
        if (!output) throw std::runtime_error("Cannot open output file");
        std::vector<double> elapsed;
        size_t output_size = 0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (size_t row = 0; row < count; ++row) {
                auto start = std::chrono::steady_clock::now();
                auto results = run(row);
                auto end = std::chrono::steady_clock::now();
                elapsed.push_back(std::chrono::duration<double, std::milli>(end - start).count());
                output_size = results[0].GetTensorTypeAndShapeInfo().GetElementCount();
                auto values = results[0].GetTensorData<float>();
                if (!std::all_of(values, values + output_size, [](float v){ return std::isfinite(v); })) throw std::runtime_error("Nonfinite model output");
                if (repeat == 0) output.write(reinterpret_cast<const char*>(values), output_size * sizeof(float));
            }
        }
        if (!output) throw std::runtime_error("Output write failed");
        std::sort(elapsed.begin(), elapsed.end());
        auto percentile = [&](double q){ return elapsed[static_cast<size_t>(std::ceil(q * elapsed.size())) - 1]; };
        std::cout << "{\"records\":" << count << ",\"output_values_per_record\":" << output_size
                  << ",\"repeats\":" << repeats << ",\"warmup\":5,\"threads\":1,\"samples\":" << elapsed.size()
                  << ",\"p50_ms\":" << percentile(.5) << ",\"p95_ms\":" << percentile(.95)
                  << ",\"mean_ms\":" << std::accumulate(elapsed.begin(), elapsed.end(), 0.) / elapsed.size() << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
