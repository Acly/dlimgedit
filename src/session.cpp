#include "session.hpp"
#include "environment.hpp"
#include "platform.hpp"

#include <dml_provider_factory.h>
#include <onnxruntime_c_api.h>

#include <memory>
#include <optional>
#include <string>

namespace dlimg {
namespace {

void check(OrtStatusPtr res) {
    if (res != nullptr) {
        auto msg = std::string(Ort::GetApi().GetErrorMessage(res));
        Ort::GetApi().ReleaseStatus(res);
        throw Exception(msg);
    }
}

std::pair<Ort::SessionOptions, OrtSessionOptions*> create_session_options() {
    OrtSessionOptions* opts;
    check(Ort::GetApi().CreateSessionOptions(&opts));
    return {Ort::SessionOptions(opts), opts};
}

std::optional<std::unique_lock<std::mutex>> lock_session(std::mutex& m,
                                                         EnvironmentImpl const& env) {
    // Locking is required for Ort::Session::Run() when using DirectML provider.
    // For other providers calling Run() concurrently is safe.
    if (env.backend == Backend::gpu && is_windows) {
        return std::unique_lock<std::mutex>(m);
    }
    return {};
}

} // namespace

Ort::Session create_session(EnvironmentImpl& env, char const* kind, char const* model) {
    auto [opts, opts_raw] = create_session_options();
    opts.SetIntraOpNumThreads(env.thread_count);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (env.backend == Backend::gpu) {
        if (is_windows) {
            // Use DirectML. The following two options are required:
            opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            opts.DisableMemPattern();

            OrtDmlApi* dml_api = nullptr;
            check(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION,
                                                        (void const**)(&dml_api)));
            check(dml_api->SessionOptionsAppendExecutionProvider_DML(opts_raw, 0));
        } else if (is_linux) {
            // Use CUDA.
            opts.AppendExecutionProvider_CUDA({});
        }
    }
    Path model_path = env.model_directory / kind / model;
    if (!exists(model_path)) {
        throw Exception(std::format("Could not find model '{}/{}' in directory '{}'.", kind, model,
                                    env.model_directory.string()));
    }
    return Ort::Session(env.onnx_env, model_path.c_str(), opts);
}

Session::Session(EnvironmentImpl& env, char const* model_kind, char const* model_name,
                 std::span<char const* const> input_names,
                 std::span<char const* const> output_names)
    : env_(env),
      session_(create_session(env, model_kind, model_name)),
      input_names_(input_names),
      output_names_(output_names) {
    ASSERT(input_names.size() == session_.GetInputCount());
    ASSERT(output_names.size() == session_.GetOutputCount());
}

Shape Session::input_shape(int index) const {
    return Shape(session_.GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape());
}

Shape Session::output_shape(int index) const {
    return Shape(session_.GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape());
}

void Session::run(std::span<Ort::Value const> inputs, std::span<Ort::Value> outputs) {
    ASSERT(inputs.size() == input_names_.size());
    ASSERT(outputs.size() == output_names_.size());

    auto lock = lock_session(mutex_, env_);
    Ort::RunOptions opts{nullptr};
    session_.Run(opts, input_names_.data(), inputs.data(), inputs.size(), output_names_.data(),
                 outputs.data(), outputs.size());
}

std::vector<Ort::Value> Session::run(std::span<Ort::Value const> inputs) {
    ASSERT(inputs.size() == input_names_.size());

    auto lock = lock_session(mutex_, env_);
    Ort::RunOptions opts{nullptr};
    return session_.Run(opts, input_names_.data(), inputs.data(), inputs.size(),
                        output_names_.data(), output_names_.size());
}

Ort::MemoryInfo const& get_memory_info(EnvironmentImpl const& env) { return env.memory_info; }

} // namespace dlimg
