#pragma once

#include <mutex>
#include <optional>

namespace dlimg {

template <typename T> class Lazy {
  public:
    template <typename... Args> T& get_or_create(Args&&... args) {
        std::call_once(flag_, [&, this]() { obj_.emplace(std::forward<Args>(args)...); });
        return *obj_;
    }

  private:
    std::once_flag flag_;
    std::optional<T> obj_;
};

} // namespace dlimg
