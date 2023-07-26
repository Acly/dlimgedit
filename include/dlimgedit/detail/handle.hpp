#pragma once

#include <dlimgedit/detail/dlimgedit.h>

#include <cstring>
#include <type_traits>

namespace dlimg {
namespace detail {

template <class To, class From>
std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                     std::is_trivially_copyable<To>::value,
                 To> inline bit_cast(From const& src) noexcept {
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

template <typename T> struct Global {
    static dlimg_Api const* api_;
};
template <typename T> dlimg_Api const* Global<T>::api_{};

} // namespace detail

inline dlimg_Api const& api() {
#ifndef DLIMGEDIT_LOAD_DYNAMIC
    if (!detail::Global<void>::api_) {
        detail::Global<void>::api_ = dlimg_init();
    }
#endif
    return *detail::Global<void>::api_;
}

namespace detail {
inline void destroy(dlimg_Environment h) { api().destroy_environment(h); }
inline void destroy(dlimg_Segmentation h) { api().destroy_segmentation(h); }
} // namespace detail

template <typename T> class Handle {
  public:
    Handle() noexcept = default;
    Handle(T* handle) noexcept : m_(handle) {}

    Handle(Handle const&) = delete;
    Handle& operator=(Handle const&) = delete;

    Handle(Handle&& other) noexcept : m_(other.m_) { other.m_ = nullptr; }
    Handle& operator=(Handle&& other) noexcept {
        std::swap(m_, other.m_);
        return *this;
    }

    virtual ~Handle() {
        if (m_) {
            detail::destroy(m_);
        }
    }

    T* handle() const noexcept { return m_; }

  protected:
    T*& emplace() noexcept { return m_; }

  private:
    T* m_ = nullptr;
};

} // namespace dlimg
