#pragma once

#include <dlimgedit/dlimgedit.hpp>

#include <cstdio>
#include <format>
#include <string_view>

namespace dlimg {

class AssertionFailure : public Exception {
  public:
    AssertionFailure(std::string_view msg) : Exception(msg.data()) {}
};

inline void assertion_failure(char const* file, int line, char const* expr) {
    auto msg = std::format("Assertion failed at {}:{}: {}", file, line, expr);
    fprintf(stderr, "%s\n", msg.c_str());
    throw AssertionFailure(msg);
}

} // namespace dlimg

#define ASSERT(cond)                                                                               \
    if (!(cond)) {                                                                                 \
        assertion_failure(__FILE__, __LINE__, #cond);                                              \
    }
