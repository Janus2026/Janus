#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace janus {
namespace layer {
class LmHead;
class WordEmbedding;
#if defined(USE_NPU)
class NpuLmHead;
class NpuWordEmbedding;
#endif
}  // namespace layer

namespace detail {
template <typename T, typename = void>
struct has_get_lm_head : std::false_type {};

template <typename T>
struct has_get_lm_head<T,
                       std::void_t<decltype(std::declval<T>()->get_lm_head())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_lm_head : std::false_type {};

template <typename T>
struct has_set_lm_head<T,
                       std::void_t<decltype(std::declval<T>()->set_lm_head(
                           std::declval<layer::LmHead&>()))>> : std::true_type {
};

template <typename T, typename = void>
struct has_get_word_embedding : std::false_type {};

template <typename T>
struct has_get_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->get_word_embedding())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_word_embedding : std::false_type {};

template <typename T>
struct has_set_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->set_word_embedding(
        std::declval<layer::WordEmbedding&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_lazy_load_model : std::false_type {};

template <typename T>
struct has_lazy_load_model<
    T,
    std::void_t<decltype(std::declval<T>()->lazy_load_model(
        std::declval<std::unique_ptr<ModelLoader>>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_free_model_weights : std::false_type {};

template <typename T>
struct has_free_model_weights<
    T,
    std::void_t<decltype(std::declval<T>()->free_model_weights())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_reload_model_weights : std::false_type {};

template <typename T>
struct has_reload_model_weights<
    T,
    std::void_t<decltype(std::declval<T>()->reload_model_weights())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_reload_model_weights_from_device : std::false_type {};

template <typename T>
struct has_reload_model_weights_from_device<
    T,
    std::void_t<
        decltype(std::declval<T>()->reload_model_weights_from_device())>>
    : std::true_type {};

#if defined(USE_NPU)
template <typename T, typename = void>
struct has_get_npu_lm_head : std::false_type {};

template <typename T>
struct has_get_npu_lm_head<
    T,
    std::void_t<decltype(std::declval<T>()->get_npu_lm_head())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_npu_lm_head : std::false_type {};

template <typename T>
struct has_set_npu_lm_head<
    T,
    std::void_t<decltype(std::declval<T>()->set_npu_lm_head(
        std::declval<layer::NpuLmHead&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_get_npu_word_embedding : std::false_type {};

template <typename T>
struct has_get_npu_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->get_npu_word_embedding())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_npu_word_embedding : std::false_type {};

template <typename T>
struct has_set_npu_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->set_npu_word_embedding(
        std::declval<layer::NpuWordEmbedding&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_init_or_refresh_rolling_runtime : std::false_type {};

template <typename T>
struct has_init_or_refresh_rolling_runtime<
    T,
    std::void_t<decltype(std::declval<T>()->init_or_refresh_rolling_runtime(
        std::declval<::janus::Stream*>(),
        std::declval<::janus::Stream*>(),
        std::declval<int32_t>(),
        std::declval<int32_t>(),
        std::declval<const std::string&>()))>> : std::true_type {};

#endif
}  // namespace detail
}  // namespace janus
