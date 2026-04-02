#pragma once

#include <torch/torch.h>

#include <nlohmann/json.hpp>
#include <optional>

#include "framework/request/dit_request_state.h"

namespace janus {

// dit related forward input params
struct DiTForwardInput {
  DiTForwardInput to(const torch::Device& device,
                     torch::ScalarType dtype) const {
    DiTForwardInput input = *this;

    if (prompt_embeds.defined()) {
      input.prompt_embeds = prompt_embeds.to(device, dtype);
    }

    if (pooled_prompt_embeds.defined()) {
      input.pooled_prompt_embeds = pooled_prompt_embeds.to(device, dtype);
    }

    if (negative_prompt_embeds.defined()) {
      input.negative_prompt_embeds = negative_prompt_embeds.to(device, dtype);
    }

    if (negative_pooled_prompt_embeds.defined()) {
      input.negative_pooled_prompt_embeds =
          negative_pooled_prompt_embeds.to(device, dtype);
    }

    if (latents.defined()) {
      input.latents = latents.to(device, dtype);
    }

    if (masked_image_latents.defined()) {
      input.masked_image_latents = masked_image_latents.to(device, dtype);
    }

    if (images.defined()) {
      input.images = images.to(device, dtype);
    }

    if (mask_images.defined()) {
      input.mask_images = mask_images.to(device, dtype);
    }
    return input;
  }

  int batch_size = 0;

  // Primary input text description for image generation
  std::vector<std::string> prompts;

  // Secondary prompt for additional details (e.g., color, lighting)
  std::vector<std::string> prompts_2;

  // Negative prompt to exclude low-quality features
  std::vector<std::string> negative_prompts;

  // Secondary negative prompt to exclude additional unwanted features
  std::vector<std::string> negative_prompts_2;

  torch::Tensor images;

  torch::Tensor mask_images;

  torch::Tensor control_image;

  torch::Tensor masked_image_latents;

  torch::Tensor prompt_embeds;

  torch::Tensor pooled_prompt_embeds;

  torch::Tensor negative_prompt_embeds;

  torch::Tensor negative_pooled_prompt_embeds;

  torch::Tensor latents;

  // generation params
  DiTGenerationParams generation_params;
};

// dit related forward output params
struct DiTForwardOutput {
  // generated tensor
  std::vector<torch::Tensor> tensors;
};

}  // namespace janus
