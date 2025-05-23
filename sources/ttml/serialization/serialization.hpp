// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "core/ttnn_fwd.hpp"

namespace ttml::optimizers {
class OptimizerBase;
}
namespace ttml::serialization {
class MsgPackFile;

void write_ttnn_tensor(MsgPackFile& file, std::string_view name, const tt::tt_metal::Tensor& tensor);
void read_ttnn_tensor(MsgPackFile& file, std::string_view name, tt::tt_metal::Tensor& tensor);

void write_autograd_tensor(
    MsgPackFile& file, std::string_view name, const ttml::autograd::TensorPtr& tensor, bool save_grads = false);
void read_autograd_tensor(MsgPackFile& file, std::string_view name, ttml::autograd::TensorPtr& tensor);

void write_named_parameters(MsgPackFile& file, std::string_view name, const ttml::autograd::NamedParameters& params);
void read_named_parameters(MsgPackFile& file, std::string_view name, ttml::autograd::NamedParameters& params);

void write_optimizer(MsgPackFile& file, std::string_view name, const optimizers::OptimizerBase* optimizer);
void read_optimizer(MsgPackFile& file, std::string_view name, optimizers::OptimizerBase* optimizer);

void write_module(MsgPackFile& file, std::string_view name, const autograd::ModuleBase* module);
void read_module(MsgPackFile& file, std::string_view name, autograd::ModuleBase* module);

}  // namespace ttml::serialization
