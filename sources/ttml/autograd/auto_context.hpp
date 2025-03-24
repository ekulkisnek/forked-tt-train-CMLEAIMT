// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <random>
#include <optional>
#include <span>

// Placeholder includes -  Replace with actual headers as needed.
// These are crucial for compilation and should reflect your project's structure.
#include "core/device.hpp" // Assuming this is where core::Device is defined
#include "tensor.hpp" // Assuming Tensor and TensorPtr are defined here.
#include "operation.hpp" // Assuming Operation is defined here.
#include "graph.hpp" // Assuming NodeId is defined here.



namespace ttml {
namespace autograd {

//  Forward declarations to avoid circular dependencies.  Adapt as needed.
//class Tensor;
//using TensorPtr = std::shared_ptr<Tensor>; // Or whatever your Tensor pointer type is.


// Main context class managing the computational graph
class AutoContext {
    // Singleton instance for global gradient tracking
    static AutoContext& instance_;

    // Stores the computational graph for backward propagation
    // Each node represents an operation with inputs and gradients
    struct Node {
        std::vector<std::shared_ptr<ttml::autograd::Tensor>> inputs;     // Input tensors for this operation
        std::vector<std::shared_ptr<ttml::autograd::Tensor>> gradients;  // Computed gradients for backprop
        ttml::autograd::Operation op;                      // The mathematical operation performed
    };

public:
    // Methods for graph manipulation and gradient computation
    void reset_graph();                    // Clears the computational graph
    void backward(std::shared_ptr<ttml::autograd::Tensor> loss);         // Initiates backpropagation
    void retain_graph(bool retain);        // Controls graph retention between backward passes

    // Memory optimization methods
    void enable_checkpointing();           // Enables gradient checkpointing
    void optimize_memory_usage();          // Implements memory-saving strategies

    // Add a method to get the instance if needed
    static AutoContext& getInstance(){
        return instance_;
    }
    
    ~AutoContext() = default;

private:
    AutoContext() = default;
};


// Definition of the static member.  Crucial for singleton pattern.
AutoContext& AutoContext::instance_ = *new AutoContext();


} // namespace autograd
} // namespace ttml