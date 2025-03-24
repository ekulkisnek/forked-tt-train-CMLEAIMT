// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// Test suite for dataset generation functionality
#include <gtest/gtest.h>
#include "datasets/generators.hpp"
#include "autograd/auto_context.hpp"

using namespace ttml::datasets;

class DatasetGeneratorTest : public ::testing::Test {
    // Base fixture for dataset testing
    // Provides common setup and teardown
protected:
    void SetUp() override {
        // Initialize random seed for reproducibility
        // Set up common test data structures
    }
};

// Tests for regression dataset generation
TEST_F(DatasetGeneratorTest, RegressionDatasetCreation) {
    // Verifies correct dataset size and shape
    MakeRegressionParams params = {100, 10, 3};
    auto dataset = make_regression(params);
    EXPECT_EQ(dataset.get_size(), params.n_samples);

    // Checks feature and target dimensionality
    auto sample = dataset.get_item(0);
    EXPECT_EQ(sample.first.size(), params.n_features);
    EXPECT_EQ(sample.second.size(), params.n_targets);
}

TEST_F(DatasetGeneratorTest, RegressionReproducibilityWithSeed) {
    // Tests reproducibility with a seed
    MakeRegressionParams params = {100, 10, 3, 0.1F, true};
    ttml::autograd::AutoContext::get_instance().set_seed(322);
    auto dataset1 = make_regression(params);
    ttml::autograd::AutoContext::get_instance().set_seed(322);
    auto dataset2 = make_regression(params);

    for (size_t i = 0; i < params.n_samples; ++i) {
        auto sample1 = dataset1.get_item(i);
        auto sample2 = dataset2.get_item(i);
        EXPECT_EQ(sample1.first, sample2.first);
        EXPECT_EQ(sample1.second, sample2.second);
    }
}

TEST_F(DatasetGeneratorTest, RegressionNoiseEffectOnTargets) {
    // Test case to check if noise affects the targets
    MakeRegressionParams params = {100, 10, 3, 0.5F, true};
    auto dataset = make_regression(params);
    auto sample = dataset.get_item(0);

    // Generate a dataset with no noise for comparison
    params.noise = 0.0F;
    auto dataset_no_noise = make_regression(params);
    auto sample_no_noise = dataset_no_noise.get_item(0);

    for (size_t t = 0; t < params.n_targets; ++t) {
        EXPECT_NE(sample.second[t], sample_no_noise.second[t]);
    }
}

TEST_F(DatasetGeneratorTest, RegressionBiasEffectOnTargets) {
    // Test case to check if bias term affects the targets
    MakeRegressionParams params = {100, 10, 3, 0.0F, true};
    // Generate a dataset with bias
    auto dataset_with_bias = make_regression(params);
    auto sample_with_bias = dataset_with_bias.get_item(0);

    // Generate a dataset without bias
    params.bias = false;
    auto dataset_without_bias = make_regression(params);
    auto sample_without_bias = dataset_without_bias.get_item(0);

    for (size_t t = 0; t < params.n_targets; ++t) {
        EXPECT_NE(sample_with_bias.second[t], sample_without_bias.second[t]);
    }
}