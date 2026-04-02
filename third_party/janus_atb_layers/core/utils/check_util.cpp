/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "atb_speed/utils/check_util.h"

#include <iostream>
#include <map>

namespace atb_speed {
// Param Type Size
const size_t PACK_QUANT_TYPE_LENGTH = 2;
const size_t LINEAR_TYPE_LENGTH = 7;
const size_t LINEAR_BIAS_TYPE_LENGTH = 4;
const int MAX_NUM_HIDDEN_LAYER = 1000;

static std::map<std::string, std::pair<std::string, std::string>> g_integerTypeMap = {
    {"int32_t", {"2147483647", "-2147483648"}},
    {"uint32_t", {"4294967295", " "}},
    {"size_t", {"18446744073709551615", " "}},
};


int CheckParamRange(const int &intParam, int min, int max)
{
    if (intParam < min) {
        std::stringstream ss;
        ss << "This param must be a number greater or equal to " << min << ", please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    if (intParam > max) {
        std::stringstream ss;
        ss << "This param must be a number less or equal to " << max << ", please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    return intParam;
}

int CheckNumHiddenLayersValid(const int &numHiddenLayers)
{
    return CheckParamRange(numHiddenLayers, 1, MAX_NUM_HIDDEN_LAYER);
}

int CheckPositive(const int &intParam)
{
    if (intParam <= 0) {
        std::stringstream ss;
        ss << "This param must be a number greater than 0, please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    return intParam;
}

template <typename T>
void CheckLinearParamsSufficient(const std::vector<std::vector<T>> &linearParam, \
    size_t numHiddenLayers, size_t thershold)
{
    if (linearParam.size() != numHiddenLayers) {
        std::stringstream ss;
        ss << "The size of param must be equal to numHiddenLayers, please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    for (auto item : linearParam) {
        if (item.size() != thershold) {
            std::stringstream ss;
            ss << "The size of vector within param must be equal to " << thershold <<" please check." << std::endl;
            throw std::runtime_error(ss.str());
        }
    }
}

void CheckSkipLayerSet(const std::vector<int> &skipLayerSet, size_t numHiddenLayers)
{
    if (skipLayerSet.size() >= numHiddenLayers) {
        std::stringstream ss;
        ss << "The size of skipLayerSet must be less than " <<
            numHiddenLayers <<
            " please check attn and mlp skipped_layers in plugin_params." << std::endl;
        throw std::runtime_error(ss.str());
    }

    for (size_t layerId : skipLayerSet) {
        if (layerId >= numHiddenLayers) {
            std::stringstream ss;
            ss << "The layer id must be greater than or equal to 0 and less than " <<
                numHiddenLayers <<
                " please check layer id in attn and mlp skipped_layers." << std::endl;
            throw std::runtime_error(ss.str());
        }
    }
}

void CheckPackQuantParamsSufficient(const std::vector<std::vector<int>> &packQuantType, size_t numHiddenLayers)
{
    CheckLinearParamsSufficient(packQuantType, numHiddenLayers, PACK_QUANT_TYPE_LENGTH);
}

void CheckLinearPackParamsSufficient(const std::vector<std::vector<int>> &linearPackType, size_t numHiddenLayers)
{
    CheckLinearParamsSufficient(linearPackType, numHiddenLayers, LINEAR_TYPE_LENGTH);
}

void CheckLinearHasBiasSufficient(const std::vector<std::vector<bool>> &linearHasBias, size_t numHiddenLayers)
{
    CheckLinearParamsSufficient(linearHasBias, numHiddenLayers, LINEAR_BIAS_TYPE_LENGTH);
}

template void CheckLinearParamsSufficient(const std::vector<std::vector<int>> &linearParam, \
    size_t numHiddenLayers, size_t thershold);
template void CheckLinearParamsSufficient(const std::vector<std::vector<bool>> &linearParam, \
    size_t numHiddenLayers, size_t thershold);
} // namespace atb_speed