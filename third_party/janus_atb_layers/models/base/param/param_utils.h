/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef ATB_SPEED_PARAM_UTILS_H
#define ATB_SPEED_PARAM_UTILS_H
#include <nlohmann/json.hpp>
#include "atb_speed/log.h"

namespace atb_speed {
namespace base {

/// A template function to verify the type of the parameter.
/// It will call `nlohmann::json`'s `get` method to extract the JSON value and convert to the target type.
/// \tparam T The acceptable data types are int, bool, float, string, uint32_t, std::vector<bool>, std::vector<int>.
/// \param paramJson An `nlohmann::json` object holds all the required parameters.
/// \param key The key used to retrieve the value from the `nlohmann::json` object.
/// \param isVector A flag indicates whether the target value is in the vector format.
/// \return The extracted value after type conversion.
template <typename T>
T FetchJsonParam(const nlohmann::json& paramJson, const std::string& key, bool isVector = false)
{
    try {
        if (isVector) {
            return paramJson.get<T>();
        } else {
            return paramJson.at(key).get<T>();
        }
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "Failed to parse parameter " << key << ": " << e.what() << ". Please check the type of param.";
        ATB_SPEED_LOG_ERROR(ss.str(), ATB_MODELS_MODEL_PARAM_JSON_INVALID);
        throw std::runtime_error(ss.str());
    }
}

} // namespace base
} // namespace atb_speed
#endif