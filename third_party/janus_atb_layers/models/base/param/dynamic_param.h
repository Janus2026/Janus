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
#ifndef ATB_SPEED_DYNAMIC_PARAM_H
#define ATB_SPEED_DYNAMIC_PARAM_H
#include <nlohmann/json.hpp>
#include "atb_speed/utils/singleton.h"
#include "operations/fusion/utils.h"
#include "models/base/param/param_utils.h"

namespace atb_speed {
namespace base {

template<typename T>
class DynamicParam {
public:
    void Parse(std::string name, nlohmann::json &paramJson)
    {
        this->enableDap_ = false;  // reset

        this->name_ = name;
        if (!paramJson.contains(name)) { return; }
        this->data_ = FetchJsonParam<T>(paramJson, name);

        std::string suffix = GetSingleton<common::DapManager>().GetSuccessorSuffix();
        if (!paramJson.contains(name + suffix)) { return; }
        this->enableDap_ = true;
        GetSingleton<common::DapManager>().SetRole(common::DapRole::SUCCESSOR);
        this->successorData_ = FetchJsonParam<T>(paramJson, name + suffix);
        GetSingleton<common::DapManager>().SetRole(common::DapRole::PRECEDER);
    }

    T& Get()
    {
        common::DapRole role = GetSingleton<common::DapManager>().GetRole();
        return role == common::DapRole::SUCCESSOR ? this->successorData_ : this->data_;
    }

    std::string name_  = "";

private:
    T data_;
    T successorData_;
    bool enableDap_ = false;
};
} // namespace base
} // namespace atb_speed
#endif