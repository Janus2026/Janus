/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include <fcntl.h>
#include <iostream>
#include <unordered_set>

#include "atb_speed/log/log_utils.h"

#include "nlohmann/json.hpp"
#include "spdlog/details/os.h"

#include "atb_speed/log/file_utils.h"
#include "atb_speed/log/log_config.h"
#include "atb_speed/log.h"

namespace atb_speed {

void LogUtils::SetMindieLogParamBool(bool& logParam, const std::string& envVar)
{
    std::string envParam = LogUtils::GetEnvParam(envVar);
    std::transform(envParam.begin(), envParam.end(), envParam.begin(), ::tolower);

    const std::unordered_set<std::string> validBoolKeys = {"true", "false", "1", "0"};
    static std::unordered_map<std::string, bool> logBoolMap = {
        { "0", false }, { "1", true },
        { "false", false }, { "true", true },
    };

    if (validBoolKeys.find(envParam) != validBoolKeys.end()) {
        logParam = logBoolMap[envParam];
    }
}

void LogUtils::SetMindieLogParamString(std::string& logParam, const std::string& envVar)
{
    std::string envParam = LogUtils::GetEnvParam(envVar);
    if (!envParam.empty()) {
        logParam = envParam;
    }
}

void LogUtils::SetMindieLogParamLevel(LogLevel& logParam, const std::string& envVar)
{
    std::string envParam = LogUtils::GetEnvParam(envVar);
    std::transform(envParam.begin(), envParam.end(), envParam.begin(), ::tolower);

    const std::unordered_set<std::string> validLevelKeys = {"debug", "info", "warn", "error", "critical"};
    static std::unordered_map<std::string, LogLevel> logLevelMap = {
        { "debug", LogLevel::debug }, { "info", LogLevel::info },
        { "warn", LogLevel::warn }, { "error", LogLevel::err },
        { "critical", LogLevel::critical },
    };

    if (validLevelKeys.find(envParam) != validLevelKeys.end()) {
        logParam = logLevelMap[envParam];
    }
}

std::string LogUtils::GetEnvParam(const std::string& mindieEnv)
{
    std::unordered_map<std::string, std::string> logFlag;
    std::vector<std::string> modules = LogUtils::Split(mindieEnv, ';');
    std::string flag;
    for (auto& module : modules) {
        module = LogUtils::Trim(module);
        size_t colonPos = module.find(':');
        if (colonPos != std::string::npos) {
            std::string moduleName = module.substr(0, colonPos);
            moduleName = LogUtils::Trim(moduleName);
            if (moduleName == COMPONENT_NAME || moduleName == ALL_COMPONENT_NAME) {
                flag = module.substr(colonPos + 1);
                flag = LogUtils::Trim(flag);
            }
        } else {
            flag = module;
        }
    }
    return flag;
}

std::string LogUtils::Trim(std::string str)
{
    if (str.empty()) {
        std::cout << "str is empty." << std::endl;
        return str;
    }

    str.erase(0, str.find_first_not_of(" "));
    str.erase(str.find_last_not_of(" ") + 1);
    return str;
}

std::vector<std::string> LogUtils::Split(const std::string &str, char delim)
{
    std::vector<std::string> tokens;
    // 1. check empty string
    if (str.empty()) {
        std::cout << "str is empty." << std::endl;
        return tokens;
    }

    auto stringFindFirstNot = [str, delim](size_t pos = 0) -> size_t {
        for (size_t i = pos; i < str.size(); i++) {
            if (str[i] != delim) {
                return i;
            }
        }
        return std::string::npos;
    };

    size_t lastPos = stringFindFirstNot(0);
    size_t pos = str.find(delim, lastPos);
    while (lastPos != std::string::npos) {
        tokens.emplace_back(str.substr(lastPos, pos - lastPos));
        lastPos = stringFindFirstNot(pos);
        pos = str.find(delim, lastPos);
    }
    return tokens;
}

void LogUtils::UpdateLogFileParam(
    std::string rotateConfig, uint32_t& maxFileSize, uint32_t& maxFiles)
{
    if (rotateConfig.empty()) {
        return;
    }
    std::istringstream configStream(rotateConfig);
    std::string option;
    std::string value;

    auto isNumeric = [](const std::string& str) {
        return !str.empty() && std::all_of(str.begin(), str.end(), ::isdigit);
    };

    while (configStream >> option) {
        if (!(configStream >> value)) {
            continue;
        }
        if (option == "-fs" && isNumeric(value)) {
            maxFileSize = static_cast<uint32_t>(std::stoi(value)) * 1024 * 1024; // 1 MB = 1024 KB = 1024 * 1024 B
        } else if (option == "-r" && isNumeric(value)) {
            maxFiles = static_cast<uint32_t>(std::stoi(value)) - 1;
        }
    }
}

void LogUtils::GetLogFileName(std::string& filename)
{
    int pid = spdlog::details::os::pid();
    auto now = std::chrono::system_clock::now();

    auto duration = now.time_since_epoch();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm nowTm = *std::localtime(&nowTime);

    std::stringstream ss;
    ss << std::put_time(&nowTm, "%Y%m%d%H%M%S");

    int millisecondsPart = milliseconds % 1000;

    const uint32_t millisecondsWidth = 3;
    ss << std::setw(millisecondsWidth) << std::setfill('0') << millisecondsPart;

    std::string timeStr = ss.str();
    filename += "/mindie-llmmodels_" + std::to_string(pid) + "_" + timeStr + ".log";
}

std::string LlmRotationFileSink::GenerateFileName(std::string &fileName, size_t index)
{
    if (index == 0u) {
        return fileName;
    }
    std::string baseName;
    std::string extName;
    std::tie(baseName, extName) = spdlog::details::file_helper::split_by_extension(fileName);
    return spdlog::fmt_lib::format(SPDLOG_FMT_STRING(SPDLOG_FILENAME_T("{}.{:02}{}")), baseName, index, extName);
}

bool LlmRotationFileSink::RenameFile(std::string &srcFileName, std::string &targetFileName)
{
    (void)spdlog::details::os::remove(targetFileName);
    return spdlog::details::os::rename(srcFileName, targetFileName) == 0;
}

void LlmRotationFileSink::Rotate()
{
    using spdlog::details::os::filename_to_str;
    using spdlog::details::os::path_exists;

    fileHelper_.close();
    for (auto i = maxFileNum_; i > 0; --i) {
        std::string src = GenerateFileName(baseFileName_, i - 1);
        if (!path_exists(src)) {
            continue;
        }
        std::string target = GenerateFileName(baseFileName_, i);
        if (!RenameFile(src, target)) {
            // retry
            spdlog::details::os::sleep_for_millis(INTERVAL_OF_SLEEP);
            if (!RenameFile(src, target)) {
                fileHelper_.reopen(true);
                currentSize_ = 0;
                std::cerr << "Error: Failed to rename " + filename_to_str(src) + " to " + filename_to_str(target);
            }
        }
    }
    fileHelper_.reopen(true);
}

} // namespace atb_speed