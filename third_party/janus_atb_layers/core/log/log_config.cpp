/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "atb_speed/log/log_config.h"

#include <fcntl.h>
#include <iostream>
#include <unordered_set>
#include <unistd.h>
#include "nlohmann/json.hpp"
#include "spdlog/details/os.h"

#include "atb_speed/log/file_utils.h"
#include "atb_speed/log/log_utils.h"
#include "atb_speed/utils/check_util.h"
#include "atb_speed/utils/file_system.h"
#include "atb_speed/log/log_error.h"

namespace atb_speed {

LogConfig::LogConfig(const LogConfig& config)
    : logToStdOut_(config.logToStdOut_),
      logToFile_(config.logToFile_),
      logVerbose_(config.logVerbose_),
      logLevel_(config.logLevel_),
      logFilePath_(config.logFilePath_),
      logFileSize_(config.logFileSize_),
      logFileCount_(config.logFileCount_) {}

int LogConfig::Init()
{
    InitLogToStdout();
    InitLogToFile();
    InitLogLevel();
    InitLogFilePath();
    InitLogVerbose();
    InitLogRotationParam();
    // CheckLogEnv();
    return LOG_OK;
}

void LogConfig::InitLogToStdout()
{
    const char *mindieLogToStdout = std::getenv("MINDIE_LOG_TO_STDOUT");
    if (mindieLogToStdout != nullptr) {
        LogUtils::SetMindieLogParamBool(logToStdOut_, mindieLogToStdout);
        return;
    }
    const char *envToStdout = std::getenv("ATB_LOG_TO_STDOUT");
    // Avoid race conditions:
    std::string toStdOut;
    if (envToStdout != nullptr) {
        toStdOut = envToStdout;
        logToStdOut_ = (toStdOut == "1");
        return;
    }
}

void LogConfig::InitLogToFile()
{
    const char *mindieLogToFile = std::getenv("MINDIE_LOG_TO_FILE");
    if (mindieLogToFile != nullptr) {
        LogUtils::SetMindieLogParamBool(logToFile_, mindieLogToFile);
        return;
    }
    const char *envToFile = std::getenv("ATB_LOG_TO_FILE");
    // Avoid race conditions:
    std::string toFile;
    if (envToFile != nullptr) {
        toFile = envToFile;
        logToFile_ = (toFile == "1");
        return;
    }
}

void LogConfig::InitLogLevel()
{
    const char *mindieLogLevel = std::getenv("MINDIE_LOG_LEVEL");
    if (mindieLogLevel != nullptr) {
        LogUtils::SetMindieLogParamLevel(logLevel_, mindieLogLevel);
        return;
    }
    const char *envLevel = std::getenv("ATB_LOG_LEVEL");
    if (envLevel != nullptr) {
        std::string envLevelStr(envLevel);
        std::transform(envLevelStr.begin(), envLevelStr.end(), envLevelStr.begin(), ::toupper);
        auto iter = LOG_LEVEL_MAP.find(envLevelStr);
        if (iter != LOG_LEVEL_MAP.end()) {
            logLevel_ =  iter->second;
            return;
        }
    }
}

void LogConfig::InitLogFilePath()
{
    const char *mindieLogPath = std::getenv("MINDIE_LOG_PATH");
    if (mindieLogPath != nullptr) {
        LogUtils::SetMindieLogParamString(logFilePath_, mindieLogPath);
        if (logFilePath_[0] != '/') {
            logFilePath_ = DEFAULT_LOG_PATH + "/" + logFilePath_ + "/debug";
        } else {
            logFilePath_ = logFilePath_ + "/log/debug";
        }
    } else {
        logFilePath_ = DEFAULT_LOG_PATH + "/debug";
    }
    LogUtils::GetLogFileName(logFilePath_);
}

void LogConfig::InitLogVerbose()
{
    const char *mindieLogVerbose = std::getenv("MINDIE_LOG_VERBOSE");
    if (mindieLogVerbose != nullptr) {
        LogUtils::SetMindieLogParamBool(logVerbose_, mindieLogVerbose);
        return;
    }
}

void LogConfig::InitLogRotationParam()
{
    const char *mindieLogRotate = std::getenv("MINDIE_LOG_ROTATE");
    if (mindieLogRotate != nullptr) {
        LogUtils::SetMindieLogParamString(logRotateConfig_, mindieLogRotate);
        LogUtils::UpdateLogFileParam(logRotateConfig_, logFileSize_, logFileCount_);
        return;
    }
}

void LogConfig::CheckLogEnv() const
{
    std::unordered_map<std::string, std::string> deprecatedEnv = {
        {"OCK_LOG_TO_STDOUT", "MINDIE_LOG_TO_STDOUT"},
        {"OCK_LOG_LEVEL", "MINDIE_LOG_LEVEL"},
        {"MINDIE_LLM_PYTHON_LOG_TO_FILE", "MINDIE_LOG_TO_FILE"},
        {"MINDIE_LLM_PYTHON_LOG_LEVEL", "MINDIE_LOG_LEVEL"},
        {"MINDIE_LLM_PYTHON_LOG_PATH", "MINDIE_LOG_PATH"},
        {"MINDIE_LLM_PYTHON_LOG_TO_STDOUT", "MINDIE_LOG_TO_STDOUT"},
        {"MINDIE_LLM_LOG_TO_FILE", "MINDIE_LOG_TO_FILE"},
        {"MINDIE_LLM_LOG_LEVEL", "MINDIE_LOG_LEVEL"},
        {"MINDIE_LLM_LOG_TO_STDOUT", "MINDIE_LOG_TO_STDOUT"},
        {"ATB_LOG_LEVEL", "MINDIE_LOG_LEVEL"},
        {"ATB_LOG_TO_FILE", "MINDIE_LOG_TO_FILE"},
        {"ATB_LOG_TO_STDOUT", "MINDIE_LOG_TO_STDOUT"},
        {"LOG_LEVEL", "MINDIE_LOG_LEVEL"},
        {"LOG_TO_FILE", "MINDIE_LOG_TO_FILE"},
        {"MINDIE_LLM_PYTHON_LOG_MAXNUM", "MINDIE_LOG_ROTATE"},
        {"MINDIE_LLM_PYTHON_LOG_MAXSIZE", "MINDIE_LOG_ROTATE"}
    };
    for (const auto& envParam : deprecatedEnv) {
        const char *env = std::getenv(envParam.first.c_str());
        if (env != nullptr) {
            std::cout << "The old environment variable " << envParam.first
                << " will be deprecated on 2025/12/31. Please use the new environment variable "
                << envParam.second << " as soon as possible." << std::endl;
        }
    }
}

void LogConfig::MakeDirsWithTimeOut(const std::string parentPath) const
{
    uint32_t limitTime = 500;
    auto start = std::chrono::steady_clock::now();
    std::chrono::milliseconds timeout(limitTime);
    while (!FileSystem::Exists(parentPath)) {
        auto it = FileSystem::Makedirs(parentPath, MAX_LOG_DIR_PERM);
        if (it) {
            break;
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            std::cout << "Create log dirs failed : timed out!" << std::endl;
            break;
        }
    }
}

int LogConfig::ValidateSettings()
{
    if (!logToFile_) {
        return LOG_OK;
    }
    if (!CheckAndGetLogPath(logFilePath_)) {
        std::cout << "Cannot get the log path." << std::endl;
        return LOG_INVALID_PARAM;
    }
    if (logFileSize_ > MAX_ROTATION_FILE_SIZE_LIMIT ||
        logFileSize_ < MIN_ROTATION_FILE_SIZE_LIMIT) {
        std::cout << "Invalid max file size, which should be greater than " <<
            MIN_ROTATION_FILE_SIZE_LIMIT << " bytes and less than " <<
            MAX_ROTATION_FILE_SIZE_LIMIT << " bytes." << std::endl;
        return LOG_INVALID_PARAM;
    }
    if (logFileCount_ > MAX_ROTATION_FILE_COUNT_LIMIT ||
        logFileCount_ < MIN_ROTATION_FILE_COUNT_LIMIT) {
        std::cout << "Invalid max file count, which should be greater than " <<
                     MIN_ROTATION_FILE_COUNT_LIMIT << " and less than " << MAX_ROTATION_FILE_COUNT_LIMIT;
        return LOG_INVALID_PARAM;
    }
    return LOG_OK;
}

bool LogConfig::CheckAndGetLogPath(const std::string& configLogPath)
{
    if (configLogPath.empty()) {
        std::cout << "The path of log in config is empty." << std::endl;
        return false;
    }

    std::string filePath = configLogPath;
    std::string baseDir = "/";
    if (configLogPath[0] != '/') { // The configLogPath is relative.
        std::string homePath;
        if (!GetHomePath(homePath).IsOk()) {
            std::cout << "Failed to get home path." << std::endl;
            return false;
        }
        baseDir = homePath;
        filePath = homePath + "/" + configLogPath;
    }

    if (filePath.length() > MAX_PATH_LENGTH) {
        std::cout << "The path of log is too long: " << filePath << std::endl;
        return false;
    }
    size_t lastSlash = filePath.rfind('/', filePath.size() - 1);
    if (lastSlash == std::string::npos) {
        std::cout << "The form of logPath is invalid: " << filePath << std::endl;
        return false;
    }

    std::string parentPath = filePath.substr(0, lastSlash);
    std::string errMsg;

    MakeDirsWithTimeOut(parentPath);

    if (!FileUtils::IsFileValid(parentPath.c_str(), errMsg, true, MAX_LOG_DIR_PERM, MAX_ROTATION_FILE_SIZE_LIMIT)) {
        throw std::runtime_error(errMsg);
    }

    int fd = open(filePath.c_str(), O_WRONLY | O_CREAT, MAX_OPEN_LOG_FILE_PERM);
    if (fd == -1) {
        throw std::runtime_error("Creating log file error: " + filePath);
    }
    close(fd);

    if (!FileUtils::RegularFilePath(filePath, baseDir, errMsg, true) ||
        !FileUtils::IsFileValid(filePath, errMsg, false, MAX_OPEN_LOG_FILE_PERM, MAX_ROTATION_FILE_SIZE_LIMIT)) {
        std::cerr << errMsg << std::endl;
        return false;
    }
    logFilePath_ = filePath;
    return true;
}

} // namespace atb_speed
