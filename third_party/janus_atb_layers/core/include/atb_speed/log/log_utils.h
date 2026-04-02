/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */

#ifndef ATB_SPEED_LOG_UTILS_H
#define ATB_SPEED_LOG_UTILS_H

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "atb_speed/log/log_config.h"
#include "spdlog/details/file_helper.h"
#include "spdlog/sinks/base_sink.h"

namespace atb_speed {

const size_t INTERVAL_OF_SLEEP = 100;

class LogUtils {
public:
    static void SetMindieLogParamBool(bool& logParam, const std::string& envVar);

    static void SetMindieLogParamString(std::string& logParam, const std::string& envVar);
    
    static void SetMindieLogParamLevel(LogLevel& logParam, const std::string& envVar);

    static std::string GetEnvParam(const std::string& mindieEnv);

    static std::string Trim(std::string str);

    static std::vector<std::string> Split(const std::string &str, char delim = ' ');

    static void UpdateLogFileParam(std::string rotateConfig, uint32_t& maxFileSize, uint32_t& maxFiles);

    static void GetLogFileName(std::string& filename);
};

class LlmRotationFileSink : public spdlog::sinks::base_sink<std::mutex> {
public:
    LlmRotationFileSink(const std::string &baseFileName, size_t maxFileSize, size_t maxFileNum,
                        const spdlog::file_event_handlers &eventHandlers)
        : baseFileName_(baseFileName), maxFileSize_(maxFileSize), maxFileNum_(maxFileNum), fileHelper_(eventHandlers)
    {
        fileHelper_.open(baseFileName_, false);
        currentSize_ = fileHelper_.size();
    }

protected:
    void sink_it_(const spdlog::details::log_msg &msg) override
    {
        spdlog::memory_buf_t formattedMsg;
        base_sink<std::mutex>::formatter_->format(msg, formattedMsg);

        size_t curSize = currentSize_ + formattedMsg.size();
        if (curSize > maxFileSize_) {
            fileHelper_.flush();
            if (fileHelper_.size() > 0) {
                Rotate();
                curSize = formattedMsg.size();
            }
        }
        fileHelper_.write(formattedMsg);
        currentSize_ = curSize;
    }

    void flush_() override { fileHelper_.flush(); }

private:
    std::string baseFileName_;
    size_t maxFileSize_;
    size_t maxFileNum_;
    spdlog::details::file_helper fileHelper_;

    size_t currentSize_{0};

private:
    std::string GenerateFileName(std::string &fileName, size_t index);

    bool RenameFile(std::string &srcFileName, std::string &targetFileName);

    void Rotate();
};

} // namespace atb_speed

#endif // ATB_SPEED_LOG_UTILS_H