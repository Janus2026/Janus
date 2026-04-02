#pragma once

#include <atomic>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "anthropic_service_impl.h"
#include "chat_service_impl.h"
#include "completion_service_impl.h"
#include "embedding_service_impl.h"
#include "image_generation_service_impl.h"
#include "models_service_impl.h"
#include "qwen3_rerank_service_impl.h"
#include "rec_completion_service_impl.h"
#include "rerank_service_impl.h"
#include "sample_service_impl.h"
#include "janus_service.pb.h"

namespace janus {

class APIService : public proto::JanusAPIService {
 public:
  APIService(Master* master,
             const std::vector<std::string>& model_names,
             const std::vector<std::string>& model_versions);
  ~APIService() = default;

  void Completions(::google::protobuf::RpcController* controller,
                   const proto::CompletionRequest* request,
                   proto::CompletionResponse* response,
                   ::google::protobuf::Closure* done) override;

  void CompletionsHttp(::google::protobuf::RpcController* controller,
                       const proto::HttpRequest* request,
                       proto::HttpResponse* response,
                       ::google::protobuf::Closure* done) override;

  void Sample(::google::protobuf::RpcController* controller,
              const proto::SampleRequest* request,
              proto::SampleResponse* response,
              ::google::protobuf::Closure* done) override;

  void SampleHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void ChatCompletions(::google::protobuf::RpcController* controller,
                       const proto::ChatRequest* request,
                       proto::ChatResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ChatCompletionsHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) override;

  void Embeddings(::google::protobuf::RpcController* controller,
                  const proto::EmbeddingRequest* request,
                  proto::EmbeddingResponse* response,
                  ::google::protobuf::Closure* done) override;

  void EmbeddingsHttp(::google::protobuf::RpcController* controller,
                      const proto::HttpRequest* request,
                      proto::HttpResponse* response,
                      ::google::protobuf::Closure* done) override;

  void ImageGeneration(::google::protobuf::RpcController* controller,
                       const proto::ImageGenerationRequest* request,
                       proto::ImageGenerationResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ImageGenerationHttp(::google::protobuf::RpcController* controller,
                           const proto::HttpRequest* request,
                           proto::HttpResponse* response,
                           ::google::protobuf::Closure* done) override;

  void Rerank(::google::protobuf::RpcController* controller,
              const proto::RerankRequest* request,
              proto::RerankResponse* response,
              ::google::protobuf::Closure* done) override;

  void RerankHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void Models(::google::protobuf::RpcController* controller,
              const proto::ModelListRequest* request,
              proto::ModelListResponse* response,
              ::google::protobuf::Closure* done) override;

  void ModelsHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void ModelVersionsHttp(::google::protobuf::RpcController* controller,
                         const proto::HttpRequest* request,
                         proto::HttpResponse* response,
                         ::google::protobuf::Closure* done) override;

  void AnthropicMessagesHttp(::google::protobuf::RpcController* controller,
                             const proto::HttpRequest* request,
                             proto::HttpResponse* response,
                             ::google::protobuf::Closure* done) override;

  void GetFreePortHttp(::google::protobuf::RpcController* controller,
                       const proto::HttpRequest* request,
                       proto::HttpResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ForkMaster(::google::protobuf::RpcController* controller,
                  const proto::MasterInfos* request,
                  proto::Status* response,
                  ::google::protobuf::Closure* done) override;

  void ForkMasterHttp(::google::protobuf::RpcController* controller,
                      const proto::HttpRequest* request,
                      proto::HttpResponse* response,
                      ::google::protobuf::Closure* done) override;

  void Sleep(::google::protobuf::RpcController* controller,
             const proto::MasterInfos* request,
             proto::Status* response,
             ::google::protobuf::Closure* done) override;

  void SleepHttp(::google::protobuf::RpcController* controller,
                 const proto::HttpRequest* request,
                 proto::HttpResponse* response,
                 ::google::protobuf::Closure* done) override;

  void Wakeup(::google::protobuf::RpcController* controller,
              const proto::MasterInfos* request,
              proto::Status* response,
              ::google::protobuf::Closure* done) override;

  void WakeupHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void LinkD2D(::google::protobuf::RpcController* controller,
               const proto::D2DLinkRequest* request,
               proto::Status* response,
               ::google::protobuf::Closure* done) override;

  void LinkD2DHttp(::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done) override;

  void UnlinkD2D(::google::protobuf::RpcController* controller,
                 const proto::D2DLinkRequest* request,
                 proto::Status* response,
                 ::google::protobuf::Closure* done) override;

  void UnlinkD2DHttp(::google::protobuf::RpcController* controller,
                     const proto::HttpRequest* request,
                     proto::HttpResponse* response,
                     ::google::protobuf::Closure* done) override;

  void Resize(::google::protobuf::RpcController* controller,
              const proto::ResizeRequest* request,
              proto::Status* response,
              ::google::protobuf::Closure* done) override;

  void ResizeHttp(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

 private:
  bool ParseForkMasterRequest(const proto::MasterInfos* request,
                              Options& options);
  void set_model_master(const std::string& model_id, Master* master);
  bool has_model_master(const std::string& model_id) const;
  bool add_model_master_if_absent(const std::string& model_id, Master* master);
  Master* get_model_master(const std::string& model_id) const;
  void remove_model_master(const std::string& model_id);

  Master* master_;
  std::atomic<bool> initial_master_slept_{false};
  mutable std::shared_mutex masters_mutex_;
  std::unordered_map<std::string, Master*> masters_;
  std::unique_ptr<AnthropicServiceImpl> anthropic_service_impl_;
  std::unique_ptr<CompletionServiceImpl> completion_service_impl_;
  std::unique_ptr<SampleServiceImpl> sample_service_impl_;
  std::unique_ptr<ChatServiceImpl> chat_service_impl_;
  std::unique_ptr<MMChatServiceImpl> mm_chat_service_impl_;
  std::unique_ptr<EmbeddingServiceImpl> embedding_service_impl_;
  std::unique_ptr<MMEmbeddingServiceImpl> mm_embedding_service_impl_;
  std::unique_ptr<ModelsServiceImpl> models_service_impl_;
  std::unique_ptr<ImageGenerationServiceImpl> image_generation_service_impl_;
  std::unique_ptr<RerankServiceImpl> rerank_service_impl_;
  std::unique_ptr<RecCompletionServiceImpl> rec_completion_service_impl_;
};

}  // namespace janus
