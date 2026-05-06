/**
 * ModelScope Integration Module
 *
 * Handles model downloading and caching from ModelScope for llama.cpp.
 * Key features:
 * - Repository file listing & automatic model file selection based on tags
 * - Download progress tracking & automatic local caching
 *
 * Configuration:
 * - Endpoint: `MODEL_ENDPOINT` env var (default: https://modelscope.cn/)
 * - Authentication: Provide token via `-hft` CLI flag or `MS_TOKEN` env var.
 *
 * Usage: llama-cli -ms <repo_id> -hff <model_file> -hft <ms_token>  (e.g., "Qwen/Qwen3-0.6B-GGUF")
 */

#include "ms.h"

#include "common.h"
#include "log.h"
#include "http.h"
#include "download.h"
#include "build-info.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <stdexcept>
#include <cctype>
#include <algorithm>
#include <map>

// isatty and system includes
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#include <pwd.h>
#endif

namespace nl = nlohmann;

namespace ms {

namespace fs = std::filesystem;

static fs::path get_cache_directory() {
    static const fs::path cache = []() {
        if (auto * p = std::getenv("LLAMA_CACHE"); p && *p) return fs::path(p);
        if (auto * p = std::getenv("MODELSCOPE_CACHE"); p && *p) return fs::path(p);

        if (auto * p = std::getenv("XDG_CACHE_HOME"); p && *p) {
            return fs::path(p) / "modelscope";
        }

#ifndef _WIN32
        const struct passwd * pw = getpwuid(getuid());
        if (pw && pw->pw_dir && *pw->pw_dir) {
            return fs::path(pw->pw_dir) / ".cache" / "modelscope";
        }
#endif

#if defined(_WIN32)
        if (auto * p = std::getenv("USERPROFILE"); p && *p) {
            return fs::path(p) / ".cache" / "modelscope";
        }
#endif

        return fs::current_path() / ".cache" / "modelscope";
    }();
    return cache;
}

static bool is_alphanum(const char c) {
    return (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9');
}

static bool is_special_char(char c) {
    return c == '/' || c == '.' || c == '-';
}

static bool is_valid_repo_id(const std::string & repo_id) {
    if (repo_id.empty() || repo_id.length() > 256) {
        return false;
    }
    int slash = 0;
    bool special = true;

    for (const char c : repo_id) {
        if (is_alphanum(c) || c == '_') {
            special = false;
        } else if (is_special_char(c)) {
            if (special) {
                return false;
            }
            slash += (c == '/');
            special = true;
        } else {
            return false;
        }
    }
    return !special && slash == 1;
}

static bool is_valid_subpath(const fs::path & base, const fs::path & subpath) {
    if (subpath.is_absolute()) {
        return false;
    }
    std::error_code ec;
    auto b = fs::absolute(base, ec).lexically_normal();
    auto t = (b / subpath).lexically_normal();
    
    if (ec) return false;

    auto [b_end, _] = std::mismatch(b.begin(), b.end(), t.begin(), t.end());
    return b_end == b.end();
}

static nl::json api_get(const std::string & url, const std::string & token = "") {
    auto [cli, parts] = common_http_client(url);
    
    httplib::Headers headers;
    headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    if (!token.empty()) {
        headers.emplace("Cookie", "m_session_id=" + token);
    }
    
    cli.set_read_timeout(30, 0);
    cli.set_write_timeout(30, 0);
    
    auto res = cli.Get(parts.path, headers);
    
    if (!res || res->status != 200) {
        std::string body = res ? res->body : "";
        if (!body.empty()) {
            try {
                auto json_error = nl::json::parse(body);
                if (json_error.contains("Message")) {
                    body = json_error["Message"].get<std::string>();
                } else if (json_error.contains("msg")) {
                    body = json_error["msg"].get<std::string>();
                }
            } catch (...) {
                // Keep original body
            }
        }
        long http_code = res ? res->status : -1;
        throw std::runtime_error("HTTP " + std::to_string(http_code) + ": " + body);
    }
    
    return nl::json::parse(res->body);
}

static std::vector<ms_file> list_files(const std::string & repo_id, const std::string & token = "") {
    std::vector<ms_file> files;
    
    const char * model_endpoint_env = std::getenv("MODEL_ENDPOINT");
    std::string modelscope_endpoint = "https://modelscope.cn/";
    if (model_endpoint_env) {
        modelscope_endpoint = model_endpoint_env;
        if (modelscope_endpoint.back() != '/') {
            modelscope_endpoint += '/';
        }
    }
    
    std::string api_url = modelscope_endpoint + "api/v1/models/" + repo_id + "/repo/files";
    
    try {
        auto response = api_get(api_url, token);

        if (response.contains("Data") && response["Data"].contains("Files")) {
            for (const auto & file_json : response["Data"]["Files"]) {
                ms_file file;
                file.repo_id = repo_id;
                file.path = file_json.value("Path", "");
                file.size = file_json.value("Size", 0ULL);
                
                if (!file.path.empty()) {
                    file.url = modelscope_endpoint + "models/" + repo_id + "/resolve/master/" + file.path;
                    files.push_back(std::move(file));
                }
            }
        }
    } catch (const std::exception & e) {
        std::string err_msg = e.what();
        if (err_msg.find("401") != std::string::npos || 
            err_msg.find("403") != std::string::npos || 
            err_msg.find("404") != std::string::npos) {
            
            if (token.empty()) {
                LOG_DBG("%s: remote list failed (no token), relying on cache.\n", __func__);
            } else {
                LOG_ERR("%s: auth failed or repo not found: %s\n", __func__, err_msg.c_str());
            }
        } else {
            LOG_ERR("%s: failed to list files for %s: %s\n", __func__, repo_id.c_str(), err_msg.c_str());
        }
    }
    
    return files;
}

static bool string_contains(const std::string & str, const std::string & substr) {
    return str.find(substr) != std::string::npos;
}

static std::string normalize_quant_tag(const std::string & tag) {
    std::string result = tag;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

static bool matches_quant_tag(const std::string & filename, const std::string & quant_tag) {
    if (quant_tag.empty()) {
        return true;
    }
    
    std::string normalized_tag = normalize_quant_tag(quant_tag);
    std::string normalized_filename = normalize_quant_tag(filename);

    std::vector<std::string> patterns = {
        "-" + normalized_tag + ".",
        "." + normalized_tag + ".",
        "_" + normalized_tag + "."
    };
    
    for (const auto & pattern : patterns) {
        if (string_contains(normalized_filename, pattern)) {
            return true;
        }
    }
    
    return false;
}

static std::string find_best_model(const std::vector<ms_file> & files, const std::string & quant_tag) {
    if (files.empty()) {
        return "";
    }
    
    std::vector<std::string> preferred_quants = {
        "q4_k_m", "q4_0", "q5_k_m", "q5_0", "q6_k", "q8_0"
    };
    
    std::vector<std::string> tags;
    if (!quant_tag.empty()) {
        tags.push_back(quant_tag);
    }
    tags.insert(tags.end(), preferred_quants.begin(), preferred_quants.end());
    
    for (const auto & tag : tags) {
        for (const auto & file : files) {
            if (string_contains(file.path, ".gguf") && matches_quant_tag(file.path, tag)) {
                return file.path;
            }
        }
    }
    
    // Fallback
    for (const auto & file : files) {
        if (string_contains(file.path, ".gguf")) {
            return file.path;
        }
    }
    
    return "";
}

static std::string get_local_path(const ms_file & file) {
    fs::path cache_dir = get_cache_directory();
    fs::path base_path = cache_dir / "hub" / "models" / file.repo_id;

    if (!is_valid_subpath(base_path, file.path)) {
        LOG_ERR("%s: security check failed for path: %s\n", __func__, file.path.c_str());
        return "";
    }

    fs::path local_path = base_path / file.path;
    return local_path.string();
}

static std::vector<ms_file> scan_local_cache(const std::string & clean_repo_id) {
    std::vector<ms_file> cached_files;
    fs::path base_cache_dir = get_cache_directory() / "hub" / "models" / clean_repo_id;
    
    std::error_code ec;
    if (fs::exists(base_cache_dir, ec) && fs::is_directory(base_cache_dir, ec)) {
        bool found_revision_dir = false;
        for (const auto & entry : fs::directory_iterator(base_cache_dir, ec)) {
            if (ec || !fs::is_directory(entry, ec)) continue;
            
            found_revision_dir = true;
            for (const auto & file_entry : fs::directory_iterator(entry.path(), ec)) {
                if (ec || !fs::is_regular_file(file_entry, ec)) continue;
                
                const std::string fname = file_entry.path().filename().string();
                // skip temporary download files
                if (fname.size() >= 19 && fname.compare(fname.size() - 19, 19, ".downloadInProgress") == 0) {
                    continue;
                }
                
                ms_file file;
                file.repo_id = clean_repo_id;
                file.path = fname;
                file.local_path = file_entry.path().string();
                file.size = fs::file_size(file_entry.path(), ec);
                cached_files.push_back(std::move(file));
            }
        }
        
        if (!found_revision_dir) {
            for (const auto & entry : fs::directory_iterator(base_cache_dir, ec)) {
                if (ec || !fs::is_regular_file(entry, ec)) continue;
                
                const std::string fname = entry.path().filename().string();
                if (fname.size() >= 19 && fname.compare(fname.size() - 19, 19, ".downloadInProgress") == 0) {
                    continue;
                }
                
                ms_file file;
                file.repo_id = clean_repo_id;
                file.path = fname;
                file.local_path = entry.path().string();
                file.size = fs::file_size(entry.path(), ec);
                cached_files.push_back(std::move(file));
            }
        }
    }
    
    return cached_files;
}

static bool is_file_valid(const ms_file & file, const fs::path & local_path) {
    std::error_code ec;
    if (!fs::exists(local_path, ec)) {
        return false;
    }
    if (file.size > 0) {
        auto size = fs::file_size(local_path, ec);
        if (ec || size != file.size) {
            return false;
        }
    }
    return true;
}

static std::string download_file_with_common(const ms_file & selected_file, const fs::path & local_path, const std::string & token = "") {
    fs::create_directories(local_path.parent_path());
    
    if (fs::exists(local_path) && !is_file_valid(selected_file, local_path)) {
        std::error_code ec;
        fs::remove(local_path, ec);
    }

    common_header_list headers;
    headers.emplace_back("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    if (!token.empty()) {
        headers.emplace_back("Cookie", "m_session_id=" + token);
    }
    
    common_download_opts opts;
    opts.headers = headers;
    opts.offline = false;

    int status = common_download_file_single(selected_file.url, local_path.string(), opts, false);
    
    if (status >= 200 && status < 400) {
        LOG_INF("%s: successfully downloaded %s\n", __func__, local_path.string().c_str());
        return local_path.string();
    } else {
        LOG_ERR("%s: download failed with status: %d\n", __func__, status);
        return "";
    }
}

std::string download_model(const std::string & clean_repo_id, const std::string & filename, bool offline, const std::string & quant_tag, const std::string & token) {
    if (!is_valid_repo_id(clean_repo_id)) {
        LOG_ERR("%s: invalid repository: %s\n", __func__, clean_repo_id.c_str());
        return "";
    }

    std::vector<ms_file> cached_files = scan_local_cache(clean_repo_id);

    if (!filename.empty()) {
        // First check cache
        for (const auto & file : cached_files) {
            if (file.path == filename) {
                LOG_DBG("%s: found file in cache: %s\n", __func__, file.local_path.c_str());
                return file.local_path;
            }
        }
        
        // Not in cache, need to download
        if (!offline) {
            auto remote_files = list_files(clean_repo_id, token);
            for (const auto & remote_file : remote_files) {
                if (remote_file.path == filename) {
                    ms_file selected_file = remote_file;
                    std::string local_path = get_local_path(selected_file);
                    if (local_path.empty()) {
                        LOG_ERR("%s: failed to determine local path for %s\n", __func__, selected_file.path.c_str());
                        return "";
                    }

                    if (is_file_valid(selected_file, local_path)) {
                        LOG_DBG("%s: file already exists with correct size: %s\n", __func__, local_path.c_str());
                        selected_file.local_path = local_path;
                        return selected_file.local_path;
                    }

                    std::string downloaded_path = download_file_with_common(selected_file, local_path, token);
                    if (!downloaded_path.empty()) {
                        selected_file.local_path = downloaded_path;
                        return selected_file.local_path;
                    }
                    break;
                }
            }

            LOG_ERR("%s: file '%s' not found in ModelScope repository %s\n", __func__, filename.c_str(), clean_repo_id.c_str());
            if (!cached_files.empty()) {
                LOG_ERR("%s: available files in cache:\n", __func__);
                for (const auto & file : cached_files) {
                    LOG_ERR("  %s\n", file.path.c_str());
                }
            }
            if (!remote_files.empty()) {
                LOG_ERR("%s: available files in repository:\n", __func__);
                for (const auto & file : remote_files) {
                    if (string_contains(file.path, ".gguf")) {
                        LOG_ERR("  %s\n", file.path.c_str());
                    }
                }
            }
            return "";
        }

        LOG_ERR("%s: file '%s' not found in cache and offline mode is enabled\n", __func__, filename.c_str());
        if (!cached_files.empty()) {
            LOG_ERR("%s: available files in cache:\n", __func__);
            for (const auto & file : cached_files) {
                LOG_ERR("  %s\n", file.path.c_str());
            }
        }
        return "";
    }

    if (!cached_files.empty()) {
        std::string selected_file_path = find_best_model(cached_files, quant_tag);
        if (!selected_file_path.empty()) {
            for (const auto & file : cached_files) {
                if (file.path == selected_file_path) {
                    LOG_DBG("%s: found best match in cache: %s\n", __func__, file.local_path.c_str());
                    return file.local_path;
                }
            }
        }
    }

    if (!offline) {
        auto remote_files = list_files(clean_repo_id, token);
        if (remote_files.empty()) {
            LOG_ERR("%s: no files found in ModelScope repository %s\n", __func__, clean_repo_id.c_str());
            return "";
        }

        std::string selected_file_path = find_best_model(remote_files, quant_tag);
        if (selected_file_path.empty()) {
            LOG_ERR("%s: no suitable GGUF file found in ModelScope repository %s\n", __func__, clean_repo_id.c_str());
            LOG_ERR("%s: available GGUF files:\n", __func__);
            for (const auto & file : remote_files) {
                if (string_contains(file.path, ".gguf")) {
                    LOG_ERR("  %s\n", file.path.c_str());
                }
            }
            return "";
        }

        // Find the selected file
        for (const auto & file : remote_files) {
            if (file.path == selected_file_path) {
                std::string local_path = get_local_path(file);
                if (local_path.empty()) {
                    LOG_ERR("%s: failed to determine local path for %s\n", __func__, file.path.c_str());
                    return "";
                }

                if (is_file_valid(file, local_path)) {
                    LOG_DBG("%s: file already exists with correct size: %s\n", __func__, local_path.c_str());
                    return local_path;
                }

                return download_file_with_common(file, local_path, token);
            }
        }
    }

    LOG_ERR("%s: failed to find or download model from ModelScope repository %s\n", __func__, clean_repo_id.c_str());
    return "";
}

download_result download_model_with_mmproj(const std::string & clean_repo_id, const std::string & filename, bool offline, const std::string & quant_tag, const std::string & token) {
    download_result result;
    
    if (!is_valid_repo_id(clean_repo_id)) {
        LOG_ERR("%s: invalid repository: %s\n", __func__, clean_repo_id.c_str());
        return result;
    }

    result.model_path = download_model(clean_repo_id, filename, offline, quant_tag, token);

    if (result.model_path.empty()) {
        LOG_ERR("%s: no model file found in ModelScope repository %s\n", __func__, clean_repo_id.c_str());
        return result;
    }

    std::vector<ms_file> all_files = scan_local_cache(clean_repo_id);
    
    if (!offline) {
        auto remote_files = list_files(clean_repo_id, token);
        std::map<std::string, ms_file> file_map;

        for (const auto & remote_file : remote_files) {
            file_map[remote_file.path] = remote_file;
        }

        for (const auto & cached_file : all_files) {
            if (file_map.find(cached_file.path) == file_map.end()) {
                file_map[cached_file.path] = cached_file;
            }
        }

        all_files.clear();
        for (const auto & [path, file] : file_map) {
            all_files.push_back(file);
        }
    }

    for (const auto & file : all_files) {
        std::string filename_lower = file.path;
        std::transform(filename_lower.begin(), filename_lower.end(), filename_lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if ((string_contains(filename_lower, ".bin") || string_contains(filename_lower, ".gguf")) &&
            (string_contains(filename_lower, "mmproj") || 
             string_contains(filename_lower, "multimodal") ||
             string_contains(filename_lower, "vision"))) {
            
            std::string local_path = file.local_path;
            if (local_path.empty()) {
                local_path = get_local_path(file);
            }

            if (local_path.empty()) {
                // Failed security/path validation - never attempt download with an empty path.
                LOG_ERR("%s: failed to determine local path for %s\n", __func__, file.path.c_str());
                continue;
            }

            if (local_path.empty() || !fs::exists(local_path)) {
                if (!offline && !file.url.empty()) {
                    local_path = download_file_with_common(file, local_path, token);
                }
            } else {
                if (!is_file_valid(file, local_path)) {
                    if (!offline && !file.url.empty()) {
                        local_path = download_file_with_common(file, local_path, token);
                    }
                }
            }
            
            if (!local_path.empty() && fs::exists(local_path)) {
                result.mmproj_path = local_path;
                break;
            }
        }
    }
    
    return result;
}

} // namespace ms
