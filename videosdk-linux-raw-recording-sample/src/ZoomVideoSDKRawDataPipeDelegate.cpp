#include "ZoomVideoSDKRawDataPipeDelegate.h"
#include <glib.h>

#include <curl/curl.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <set>
#include <mutex>

#ifdef HAVE_LIBYUV
#include <libyuv.h>
#endif

using namespace ZOOMVIDEOSDK;

namespace {

bool ensure_curl_global_init()
{
    static bool initialized = false;
    if (!initialized)
    {
        CURLcode rc = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (rc != CURLE_OK)
        {
            g_printerr("curl_global_init failed: %d\n", static_cast<int>(rc));
            return false;
        }
        initialized = true;
    }
    return true;
}

const char* get_analyzer_url()
{
    static std::string url;
    if (url.empty())
    {
        const char* env = std::getenv("ANALYZER_URL");
        url = env && *env ? env : "http://host.docker.internal:8001/frame";
    }
    return url.c_str();
}

enum class FrameMode
{
    kRgb24,
    kY8,
};

struct FrameFormatConfig
{
    FrameMode mode;
    std::string pixel_format;
    int bytes_per_pixel;
};

std::string to_lower_copy(const char* s)
{
    if (!s)
    {
        return "";
    }
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

const FrameFormatConfig& get_frame_format_config()
{
    static const FrameFormatConfig cfg = []() {
        const std::string mode = to_lower_copy(std::getenv("ANALYZER_FRAME_FORMAT"));
        if (mode == "y8")
        {
            return FrameFormatConfig{FrameMode::kY8, "y8", 1};
        }
        return FrameFormatConfig{FrameMode::kRgb24, "rgb24", 3};
    }();
    return cfg;
}

unsigned char clamp_to_u8(int x)
{
    if (x < 0)
    {
        return 0;
    }
    if (x > 255)
    {
        return 255;
    }
    return static_cast<unsigned char>(x);
}

bool extract_y8(
    YUVRawDataI420* data,
    int width,
    int height,
    std::vector<unsigned char>& out)
{
    const unsigned char* y = reinterpret_cast<const unsigned char*>(data->GetYBuffer());
    if (!y)
    {
        g_printerr("[FRAME_TX] missing Y buffer, skipping frame\n");
        return false;
    }

    out.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    const int y_stride = width;
    for (int row = 0; row < height; ++row)
    {
        std::memcpy(out.data() + static_cast<size_t>(row) * static_cast<size_t>(width),
                    y + static_cast<size_t>(row) * static_cast<size_t>(y_stride),
                    static_cast<size_t>(width));
    }
    return true;
}

bool convert_i420_to_rgb24(
    YUVRawDataI420* data,
    int width,
    int height,
    std::vector<unsigned char>& out_rgb)
{
    const unsigned char* y = reinterpret_cast<const unsigned char*>(data->GetYBuffer());
    const unsigned char* u = reinterpret_cast<const unsigned char*>(data->GetUBuffer());
    const unsigned char* v = reinterpret_cast<const unsigned char*>(data->GetVBuffer());

    if (!y || !u || !v)
    {
        g_printerr("[FRAME_TX] missing I420 planes (Y/U/V), skipping frame\n");
        return false;
    }

    const int y_stride = width;
    const int uv_stride = (width + 1) / 2;
    const int rgb_stride = width * 3;

    out_rgb.resize(static_cast<size_t>(height) * static_cast<size_t>(rgb_stride));

#ifdef HAVE_LIBYUV
    const int rc = libyuv::I420ToRGB24(
        y, y_stride,
        u, uv_stride,
        v, uv_stride,
        out_rgb.data(), rgb_stride,
        width, height);

    if (rc != 0)
    {
        g_printerr("[FRAME_TX] libyuv I420ToRGB24 failed (rc=%d), skipping frame\n", rc);
        return false;
    }
    return true;
#else
    // Fallback conversion when libyuv is unavailable.
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            const int y_idx = row * y_stride + col;
            const int uv_idx = (row / 2) * uv_stride + (col / 2);

            const int Y = static_cast<int>(y[y_idx]);
            const int U = static_cast<int>(u[uv_idx]) - 128;
            const int V = static_cast<int>(v[uv_idx]) - 128;

            const int C = Y - 16;
            const int D = U;
            const int E = V;

            const int R = (298 * C + 409 * E + 128) >> 8;
            const int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
            const int B = (298 * C + 516 * D + 128) >> 8;

            unsigned char* dst = out_rgb.data() + static_cast<size_t>(row) * static_cast<size_t>(rgb_stride)
                                 + static_cast<size_t>(col) * 3u;
            dst[0] = clamp_to_u8(R);
            dst[1] = clamp_to_u8(G);
            dst[2] = clamp_to_u8(B);
        }
    }
    return true;
#endif
}

void log_frame_tx_once(
    const char* user_id,
    int instance_id,
    const char* pixel_format,
    int width,
    int height,
    int bytes)
{
    static std::mutex mu;
    static std::set<std::string> seen;

    const std::string uid = (user_id && user_id[0] != '\0') ? user_id : "unknown";
    const std::string key = uid + "|" + std::to_string(instance_id);

    std::lock_guard<std::mutex> lock(mu);
    if (seen.insert(key).second)
    {
        g_print("[FRAME_TX] format=%s w=%d h=%d bytes=%d\n",
                pixel_format, width, height, bytes);
    }
}

void send_frame_http(
    const unsigned char* buffer,
    int length,
    int width,
    int height,
    const char* user_id,
    int instance_id,
    const char* pixel_format,
    int bytes_per_pixel)
{
    if (!buffer || length <= 0 || width <= 0 || height <= 0)
    {
        return;
    }

    if (!ensure_curl_global_init())
    {
        return;
    }

    CURL* curl = curl_easy_init();
    if (!curl)
    {
        g_printerr("curl_easy_init failed\n");
        return;
    }

    curl_easy_setopt(curl, CURLOPT_URL, get_analyzer_url());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buffer);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(length));
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 500L);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/octet-stream");

    char header_buf[128];

    std::snprintf(header_buf, sizeof(header_buf), "X-Width: %d", width);
    headers = curl_slist_append(headers, header_buf);

    std::snprintf(header_buf, sizeof(header_buf), "X-Height: %d", height);
    headers = curl_slist_append(headers, header_buf);

    const char* safe_user_id = (user_id && user_id[0] != '\0') ? user_id : "unknown";
    std::string uid_header = std::string("X-User-Id: ") + safe_user_id;
    headers = curl_slist_append(headers, uid_header.c_str());

    std::snprintf(header_buf, sizeof(header_buf), "X-Instance-Id: %d", instance_id);
    headers = curl_slist_append(headers, header_buf);

    headers = curl_slist_append(headers, "X-Source: zoom-bot");

    std::string pixel_header = std::string("X-Pixel-Format: ") + pixel_format;
    headers = curl_slist_append(headers, pixel_header.c_str());

    std::snprintf(header_buf, sizeof(header_buf), "X-Bytes-Per-Pixel: %d", bytes_per_pixel);
    headers = curl_slist_append(headers, header_buf);

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK)
    {
        g_printerr("curl_easy_perform failed: %s\n", curl_easy_strerror(res));
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}

} // namespace

std::vector<ZoomVideoSDKRawDataPipeDelegate*> ZoomVideoSDKRawDataPipeDelegate::list_;
int ZoomVideoSDKRawDataPipeDelegate::instance_count = 0;

int j = 0;

// Keep existing global throttling behavior unchanged.
static int g_frame_counter = 0;

ZoomVideoSDKRawDataPipeDelegate::ZoomVideoSDKRawDataPipeDelegate(IZoomVideoSDKUser* user)
{
    instance_id_ = instance_count++;
    user_ = user;
    user_->GetVideoPipe()->subscribe(ZoomVideoSDKResolution_360P, this);
    list_.push_back(this);
}

ZoomVideoSDKRawDataPipeDelegate::ZoomVideoSDKRawDataPipeDelegate(IZoomVideoSDKUser* user, bool isShareScreen)
{
    instance_id_ = instance_count++;
    user_ = user;
    user_->GetSharePipe()->subscribe(ZoomVideoSDKResolution_360P, this);
    list_.push_back(this);
}

ZoomVideoSDKRawDataPipeDelegate::~ZoomVideoSDKRawDataPipeDelegate()
{
    log(L"********** [%d] Finishing encoding, user: %s, %dx%d.\n",
        instance_id_, user_->getUserName(), in_width, in_height);

    user_->GetVideoPipe()->unSubscribe(this);
    log(L"********** [%d] UnSubscribe, user: %s.\n",
        instance_id_, user_->getUserName());
    instance_count--;
    user_ = nullptr;
}

ZoomVideoSDKRawDataPipeDelegate* ZoomVideoSDKRawDataPipeDelegate::find_instance(IZoomVideoSDKUser* user)
{
    for (auto iter = list_.begin(); iter != list_.end(); iter++)
    {
        ZoomVideoSDKRawDataPipeDelegate* item = *iter;
        if (item->user_ == user)
        {
            return item;
        }
    }
    return nullptr;
}

void ZoomVideoSDKRawDataPipeDelegate::stop_encoding_for(IZoomVideoSDKUser* user)
{
    ZoomVideoSDKRawDataPipeDelegate* encoder = ZoomVideoSDKRawDataPipeDelegate::find_instance(user);
    if (encoder)
    {
        encoder->~ZoomVideoSDKRawDataPipeDelegate();
    }
}

void ZoomVideoSDKRawDataPipeDelegate::stop_encoding_for(IZoomVideoSDKUser* user, bool isShareScreen)
{
    ZoomVideoSDKRawDataPipeDelegate* encoder = ZoomVideoSDKRawDataPipeDelegate::find_instance(user);
    if (encoder)
    {
        encoder->~ZoomVideoSDKRawDataPipeDelegate();
    }
}

void ZoomVideoSDKRawDataPipeDelegate::onRawDataFrameReceived(YUVRawDataI420* data)
{
    const zchar_t* user_id = user_->getUserID();
    const int width = data->GetStreamWidth();
    const int height = data->GetStreamHeight();
    const int source_id = data->GetSourceID();

    if ((source_id != current_sourceID) &&
        (source_id == 0 || (user_id && std::strlen(user_id) > 0)))
    {
        log(L"********** [%d] Start encoding, user: %s, %dx%d, sourceID: %d.\n",
            instance_id_, user_->getUserName(), width, height, source_id);

        current_sourceID = source_id;
        in_width = width;
        in_height = height;
    }

    // Throttle unchanged: send every 2nd frame.
    g_frame_counter++;
    if (g_frame_counter % 2 != 0)
    {
        return;
    }

    if (width <= 0 || height <= 0)
    {
        return;
    }

    const FrameFormatConfig& fmt = get_frame_format_config();

    std::vector<unsigned char> payload;
    bool ok = false;

    if (fmt.mode == FrameMode::kY8)
    {
        ok = extract_y8(data, width, height, payload);
        if (!ok)
        {
            g_printerr("[FRAME_TX] y8 extraction failed, skipping frame\n");
            return;
        }
    }
    else
    {
        ok = convert_i420_to_rgb24(data, width, height, payload);
        if (!ok)
        {
            g_printerr("[FRAME_TX] rgb24 conversion failed, skipping frame\n");
            return;
        }
    }

    log_frame_tx_once(user_id, instance_id_, fmt.pixel_format.c_str(), width, height,
                      static_cast<int>(payload.size()));

    send_frame_http(
        payload.data(),
        static_cast<int>(payload.size()),
        width,
        height,
        user_id,
        instance_id_,
        fmt.pixel_format.c_str(),
        fmt.bytes_per_pixel);
}

void ZoomVideoSDKRawDataPipeDelegate::onRawDataStatusChanged(RawDataStatus status)
{
    log(L"********** [%d] onRawDataStatusChanged, user: %s, %d.\n",
        instance_id_, user_->getUserName(), status);
    if (status == RawData_On)
    {
    }
    else
    {
    }
}

void ZoomVideoSDKRawDataPipeDelegate::err_msg(int code)
{
}

void ZoomVideoSDKRawDataPipeDelegate::log(const wchar_t* format, ...)
{
    va_list args;
    va_start(args, format);
    vwprintf(format, args);
    va_end(args);
}
