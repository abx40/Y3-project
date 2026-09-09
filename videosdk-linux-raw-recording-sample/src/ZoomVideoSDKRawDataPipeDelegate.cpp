#include "ZoomVideoSDKRawDataPipeDelegate.h"
#include <glib.h>

// NEW: extra headers for HTTP + buffer handling
#include <curl/curl.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdarg>   // required for va_list, va_start, vwprintf
#include <cstdio>
#include <cstdlib>

using namespace ZOOMVIDEOSDK;

// ----------------------------------------------------------------------
// Anonymous namespace for helper functions used only in this file
// ----------------------------------------------------------------------
namespace {

// Lazily initialise libcurl once per process
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

/**
 * Resolve the analyzer URL from env (ANALYZER_URL) or use a default.
 */
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

/**
 * Send a single grayscale frame (Y-plane only) via HTTP POST.
 *
 * - buffer: width*height bytes of grayscale data
 * - length: size of buffer
 * - width, height: frame size
 * - userId: Zoom user ID for debugging / tracking
 * - instanceId: delegate instance (for logs / tracing)
 */
void send_frame_http(
    const unsigned char *buffer,
    int length,
    int width,
    int height,
    const char *userId,
    int instanceId)
{
    if (!buffer || length <= 0 || width <= 0 || height <= 0)
    {
        return;
    }

    if (!ensure_curl_global_init())
    {
        return;
    }

    CURL *curl = curl_easy_init();
    if (!curl)
    {
        g_printerr("curl_easy_init failed\n");
        return;
    }

    curl_easy_setopt(curl, CURLOPT_URL, get_analyzer_url());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buffer);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(length));

    // modest timeout so the bot doesn't hang if the server is slow
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 500L);

    // Build headers with metadata
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/octet-stream");

    char headerBuf[128];

    std::snprintf(headerBuf, sizeof(headerBuf), "X-Width: %d", width);
    headers = curl_slist_append(headers, headerBuf);

    std::snprintf(headerBuf, sizeof(headerBuf), "X-Height: %d", height);
    headers = curl_slist_append(headers, headerBuf);

    std::snprintf(headerBuf, sizeof(headerBuf), "X-Instance-Id: %d", instanceId);
    headers = curl_slist_append(headers, headerBuf);

    if (userId && userId[0] != '\0')
    {
        std::string uidHeader = std::string("X-User-Id: ") + userId;
        headers = curl_slist_append(headers, uidHeader.c_str());
    }

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
// ----------------------------------------------------------------------


std::vector<ZoomVideoSDKRawDataPipeDelegate *> ZoomVideoSDKRawDataPipeDelegate::list_;
int ZoomVideoSDKRawDataPipeDelegate::instance_count = 0;

// Global test var kept as in original
int j = 0;

// Simple global frame counter to throttle HTTP sends
static int g_frame_counter = 0;

ZoomVideoSDKRawDataPipeDelegate::ZoomVideoSDKRawDataPipeDelegate(IZoomVideoSDKUser *user)
{
    instance_id_ = instance_count++;
    user_ = user;
    user_->GetVideoPipe()->subscribe(ZoomVideoSDKResolution_360P, this);
    list_.push_back(this);
}

ZoomVideoSDKRawDataPipeDelegate::ZoomVideoSDKRawDataPipeDelegate(IZoomVideoSDKUser *user, bool isShareScreen)
{
    instance_id_ = instance_count++;
    user_ = user;
    user_->GetSharePipe()->subscribe(ZoomVideoSDKResolution_360P, this);
    list_.push_back(this);
}

ZoomVideoSDKRawDataPipeDelegate::~ZoomVideoSDKRawDataPipeDelegate()
{
    // finish ffmpeg encoding
    log(L"********** [%d] Finishing encoding, user: %s, %dx%d.\n",
        instance_id_, user_->getUserName(), in_width, in_height);

    user_->GetVideoPipe()->unSubscribe(this);
    log(L"********** [%d] UnSubscribe, user: %s.\n",
        instance_id_, user_->getUserName());
    instance_count--;
    user_ = nullptr;
}


ZoomVideoSDKRawDataPipeDelegate *ZoomVideoSDKRawDataPipeDelegate::find_instance(IZoomVideoSDKUser *user)
{
    for (auto iter = list_.begin(); iter != list_.end(); iter++)
    {
        ZoomVideoSDKRawDataPipeDelegate *item = *iter;
        if (item->user_ == user)
        {
            return item;
        }
    }
    return nullptr;
}

void ZoomVideoSDKRawDataPipeDelegate::stop_encoding_for(IZoomVideoSDKUser *user)
{
    ZoomVideoSDKRawDataPipeDelegate *encoder = ZoomVideoSDKRawDataPipeDelegate::find_instance(user);
    if (encoder)
    {
        encoder->~ZoomVideoSDKRawDataPipeDelegate();
    }
}

void ZoomVideoSDKRawDataPipeDelegate::stop_encoding_for(IZoomVideoSDKUser *user, bool isShareScreen)
{
    ZoomVideoSDKRawDataPipeDelegate *encoder = ZoomVideoSDKRawDataPipeDelegate::find_instance(user);
    if (encoder)
    {
        encoder->~ZoomVideoSDKRawDataPipeDelegate();
    }
}

void ZoomVideoSDKRawDataPipeDelegate::onRawDataFrameReceived(YUVRawDataI420 *data)
{
    const zchar_t *userName = user_->getUserName();
    const zchar_t *userID = user_->getUserID();
    const int width = data->GetStreamWidth();
    const int height = data->GetStreamHeight();
    const int bufLen = data->GetBufferLen();
    const int rotation = data->GetRotation();
    const int sourceID = data->GetSourceID();

    if ((sourceID != current_sourceID) && (sourceID == 0 || strlen(userID) > 0) // to skip frames when sourceID comes in but userID is not ready, otherwise create another sepreate file for this moment.
    )
    {
        log(L"********** [%d] Start encoding, user: %s, %dx%d, sourceID: %d.\n",
            instance_id_, user_->getUserName(), width, height, sourceID);

        current_sourceID = sourceID;
        in_width = width;
        in_height = height;
    }
    else
    {
    }

    // ------------------------------------------------------------------
    // NEW: send frames over HTTP (throttled)
    // ------------------------------------------------------------------

    // Throttle so we don't DDoS your inference server:
    // send 15 out of every 30 frames (tweak as you like).
    g_frame_counter++;
    if (g_frame_counter % 2 != 0)
    {
        return;
    }

    // Grab the Y-plane (luma) as a simple grayscale image
// Y plane pointer
	unsigned char *yBuffer = reinterpret_cast<unsigned char*>(data->GetYBuffer());

	// For I420 with no padding, stride = width
	int yStride = width;

    if (!yBuffer || width <= 0 || height <= 0)
    {
        return;
    }

    // Flatten into width*height contiguous buffer
    std::vector<unsigned char> gray;
    gray.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

    for (int row = 0; row < height; ++row)
    {
        const unsigned char *srcRow = yBuffer + row * yStride;
        unsigned char *dstRow = gray.data() + row * width;
        std::memcpy(dstRow, srcRow, static_cast<size_t>(width));
    }

    // Fire-and-forget HTTP POST
    send_frame_http(
        gray.data(),
        static_cast<int>(gray.size()),
        width,
        height,
        userID,
        instance_id_);
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
    // You can wire this into your own logging if you like
}

void ZoomVideoSDKRawDataPipeDelegate::log(const wchar_t *format, ...)
{
    // Minimal logging implementation so logs don't silently vanish
    va_list args;
    va_start(args, format);
    vwprintf(format, args);
    va_end(args);
}
