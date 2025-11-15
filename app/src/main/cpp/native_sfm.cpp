// Minimal JNI bridge to run SFM with OpenCV+sfm and align to ENU using DJI XMP
#include <jni.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <regex>
#include <cmath>
#include <cstring>
#include <functional>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <cerrno>
#include <algorithm>
#include <dirent.h>
#include <mutex>
#include <memory>
#include <chrono>
#include <cstdio>
#include <Eigen/Core>

#include <android/log.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/sfm.hpp>
// Ceres for global BA
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/jet.h>

#include <glog/logging.h>
#include "colmap/exe/feature.h"
#include "colmap/exe/sfm.h"
#include "colmap/exe/model.h"
#include "colmap/exe/image.h"
#include "colmap/exe/mvs.h"
#include "colmap/util/logging.h"

#include <sys/stat.h>
#include <sys/types.h>

#define LOG_TAG "sfm_native"
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using std::string;
using std::vector;

namespace {

std::once_flag g_glog_init_flag;

void ensure_glog_initialized() {
    std::call_once(g_glog_init_flag, [](){
        static char arg0[] = "colmap_android";
        static char* argv[] = {arg0, nullptr};
        colmap::InitializeGlog(argv);
        FLAGS_logtostderr = true;
        FLAGS_alsologtostderr = false;
        FLAGS_colorlogtostderr = false;
        FLAGS_log_prefix = false;
    });
}

bool dir_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool file_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

class FileLogSink : public google::LogSink {
public:
    FileLogSink() = default;

    void set_path(const std::string& path) {
        std::lock_guard<std::mutex> lock(mu_);
        path_ = path;
    }

    void send(google::LogSeverity severity,
              const char* /*full_filename*/,
              const char* /*base_filename*/,
              int /*line*/,
              const google::LogMessageTime& logmsgtime,
              const char* message,
              size_t message_len) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (path_.empty()) return;
        std::ofstream ofs(path_, std::ios::app);
        if (!ofs) return;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d",
                     logmsgtime.hour(), logmsgtime.min(), logmsgtime.sec(),
                     static_cast<int>(logmsgtime.usec() / 1000));
        const char* sev_str = severity_to_string(severity);
        std::string line = std::string("[COLMAP][") + sev_str + "] " + buf +
                           " " + std::string(message, message_len) + "\n";
        ofs << line;
        // Mirror COLMAP/glog output to logcat as well so that crashes
        // like FORTIFY aborts still have nearby context in logcat.
        int prio = ANDROID_LOG_INFO;
        switch (severity) {
            case google::GLOG_INFO:    prio = ANDROID_LOG_INFO; break;
            case google::GLOG_WARNING: prio = ANDROID_LOG_WARN; break;
            case google::GLOG_ERROR:   prio = ANDROID_LOG_ERROR; break;
            case google::GLOG_FATAL:   prio = ANDROID_LOG_FATAL; break;
            default:                   prio = ANDROID_LOG_INFO; break;
        }
        __android_log_print(prio, LOG_TAG, "%.*s",
                            static_cast<int>(line.size()),
                            line.c_str());
    }

private:
    std::string path_;
    std::mutex mu_;

    const char* severity_to_string(google::LogSeverity severity) const {
        switch (severity) {
            case google::GLOG_INFO: return "I";
            case google::GLOG_WARNING: return "W";
            case google::GLOG_ERROR: return "E";
            case google::GLOG_FATAL: return "F";
            default: return "?";
        }
    }
};

// Global sink instance with process lifetime to avoid mutex destruction
// while COLMAP worker threads may still be logging.
FileLogSink* g_file_log_sink = nullptr;
std::once_flag g_file_log_sink_once;

FileLogSink* GetOrCreateFileLogSink() {
    std::call_once(g_file_log_sink_once, []() {
        // Intentionally leaked for the lifetime of the process to keep the
        // internal mutex alive as long as glog may log.
        g_file_log_sink = new FileLogSink();
        google::AddLogSink(g_file_log_sink);
    });
    return g_file_log_sink;
}

std::string pick_sparse_model_dir(const std::string& base) {
    std::string preferred = base + "/0";
    if (dir_exists(preferred)) return preferred;
    DIR* dir = opendir(base.c_str());
    if (!dir) return "";
    struct dirent* ent;
    std::string found;
    while ((ent = readdir(dir)) != nullptr) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        std::string candidate = base + "/" + ent->d_name;
        struct stat st{};
        if (::stat(candidate.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            found = candidate;
            break;
        }
    }
    closedir(dir);
    return found;
}

std::string pick_sparse_model_for_dense(const std::string& runDir) {
    const std::string aligned = runDir + "/sparse_aligned";
    if (dir_exists(aligned)) {
        if (file_exists(aligned + "/images.bin") && file_exists(aligned + "/points3D.bin")) {
            return aligned;
        }
        const std::string nested = pick_sparse_model_dir(aligned);
        if (!nested.empty()) return nested;
    }
    const std::string sparse = runDir + "/sparse";
    if (dir_exists(sparse)) {
        const std::string nested = pick_sparse_model_dir(sparse);
        if (!nested.empty()) return nested;
        if (file_exists(sparse + "/images.bin") && file_exists(sparse + "/points3D.bin")) {
            return sparse;
        }
    }
    return "";
}

using StageFunc = int(*)(int, char**);

bool run_colmap_stage(const std::string& name,
                      StageFunc func,
                      const std::vector<std::string>& args,
                      std::ostringstream& log,
                      const std::function<void()>& flush,
                      const std::function<void(const std::string&)>& logI,
                      const std::function<void(const std::string&)>& logE) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    logI("[COLMAP] " + name + "...");
    auto t0 = std::chrono::steady_clock::now();
    int ret = func(static_cast<int>(argv.size()), argv.data());
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (ret == 0) {
        std::ostringstream ss;
        ss << name << " 完成，用时 " << std::fixed << std::setprecision(1) << secs << " s";
        logI(ss.str());
        return true;
    }
    std::ostringstream ss;
    ss << name << " 失败，返回码=" << ret;
    logE(ss.str());
    return false;
}

std::string JStringToString(JNIEnv* env, jstring js) {
    if (!js) return "";
    const char* chars = env->GetStringUTFChars(js, nullptr);
    std::string out(chars ? chars : "");
    env->ReleaseStringUTFChars(js, chars);
    return out;
}

struct DjiMeta {
    bool hasIntrinsics = false;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    // Distortion as k1,k2,p1,p2,k3
    bool hasDist = false;
    double k1=0,k2=0,p1=0,p2=0,k3=0;

    bool hasGeo = false;
    double lat = 0, lon = 0, alt = 0; // WGS84 deg, deg, meters

    bool hasGimbal = false;
    double yaw=0, pitch=0, roll=0; // degrees
};

bool read_file(const string &path, string &out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    std::ostringstream ss;
    ss << ifs.rdbuf();
    out = ss.str();
    return true;
}

// Simple mkdir -p implementation (best effort). Returns true if the final
// directory exists or was created.
bool mkdir_p(const std::string &path) {
    if (path.empty()) return false;
    // Handle absolute vs relative paths
    std::string cur;
    if (path[0] == '/') cur = "/";
    std::stringstream ss(path);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (part.empty() || part == ".") continue;
        if (!cur.empty() && cur.back() != '/') cur.push_back('/');
        cur += part;
        if (::mkdir(cur.c_str(), 0755) != 0) {
            if (errno == EEXIST) continue;
            // If something else failed, continue checking existence
        }
    }
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Extract XMP packet from JPEG by searching for <x:xmpmeta> ... </x:xmpmeta>
bool extract_xmp_xml(const string &jpeg, string &xmp) {
    size_t start = jpeg.find("<x:xmpmeta");
    if (start == string::npos) start = jpeg.find("<xmpmeta");
    if (start == string::npos) return false;
    size_t end = jpeg.find("</x:xmpmeta>", start);
    if (end == string::npos) end = jpeg.find("</xmpmeta>", start);
    if (end == string::npos) return false;
    end += strlen("</x:xmpmeta>");
    xmp = jpeg.substr(start, end - start);
    return true;
}

// Robustly extract XMP from JPEG APP1 (http://ns.adobe.com/xap/1.0/)
bool extract_xmp_from_jpeg_app1(const string &jpegPath, string &xmp) {
    std::ifstream ifs(jpegPath, std::ios::binary);
    if (!ifs) return false;
    // Read whole file into buffer (OK for typical JPEG sizes)
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (buf.size() < 4) return false;
    // JPEG SOI check
    if (!(buf[0] == 0xFF && buf[1] == 0xD8)) return false;
    size_t i = 2;
    auto read_be16 = [&](size_t off)->uint16_t{
        return (uint16_t)((buf[off] << 8) | buf[off+1]);
    };
    const char xap_preamble[] = "http://ns.adobe.com/xap/1.0/\0"; // 29 bytes with NUL terminator
    const size_t pre_len = sizeof(xap_preamble); // include NUL in compare
    while (i + 4 <= buf.size()) {
        if (buf[i] != 0xFF) break; // invalid
        unsigned char marker = buf[i+1];
        i += 2;
        if (marker == 0xDA /*SOS*/ || marker == 0xD9 /*EOI*/) break;
        if (i + 2 > buf.size()) break;
        uint16_t seglen = read_be16(i);
        if (seglen < 2) break;
        i += 2;
        if (i + (seglen - 2) > buf.size()) break;
        if (marker == 0xE1 /*APP1*/) {
            const unsigned char* data = &buf[i];
            size_t dlen = seglen - 2;
            // Check XMP preamble with NUL
            if (dlen > pre_len && std::memcmp(data, xap_preamble, pre_len) == 0) {
                // The actual XML follows the 29-byte preamble (preamble includes NUL).
                // Align to the first '<' to be robust against stray bytes.
                const unsigned char* p = data + pre_len;
                size_t left = dlen - pre_len;
                while (left && *p != '<') { ++p; --left; }
                xmp.assign(reinterpret_cast<const char*>(p), left);
                return true;
            }
        }
        i += (seglen - 2);
    }
    return false;
}

// Helper: get attribute value like drone-dji:CalibratedFocalLength="..."
bool get_xmp_attr_double(const string &xmp, const string &attrName, double &val) {
    std::regex re(attrName + "=\"([^\"]+)\"");
    std::smatch m;
    if (std::regex_search(xmp, m, re)) {
        try { val = std::stod(m[1].str()); return true; } catch (...) {}
    }
    return false;
}

bool parse_dji_xmp(const string &xmp, DjiMeta &meta) {
    double focalCalib = 0, cx = 0, cy = 0;
    bool hasF = get_xmp_attr_double(xmp, "drone-dji:CalibratedFocalLength", focalCalib);
    bool hasCx = get_xmp_attr_double(xmp, "drone-dji:CalibratedOpticalCenterX", cx);
    bool hasCy = get_xmp_attr_double(xmp, "drone-dji:CalibratedOpticalCenterY", cy);

    // DewarpData: date;fx,fy,dx,dy,k1,k2,p1,p2,k3
    std::regex reDewarp("drone-dji:DewarpData=\"([^\"]+)\"");
    std::smatch m;
    bool hasDewarp = std::regex_search(xmp, m, reDewarp);
    double fx=0, fy=0, dx=0, dy=0, k1=0, k2=0, p1=0, p2=0, k3=0;
    if (hasDewarp) {
        string payload = m[1].str();
        // split at ';' then by ','
        auto sc = payload.find(';');
        string nums = sc!=string::npos ? payload.substr(sc+1) : payload;
        std::vector<double> v; v.reserve(10);
        std::stringstream ss(nums);
        string item;
        while (std::getline(ss, item, ',')) {
            try { v.push_back(std::stod(item)); } catch (...) {}
        }
        if (v.size() >= 9) {
            fx=v[0]; fy=v[1]; dx=v[2]; dy=v[3]; k1=v[4]; k2=v[5]; p1=v[6]; p2=v[7]; k3=v[8];
        }
    }

    // GPS
    double lat=0, lon=0, alt=0; bool hasLat=false, hasLon=false, hasAlt=false;
    hasLat = get_xmp_attr_double(xmp, "drone-dji:GpsLatitude", lat);
    hasLon = get_xmp_attr_double(xmp, "drone-dji:GpsLongitude", lon);
    hasAlt = get_xmp_attr_double(xmp, "drone-dji:AbsoluteAltitude", alt);

    // Gimbal
    double yaw=0,pitch=0,roll=0; bool hasYaw=false, hasPit=false, hasRoll=false;
    hasYaw = get_xmp_attr_double(xmp, "drone-dji:GimbalYawDegree", yaw);
    hasPit = get_xmp_attr_double(xmp, "drone-dji:GimbalPitchDegree", pitch);
    hasRoll= get_xmp_attr_double(xmp, "drone-dji:GimbalRollDegree", roll);

    // Fill meta
    if (hasF || hasDewarp) {
        meta.fx = hasF ? focalCalib : fx;
        meta.fy = hasF ? focalCalib : fy;
        meta.cx = (hasCx ? cx : 0) + (hasDewarp ? dx : 0);
        meta.cy = (hasCy ? cy : 0) + (hasDewarp ? dy : 0);
        meta.hasIntrinsics = (meta.fx>0 && meta.fy>0 && meta.cx>0 && meta.cy>0);
        if (hasDewarp) { meta.k1=k1; meta.k2=k2; meta.p1=p1; meta.p2=p2; meta.k3=k3; meta.hasDist = true; }
    }
    if (hasLat && hasLon && hasAlt) { meta.lat=lat; meta.lon=lon; meta.alt=alt; meta.hasGeo=true; }
    if (hasYaw && hasPit && hasRoll) { meta.yaw=yaw; meta.pitch=pitch; meta.roll=roll; meta.hasGimbal=true; }
    return meta.hasIntrinsics || meta.hasGeo || meta.hasGimbal;
}

bool extract_dji_meta(const string &jpegPath, DjiMeta &meta) {
    string xmp;
    // Prefer APP1 scan; fallback to tag search within full file
    if (!extract_xmp_from_jpeg_app1(jpegPath, xmp)) {
        string buf;
        if (!read_file(jpegPath, buf)) return false;
        if (!extract_xmp_xml(buf, xmp)) return false;
    }
    return parse_dji_xmp(xmp, meta);
}

// WGS84 -> ECEF
cv::Point3d llh_to_ecef(double lat_deg, double lon_deg, double h_m) {
    const double a = 6378137.0; // WGS84 equatorial radius (m)
    const double e2 = 6.69437999014e-3; // eccentricity^2
    double lat = lat_deg * M_PI/180.0;
    double lon = lon_deg * M_PI/180.0;
    double sinLat = std::sin(lat), cosLat = std::cos(lat);
    double sinLon = std::sin(lon), cosLon = std::cos(lon);
    double N = a / std::sqrt(1.0 - e2 * sinLat * sinLat);
    double x = (N + h_m) * cosLat * cosLon;
    double y = (N + h_m) * cosLat * sinLon;
    double z = (N * (1.0 - e2) + h_m) * sinLat;
    return {x,y,z};
}

// ECEF -> ENU (origin at ref llh)
cv::Matx33d ecef_to_enu_R(double lat0_deg, double lon0_deg) {
    double lat = lat0_deg * M_PI/180.0;
    double lon = lon0_deg * M_PI/180.0;
    double sL = std::sin(lat), cL = std::cos(lat);
    double sO = std::sin(lon), cO = std::cos(lon);
    // Rows of R such that v_enu = R * (v_ecef - ref_ecef)
    return cv::Matx33d(
        -sO,        cO,       0,
        -sL*cO,    -sL*sO,    cL,
         cL*cO,     cL*sO,    sL
    );
}

cv::Point3d ecef_to_enu(const cv::Point3d &ecef, const cv::Point3d &ecef0, const cv::Matx33d &R) {
    cv::Vec3d d(ecef.x - ecef0.x, ecef.y - ecef0.y, ecef.z - ecef0.z);
    cv::Vec3d e = R * d;
    return {e[0], e[1], e[2]};
}

// Umeyama similarity: src -> dst (dst ≈ s*R*src + t)
struct Similarity { double s; cv::Matx33d R; cv::Vec3d t; };

// Per-camera/point reprojection with intrinsics [fx,fy,cx,cy] and distortion [k1,k2,p1,p2,k3]
struct ReprojErrorKD {
    double u, v;
    explicit ReprojErrorKD(double u_, double v_) : u(u_), v(v_) {}
    template<typename T>
    bool operator()(const T* const cam, const T* const point, const T* const Kpar, const T* const Dpar, T* residuals) const {
        const T fx = Kpar[0], fy = Kpar[1], cx = Kpar[2], cy = Kpar[3];
        const T k1 = Dpar[0], k2 = Dpar[1], p1 = Dpar[2], p2 = Dpar[3], k3 = Dpar[4];
        T p[3] = { T(point[0]), T(point[1]), T(point[2]) };
        T pc[3];
        ceres::AngleAxisRotatePoint(cam, p, pc);
        pc[0] += cam[3]; pc[1] += cam[4]; pc[2] += cam[5];
        // normalize
        T x = pc[0] / pc[2];
        T y = pc[1] / pc[2];
        // radial + tangential distortion
        T r2 = x*x + y*y;
        T r4 = r2*r2;
        T r6 = r4*r2;
        T radial = T(1.0) + k1*r2 + k2*r4 + k3*r6;
        T x_tan = T(2.0)*p1*x*y + p2*(r2 + T(2.0)*x*x);
        T y_tan = p1*(r2 + T(2.0)*y*y) + T(2.0)*p2*x*y;
        T xd = x*radial + x_tan;
        T yd = y*radial + y_tan;
        T u_hat = fx * xd + cx;
        T v_hat = fy * yd + cy;
        residuals[0] = u_hat - T(u);
        residuals[1] = v_hat - T(v);
        return true;
    }
    static ceres::CostFunction* Create(double u, double v) {
        return new ceres::AutoDiffCostFunction<ReprojErrorKD, 2, 6, 3, 4, 5>(new ReprojErrorKD(u, v));
    }
};

// GPS prior on camera center with global similarity (r_sim[0..2], t_sim[3..5], log_s[6])
struct GpsCenterError {
    double ex, ey, ez, wx, wy, wz;
    GpsCenterError(double x, double y, double z, double wx_, double wy_, double wz_) : ex(x), ey(y), ez(z), wx(wx_), wy(wy_), wz(wz_) {}
    template<typename T>
    bool operator()(const T* const cam, const T* const sim, T* residuals) const {
        // Camera center in SFM: C = -R^T t
        T Rm[9]; ceres::AngleAxisToRotationMatrix(cam, Rm);
        T t0 = cam[3], t1 = cam[4], t2 = cam[5];
        T C0 = -(Rm[0]*t0 + Rm[3]*t1 + Rm[6]*t2);
        T C1 = -(Rm[1]*t0 + Rm[4]*t1 + Rm[7]*t2);
        T C2 = -(Rm[2]*t0 + Rm[5]*t1 + Rm[8]*t2);
        // Similarity to ENU: X_enu = s * R_sim * C + t_sim
        T Rsm[9]; ceres::AngleAxisToRotationMatrix(sim, Rsm);
        T s = ceres::exp(sim[6]);
        T Xe0 = s * (Rsm[0]*C0 + Rsm[1]*C1 + Rsm[2]*C2) + sim[3];
        T Xe1 = s * (Rsm[3]*C0 + Rsm[4]*C1 + Rsm[5]*C2) + sim[4];
        T Xe2 = s * (Rsm[6]*C0 + Rsm[7]*C1 + Rsm[8]*C2) + sim[5];
        residuals[0] = (Xe0 - T(ex)) * T(wx);
        residuals[1] = (Xe1 - T(ey)) * T(wy);
        residuals[2] = (Xe2 - T(ez)) * T(wz);
        return true;
    }
    static ceres::CostFunction* Create(double ex,double ey,double ez,double wx,double wy,double wz) {
        return new ceres::AutoDiffCostFunction<GpsCenterError, 3, 6, 7>(new GpsCenterError(ex,ey,ez,wx,wy,wz));
    }
};

// Ceres reprojection residual: camera params (angle-axis r[0..2], t[3..5]), point X[0..2]
struct ReprojError {
    double fx, fy, cx, cy; double u, v;
    ReprojError(double fx_, double fy_, double cx_, double cy_, double u_, double v_)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_), u(u_), v(v_) {}
    template<typename T>
    bool operator()(const T* const cam, const T* const point, T* residuals) const {
        T p[3] = { T(point[0]), T(point[1]), T(point[2]) };
        T pc[3];
        ceres::AngleAxisRotatePoint(cam, p, pc);
        pc[0] += cam[3]; pc[1] += cam[4]; pc[2] += cam[5];
        T u_hat = T(fx) * pc[0] / pc[2] + T(cx);
        T v_hat = T(fy) * pc[1] / pc[2] + T(cy);
        residuals[0] = u_hat - T(u);
        residuals[1] = v_hat - T(v);
        return true;
    }
    static ceres::CostFunction* Create(double fx,double fy,double cx,double cy,double u,double v) {
        return new ceres::AutoDiffCostFunction<ReprojError, 2, 6, 3>(new ReprojError(fx,fy,cx,cy,u,v));
    }
};

Similarity umeyama(const vector<cv::Point3d> &src, const vector<cv::Point3d> &dst) {
    CV_Assert(src.size() == dst.size());
    int n = (int)src.size();
    cv::Vec3d mu_s(0,0,0), mu_d(0,0,0);
    for (int i=0;i<n;++i){ mu_s += cv::Vec3d(src[i].x,src[i].y,src[i].z); mu_d += cv::Vec3d(dst[i].x,dst[i].y,dst[i].z);} 
    mu_s *= (1.0/n); mu_d *= (1.0/n);
    cv::Matx33d Sigma = cv::Matx33d::zeros();
    double var_s = 0.0;
    for (int i=0;i<n;++i){
        cv::Vec3d xs(src[i].x,src[i].y,src[i].z); xs -= mu_s;
        cv::Vec3d yd(dst[i].x,dst[i].y,dst[i].z); yd -= mu_d;
        Sigma += yd * xs.t();
        var_s += xs.dot(xs);
    }
    Sigma *= (1.0/n);
    var_s /= n;
    cv::SVD svd_obj{cv::Mat(Sigma)};
    cv::Mat U = svd_obj.u; cv::Mat Vt = svd_obj.vt; cv::Mat S = cv::Mat::diag(svd_obj.w);
    cv::Mat Rm = U * Vt;
    if (cv::determinant(Rm) < 0) {
        cv::Mat D = cv::Mat::eye(3,3,CV_64F); D.at<double>(2,2) = -1.0;
        Rm = U * D * Vt;
        S.at<double>(2,2) *= -1.0; // reflect last singular value
    }
    double trS = S.at<double>(0,0) + S.at<double>(1,1) + S.at<double>(2,2);
    double s = trS / var_s;
    cv::Vec3d t = mu_d - s * cv::Matx33d(Rm) * mu_s;
    return {s, cv::Matx33d(Rm), t};
}

string write_ply(const string &path, const vector<cv::Point3d> &pts) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return "";
    ofs << "ply\nformat ascii 1.0\n";
    ofs << "element vertex " << pts.size() << "\n";
    ofs << "property float x\nproperty float y\nproperty float z\nend_header\n";
    for (auto &p: pts) {
        ofs << (float)p.x << " " << (float)p.y << " " << (float)p.z << "\n";
    }
    return path;
}

string write_extrinsics_json(const string &path,
                             const vector<string> &images,
                             const vector<cv::Matx33d> &Rs,
                             const vector<cv::Vec3d> &ts,
                             const vector<cv::Point3d> &centers,
                             const cv::Matx33d &K) {
    std::ofstream ofs(path);
    if (!ofs) return "";
    ofs << "{\n  \"K\": [" << K(0,0) << "," << K(0,1) << "," << K(0,2)
        << "," << K(1,0) << "," << K(1,1) << "," << K(1,2)
        << "," << K(2,0) << "," << K(2,1) << "," << K(2,2) << "],\n";
    ofs << "  \"cameras\": [\n";
    for (size_t i=0;i<images.size();++i) {
        ofs << "    {\n";
        ofs << "      \"image\": \"" << images[i] << "\",\n";
        ofs << "      \"R\": [";
        for (int r=0;r<3;++r) for (int c=0;c<3;++c) {
            ofs << Rs[i](r,c) << (r==2 && c==2?"],\n":",");
        }
        ofs << "      \"t\": [" << ts[i][0] << "," << ts[i][1] << "," << ts[i][2] << "],\n";
        ofs << "      \"C\": [" << centers[i].x << "," << centers[i].y << "," << centers[i].z << "]\n";
        ofs << "    }" << (i+1<images.size()?",":"") << "\n";
    }
    ofs << "  ]\n}\n";
    return path;
}

string write_colmap_extrinsics_json(
        const string &modelDir,
        const string &jsonPath,
        const std::function<void(const std::string&)> &logI,
        const std::function<void(const std::string&)> &logE) {
    try {
        colmap::Reconstruction reconstruction;
        reconstruction.Read(modelDir);
        if (reconstruction.NumRegImages() == 0) {
            logE("COLMAP reconstruction has no registered images; skip camera JSON export");
            return "";
        }

        // Build a single intrinsic matrix K from the first camera.
        cv::Matx33d K(1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0);
        if (!reconstruction.Cameras().empty()) {
            const auto &camPair = *reconstruction.Cameras().begin();
            const colmap::Camera &cam = camPair.second;
            const double fx = cam.FocalLengthX();
            const double fy = cam.FocalLengthY();
            const double cx = cam.PrincipalPointX();
            const double cy = cam.PrincipalPointY();
            K = cv::Matx33d(
                    fx, 0.0, cx,
                    0.0, fy, cy,
                    0.0, 0.0, 1.0);
        }

        vector<string> images;
        vector<cv::Matx33d> Rs;
        vector<cv::Vec3d> ts;
        vector<cv::Point3d> centers;
        images.reserve(reconstruction.NumRegImages());
        Rs.reserve(reconstruction.NumRegImages());
        ts.reserve(reconstruction.NumRegImages());
        centers.reserve(reconstruction.NumRegImages());

        for (const auto image_id : reconstruction.RegImageIds()) {
            const colmap::Image &image = reconstruction.Image(image_id);
            if (!image.HasPose()) continue;
            images.push_back(image.Name());

            const colmap::Rigid3d &cam_from_world = image.CamFromWorld();
            const Eigen::Matrix3d R_eigen = cam_from_world.rotation.toRotationMatrix();
            const Eigen::Vector3d t_eigen = cam_from_world.translation;

            cv::Matx33d R;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    R(r, c) = R_eigen(r, c);
                }
            }
            Rs.push_back(R);
            ts.emplace_back(t_eigen(0), t_eigen(1), t_eigen(2));

            const Eigen::Vector3d C_eigen = image.ProjectionCenter();
            centers.emplace_back(C_eigen(0), C_eigen(1), C_eigen(2));
        }

        if (images.empty()) {
            logE("COLMAP reconstruction has no images with valid pose; skip camera JSON export");
            return "";
        }

        const string res = write_extrinsics_json(jsonPath, images, Rs, ts, centers, K);
        if (res.empty()) {
            logE(string("Failed writing COLMAP camera JSON to ") + jsonPath);
        } else {
            logI(string("Wrote COLMAP camera JSON: ") + jsonPath);
        }
        return res;
    } catch (const std::exception &e) {
        logE(string("Exception during COLMAP camera JSON export: ") + e.what());
    } catch (...) {
        logE("Unknown error during COLMAP camera JSON export");
    }
    return "";
}

struct Stat { double mean=0, median=0, rmse=0, p90=0; size_t n=0; };

string write_ba_stats_json(const string &path,
                           const Stat &pre,
                           const Stat &post,
                           const vector<Stat> &per_cam_post,
                           const std::array<double,4> &K_pre,
                           const std::array<double,4> &K_post,
                           const std::array<double,5> &D_pre,
                           const std::array<double,5> &D_post,
                           bool gps_used,
                           size_t gps_count,
                           double gps_rmse) {
    std::ofstream ofs(path);
    if (!ofs) return "";
    ofs << std::fixed << std::setprecision(6);
    ofs << "{\n";
    ofs << "  \"K_pre\": [" << K_pre[0] << "," << K_pre[1] << "," << K_pre[2] << "," << K_pre[3] << "],\n";
    ofs << "  \"K_post\": [" << K_post[0] << "," << K_post[1] << "," << K_post[2] << "," << K_post[3] << "],\n";
    ofs << "  \"pre\": {\"mean\": " << pre.mean << ", \"median\": " << pre.median
        << ", \"rmse\": " << pre.rmse << ", \"p90\": " << pre.p90 << ", \"n\": " << pre.n << "},\n";
    ofs << "  \"post\": {\"mean\": " << post.mean << ", \"median\": " << post.median
        << ", \"rmse\": " << post.rmse << ", \"p90\": " << post.p90 << ", \"n\": " << post.n << "},\n";
    ofs << "  \"D_pre\": [" << D_pre[0] << "," << D_pre[1] << "," << D_pre[2] << "," << D_pre[3] << "," << D_pre[4] << "],\n";
    ofs << "  \"D_post\": [" << D_post[0] << "," << D_post[1] << "," << D_post[2] << "," << D_post[3] << "," << D_post[4] << "],\n";
    ofs << "  \"per_camera\": [\n";
    for (size_t i=0;i<per_cam_post.size();++i) {
        const auto &s = per_cam_post[i];
        ofs << "    {\"index\": " << i << ", \"mean\": " << s.mean << ", \"median\": " << s.median
            << ", \"rmse\": " << s.rmse << ", \"p90\": " << s.p90 << ", \"n\": " << s.n << "}" << (i+1<per_cam_post.size()?",":"") << "\n";
    }
    ofs << "  ],\n";
    ofs << "  \"gps_used\": " << (gps_used?"true":"false") << ", \"gps_count\": " << gps_count << ", \"gps_rmse\": " << gps_rmse << "\n";
    ofs << "}\n";
    return path;
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_rebuild_1edge_SfmNative_runSfm(
        JNIEnv* env, jclass /*clazz*/, jobjectArray jImagePaths, jstring jOutDir, jboolean jAlignGps, jint jMaxLongEdge,
        jint jMode, jint jStride, jint jWindow, jint jKNeighbors) {
    std::ostringstream log;
    std::function<void()> flush; // set after outDir + writer are ready
    auto logI = [&](const std::string &msg){
        ALOGI("%s", msg.c_str());
        log << "[I] " << msg << "\n";
        if (flush) flush();
    };
    auto logE = [&](const std::string &msg){
        ALOGE("%s", msg.c_str());
        log << "[E] " << msg << "\n";
        if (flush) flush();
    };
    // Collect inputs
    vector<string> images;
    const jsize n = env->GetArrayLength(jImagePaths);
    images.reserve(n);
    for (jsize i=0;i<n;++i) {
        jstring s = (jstring)env->GetObjectArrayElement(jImagePaths, i);
        const char* c = env->GetStringUTFChars(s, nullptr);
        images.emplace_back(c);
        env->ReleaseStringUTFChars(s, c);
        env->DeleteLocalRef(s);
    }
    const char* outDirC = env->GetStringUTFChars(jOutDir, nullptr);
    string outDir(outDirC ? outDirC : "");
    env->ReleaseStringUTFChars(jOutDir, outDirC);

    string logPath = outDir + "/sfm_run.log";
    // Ensure output directory exists before any writes
    if (!mkdir_p(outDir)) {
        logE(std::string("Cannot create outDir: ") + outDir);
        string out = log.str() + "Create outDir failed\n";
        return env->NewStringUTF(out.c_str());
    }
    // Track incremental flush position to avoid rewriting whole file each time
    size_t lastFlushedLen = 0;
    // Initialize/clear log file once
    if (!outDir.empty()) {
        std::ofstream ofs_init(logPath, std::ios::trunc);
        if (!ofs_init) ALOGE("Failed to open log for init: %s", logPath.c_str());
    }
    auto write_log_to_file = [&](const string &dir){
        if (dir.empty()) return;
        const std::string sCur = log.str();
        if (sCur.size() <= lastFlushedLen) return; // nothing new
        std::ofstream ofs(logPath, std::ios::app);
        if (ofs) {
            ofs.write(sCur.data() + lastFlushedLen, static_cast<std::streamsize>(sCur.size() - lastFlushedLen));
            ofs.close();
            lastFlushedLen = sCur.size();
        } else {
            ALOGE("Failed to append to log %s", logPath.c_str());
        }
    };

    // Enable incremental flushing for UI tailing
    flush = [&](){ write_log_to_file(outDir); };

    {
        std::ostringstream ss;
        ss << "RunSfm start: " << images.size() << " images, outDir=" << outDir
           << ", alignGps=" << (jAlignGps?"true":"false")
           << ", maxLongEdge=" << (int)jMaxLongEdge
           << ", mode=" << (int)jMode
           << ", stride=" << (int)jStride
           << ", window=" << (int)jWindow
           << ", kNeighbors=" << (int)jKNeighbors;
        logI(ss.str());
    }
    // Enable OpenCV optimizations and multithreading
    try {
        cv::setUseOptimized(true);
    } catch (...) {}
    int nCpus = 1;
    try {
        nCpus = std::max(1, cv::getNumberOfCPUs());
        cv::setNumThreads(nCpus);
    } catch (...) {}
    {
        std::ostringstream ss;
        ss << "OpenCV threads: " << cv::getNumThreads() << ", CPUs detected: " << nCpus;
        logI(ss.str());
    }
    for (size_t i=0;i<images.size() && i<6; ++i) {
        std::ostringstream ss;
        ss << "image[" << i << "]=" << images[i];
        logI(ss.str());
    }

    if (images.size() < 3) {
        string msg = "Need at least 3 images";
        logE(msg);
        write_log_to_file(outDir);
        string out = log.str() + msg + "\nLog: " + logPath;
        return env->NewStringUTF(out.c_str());
    }

    // Apply optional mode-1 sampling (stride/window) to the original image list
    int mode = (int)jMode;
    int stride = std::max(1, (int)jStride);
    int win = std::max(0, (int)jWindow);
    int kNei = std::max(6, (int)jKNeighbors);
    vector<string> images_sel = images;
    if (mode == 1) {
        vector<string> tmp;
        tmp.reserve((images.size()+stride-1)/stride);
        for (size_t i=0;i<images.size(); i+= (size_t)stride) tmp.push_back(images[i]);
        if (win > 0 && (int)tmp.size() > win) {
            int n = (int)tmp.size();
            int mid = n/2;
            int start = std::max(0, std::min(mid - win/2, n - win));
            vector<string> tmp2; tmp2.reserve(win);
            for (int i=0;i<win;++i) tmp2.push_back(tmp[start + i]);
            images_sel.swap(tmp2);
        } else {
            images_sel.swap(tmp);
        }
        std::ostringstream ss;
        ss << "Mode 1: sequential-lite sampling -> " << images_sel.size() << " images (from " << images.size() << ")";
        logI(ss.str());
    }

    // If sampling reduced below 3 images, abort early
    if (images_sel.size() < 3) {
        string msg = "After sampling, need at least 3 images";
        logE(msg);
        write_log_to_file(outDir);
        string out = log.str() + msg + "\nLog: " + logPath;
        return env->NewStringUTF(out.c_str());
    }

    // Read first image to know original size
    cv::Mat firstGray = cv::imread(images_sel[0], cv::IMREAD_GRAYSCALE);
    if (firstGray.empty()) {
        string msg = "Cannot read first image";
        logE(msg);
        write_log_to_file(outDir);
        string out = log.str() + msg + "\nLog: " + logPath;
        return env->NewStringUTF(out.c_str());
    }
    int orig_w = firstGray.cols;
    int orig_h = firstGray.rows;

    // Extract intrinsics from first image XMP
    DjiMeta meta;
    if (!extract_dji_meta(images_sel[0], meta) || !meta.hasIntrinsics) {
        logE("Failed to parse DJI XMP intrinsics; using fallback intrinsics.");
        // Fallback based on loaded first image size
        meta.cx = orig_w * 0.5; meta.cy = orig_h * 0.5;
        // Assume 35mm equiv 24mm and sensor width 36mm -> fx ~ f_px = (f_mm/sensor_mm) * img_px; heuristic
        double f35 = 24.0; // fallback
        meta.fx = meta.fy = (f35/36.0) * orig_w;
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2)
               << "Fallback intrinsics: fx=" << meta.fx << " fy=" << meta.fy
               << " cx=" << meta.cx << " cy=" << meta.cy
               << " (image " << orig_w << "x" << orig_h << ")";
            logI(ss.str());
        }
    } else {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2)
           << "XMP intrinsics: fx=" << meta.fx << " fy=" << meta.fy
           << " cx=" << meta.cx << " cy=" << meta.cy
           << " (dist=" << (meta.hasDist?"yes":"no") << ")";
        logI(ss.str());
    }
    // Compute uniform scale based on requested maxLongEdge (like COLMAP's max_image_size)
    double s = 1.0;
    int maxLongEdge = (int)jMaxLongEdge;
    if (maxLongEdge > 0) {
        int long0 = std::max(orig_w, orig_h);
        if (long0 > maxLongEdge) s = (double)maxLongEdge / (double)long0;
    }

    // Scale intrinsics if downscaling
    if (s < 1.0) {
        meta.fx *= s; meta.fy *= s; meta.cx *= s; meta.cy *= s;
    }
    cv::Matx33d K(meta.fx, 0, meta.cx,
                  0, meta.fy, meta.cy,
                  0, 0, 1);
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2)
           << "K = [[" << K(0,0) << ", " << K(0,1) << ", " << K(0,2)
           << "], [" << K(1,0) << ", " << K(1,1) << ", " << K(1,2)
           << "], [" << K(2,0) << ", " << K(2,1) << ", " << K(2,2) << "]]";
        logI(ss.str());
    }

    // Optionally create uniformly downscaled copies for SFM
    std::vector<std::string> images_use;
    images_use.reserve(images_sel.size());
    // Always use original paths for metadata (EXIF/XMP) lookups to avoid loss
    // of tags when writing downscaled JPEGs without metadata.
    const std::vector<std::string> &images_meta = images_sel;
    if (s < 1.0) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4)
           << "Downscaling images by s=" << s << " to fit maxLongEdge=" << maxLongEdge << " px";
        logI(ss.str());

        std::string dsDir = outDir + "/downscaled_" + std::to_string(maxLongEdge);
        // Create directory (recursive) – best-effort
        mkdir_p(dsDir);
        for (size_t i = 0; i < images_sel.size(); ++i) {
            const std::string &srcPath = images_sel[i];
            cv::Mat src = cv::imread(srcPath, cv::IMREAD_UNCHANGED);
            if (src.empty()) {
                std::ostringstream es; es << "Failed to read image for downscale: " << srcPath;
                logE(es.str());
                continue;
            }
            int newW = (int)std::round(src.cols * s);
            int newH = (int)std::round(src.rows * s);
            if (newW < 1) newW = 1; if (newH < 1) newH = 1;
            cv::Mat dst;
            cv::resize(src, dst, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);
            // derive dest filename
            size_t pos = srcPath.find_last_of("/");
            std::string base = (pos == std::string::npos) ? srcPath : srcPath.substr(pos + 1);
            std::string dstPath = dsDir + "/" + base;
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
            if (!cv::imwrite(dstPath, dst, params)) {
                std::ostringstream es; es << "Failed to write downscaled image: " << dstPath;
                logE(es.str());
                continue;
            }
            images_use.emplace_back(dstPath);
            if (i < 3) { // brief log for first few
                std::ostringstream ls; ls << "downscaled[" << i << "]: " << src.cols << "x" << src.rows
                                          << " -> " << newW << "x" << newH << " => " << dstPath;
                logI(ls.str());
            }
        }
        if (images_use.size() != images_sel.size()) {
            logE("Warning: some images failed to downscale; proceeding with successfully written ones.");
        }
        if (images_use.empty()) {
            // fallback to originals if downscale failed
            images_use = images_sel;
            logE("Downscale produced no images, using originals.");
        }
    } else {
        images_use = images_sel;
    }

    // Branch by mode: default/libmv vs k-neighbors experimental
    vector<cv::Matx33d> Rs; 
    vector<cv::Vec3d> ts; 
    vector<cv::Point3d> pts;
    bool skipGpsAlign = false;

    if ((int)jMode == 2) {
        logI("Mode 2: sequential (COLMAP-like) with kNN + loop");
        const int n = (int)images_use.size();
        std::vector<cv::Mat> grays(n);
        for (int i=0;i<n;++i) grays[i] = cv::imread(images_use[i], cv::IMREAD_GRAYSCALE);
        auto orb = cv::ORB::create(2000, 1.2f, 8);
        std::vector<std::vector<cv::KeyPoint>> kps(n);
        std::vector<cv::Mat> desc(n);
        for (int i=0;i<n;++i) {
            if (grays[i].empty()) continue;
            const int long_edge = std::max(grays[i].cols, grays[i].rows);
            // Scale features so that long_edge=4000px -> maxFeatures=8192
            int nfeatures = (int)std::lround((double)long_edge * 8192.0 / 4000.0);
            nfeatures = std::clamp(nfeatures, 1000, 16384);
            try { orb.dynamicCast<cv::ORB>()->setMaxFeatures(nfeatures); } catch (...) {}
            orb->detectAndCompute(grays[i], cv::noArray(), kps[i], desc[i]);
        }
        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        Rs.assign(n, cv::Matx33d::eye());
        ts.assign(n, cv::Vec3d(0,0,0));
        std::vector<uchar> pose_ok(n, 0);
        pose_ok[0] = 1; // reference
        struct BAObs { int cam; int kp; cv::Point2f m; };
        std::vector<std::vector<BAObs>> point_obs; // observations per 3D point index

        // Prepare distortion (if any), and adaptive thresholds
        cv::Mat distCoeffs;
        if (meta.hasDist) {
            distCoeffs = (cv::Mat_<double>(1,5) << meta.k1, meta.k2, meta.p1, meta.p2, meta.k3);
        }
        // Slightly tighter essential matrix RANSAC threshold (in pixels)
        double thr_e = 0.8;
        // Stricter triangulation inlier threshold for ORB; reduces foggy points
        double reproj_thresh_px = 1.0;
        {
            std::ostringstream ss; ss << std::fixed << std::setprecision(2)
                << "Adaptive thresholds: E_ransac_px=" << thr_e << ", reproj_px=" << reproj_thresh_px;
            logI(ss.str());
        }

        // Pass 1 (disabled): original adjacent-pair chaining (replaced by seeded + PnP in new pipeline)
        if (false) for (int i=0;i<n-1;++i) {
            int j_adj = i+1;
            if (!desc[i].empty() && !desc[j_adj].empty()) {
                std::vector<std::vector<cv::DMatch>> knn;
                matcher.knnMatch(desc[i], desc[j_adj], knn, 2);
                std::vector<cv::Point2f> p1, p2;
                p1.reserve(knn.size()); p2.reserve(knn.size());
                for (auto &vv: knn) if (vv.size()>=2) if (vv[0].distance < 0.70f * vv[1].distance) {
                    p1.push_back(kps[i][vv[0].queryIdx].pt);
                    p2.push_back(kps[j_adj][vv[0].trainIdx].pt);
                }
                if ((int)p1.size() >= 8) {
                    // Undistort correspondence pixels if distortion available
                    // We use pixel-domain geometry consistently:
                    // undistortPoints(..., K, D, noArray(), K) -> undistorted PIXEL coords,
                    // then findEssentialMat(..., K, threshold_px) expects pixel coords.
                    if (!distCoeffs.empty()) {
                        std::vector<cv::Point2f> p1u, p2u;
                        cv::undistortPoints(p1, p1u, cv::Mat(K), distCoeffs, cv::noArray(), cv::Mat(K));
                        cv::undistortPoints(p2, p2u, cv::Mat(K), distCoeffs, cv::noArray(), cv::Mat(K));
                        p1.swap(p1u); p2.swap(p2u);
                    }
                    cv::Mat inlierMask;
                    cv::Mat E = cv::findEssentialMat(p1, p2, cv::Mat(K), cv::RANSAC, 0.999, thr_e, inlierMask);
                    if (!E.empty()) {
                        cv::Mat Rm, tm;
                        int ninl = cv::recoverPose(E, p1, p2, cv::Mat(K), Rm, tm, inlierMask);
                        if (ninl >= 8) {
                            // Compose world-to-camera transforms correctly using the
                            // relative motion from recoverPose (camera i -> camera j):
                            //   R_j = R_ij * R_i
                            //   t_j = R_ij * t_i + t_ij
                            // where X_c = R * X_w + t.
                            cv::Matx33d Rrel(Rm);
                            cv::Vec3d trel(tm.at<double>(0), tm.at<double>(1), tm.at<double>(2));
                            Rs[j_adj] = Rrel * Rs[i];
                            ts[j_adj] = Rrel * ts[i] + trel; // unit-baseline chaining
                            pose_ok[j_adj] = 1;
                            std::ostringstream ps; ps << "adjacent pair ("<<i<<","<<j_adj<<") inliers="<<ninl; logI(ps.str());
                        }
                    }
                }
            }
        }

        // Pass 2: build multi-view tracks across k-neighbors and sparse loop-closures, then triangulate
        struct DSU { std::vector<int> p, r; int add(){p.push_back((int)p.size()); r.push_back(0); return (int)p.size()-1;} int f(int x){return p[x]==x?x:p[x]=f(p[x]);} void u_(int a,int b){a=f(a); b=f(b); if(a==b) return; if(r[a]<r[b]) std::swap(a,b); p[b]=a; if(r[a]==r[b]) r[a]++;} } dsu;
        std::unordered_map<long long,int> node_id; // key = (cam<<32)|kp
        std::vector<BAObs> obs_by_id; obs_by_id.reserve(100000);
        std::vector<int> cam_obs_unique(n, 0);
        auto keyOf = [](int cam, int kp)->long long { return ( (long long)cam<<32) | (unsigned long long)kp; };
        auto ensure_node = [&](int cam, int kp, const cv::Point2f &m)->int{
            long long key = keyOf(cam,kp);
            auto it = node_id.find(key);
            if (it != node_id.end()) return it->second;
            int id = dsu.add();
            node_id.emplace(key, id);
            if ((int)obs_by_id.size() <= id) obs_by_id.resize(id+1);
            obs_by_id[id] = BAObs{cam, kp, m};
            if (cam>=0 && cam<n) ++cam_obs_unique[cam];
            return id;
        };
        struct PairScore { int i, j, ninl; double med_parallax; cv::Mat R, t; };
        std::vector<PairScore> pair_scores; pair_scores.reserve(4096);
        auto add_pair_matches = [&](int i, int j){
            if (i<0||j<0||i>=n||j>=n) return; if (i==j) return; if (desc[i].empty()||desc[j].empty()) return;
            std::vector<std::vector<cv::DMatch>> knn2; matcher.knnMatch(desc[i], desc[j], knn2, 2);
            std::vector<cv::Point2f> q1, q2; std::vector<std::pair<int,int>> idx;
            q1.reserve(knn2.size()); q2.reserve(knn2.size()); idx.reserve(knn2.size());
            for (auto &vv: knn2) if (vv.size()>=2) if (vv[0].distance < 0.70f * vv[1].distance) {
                q1.push_back(kps[i][vv[0].queryIdx].pt);
                q2.push_back(kps[j][vv[0].trainIdx].pt);
                idx.emplace_back(vv[0].queryIdx, vv[0].trainIdx);
            }
            if ((int)q1.size() < 8) return;
            // Undistort for robust inlier selection
            std::vector<cv::Point2f> uq1=q1, uq2=q2;
            if (!distCoeffs.empty()) {
                // Pixel-domain convention; undistorted coordinates remain in pixels when P=K
                cv::undistortPoints(q1, uq1, cv::Mat(K), distCoeffs, cv::noArray(), cv::Mat(K));
                cv::undistortPoints(q2, uq2, cv::Mat(K), distCoeffs, cv::noArray(), cv::Mat(K));
            }
            cv::Mat mask2;
            cv::Mat E2 = cv::findEssentialMat(uq1, uq2, cv::Mat(K), cv::RANSAC, 0.999, thr_e, mask2);
            if (E2.empty()) return;
            int ninl=0; std::vector<cv::Point2f> in1, in2; in1.reserve(q1.size()); in2.reserve(q2.size());
            for (int k=0;k<(int)mask2.total();++k) if (mask2.at<uchar>(k)) {
                int qi = idx[k].first; int qj = idx[k].second;
                // Store raw pixel observations for BA with distortion
                cv::Point2f m1 = kps[i][qi].pt;
                cv::Point2f m2 = kps[j][qj].pt;
                int id1 = ensure_node(i, qi, m1);
                int id2 = ensure_node(j, qj, m2);
                dsu.u_(id1, id2); ++ninl; in1.push_back(uq1[k]); in2.push_back(uq2[k]);
            }
            if (ninl >= 8) {
                cv::Mat Rm, tm, im; int ninl2 = cv::recoverPose(cv::findEssentialMat(in1, in2, cv::Mat(K), cv::RANSAC, 0.999, thr_e, im), in1, in2, cv::Mat(K), Rm, tm, im);
                // Compute median parallax for scoring
                std::vector<double> ang; ang.reserve(in1.size());
                for (size_t k=0;k<in1.size();++k) {
                    double x1=(in1[k].x-K(0,2))/K(0,0), y1=(in1[k].y-K(1,2))/K(1,1);
                    double x2=(in2[k].x-K(0,2))/K(0,0), y2=(in2[k].y-K(1,2))/K(1,1);
                    cv::Vec3d d1(x1,y1,1.0); d1/=std::sqrt(d1.dot(d1)); cv::Vec3d d2(x2,y2,1.0); d2/=std::sqrt(d2.dot(d2));
                    double c = std::clamp(d1.dot(d2), -1.0, 1.0); ang.push_back(std::acos(c)*180.0/M_PI);
                }
                double med = 0.0; if (!ang.empty()) { std::sort(ang.begin(), ang.end()); med = ang[ang.size()/2]; }
                pair_scores.push_back(PairScore{i,j,ninl2>0?ninl2:ninl, med, Rm, tm});
            }
        };

        // Build pairs: symmetric k-neighbors window (forward only to avoid duplicates) plus sparse loop closures
        std::unordered_set<long long> pairset;
        auto pairkey = [](int a,int b)->long long{ if (a>b) std::swap(a,b); return ((long long)a<<32) | (unsigned long long)b; };
        for (int i=0;i<n-1;++i) {
            // Encourage larger neighborhood to increase parallax across pairs
            int kNei = std::max(6, (int)jKNeighbors);
            int j0 = std::max(0, i - kNei);
            int j1 = std::min(n-1, i + kNei);
            for (int j=i+1; j<=j1; ++j) {
                long long pk = pairkey(i,j); if (pairset.insert(pk).second) add_pair_matches(i,j);
            }
        }
        if (n >= 4) {
            int stride = std::max(1, n/12); // ~12 closures per sequence
            for (int i=0;i<n; i+=stride) {
                long long pk1 = pairkey(i, 0); if (pairset.insert(pk1).second) add_pair_matches(i, 0);
                long long pk2 = pairkey(i, n-1); if (pairset.insert(pk2).second) add_pair_matches(i, n-1);
            }
        }

        // Build DSU groups and image_nodes index for later triangulation / PnP
        std::unordered_map<int, std::vector<int>> groups; groups.reserve(node_id.size());
        for (const auto &kv : node_id) {
            int id = kv.second; int root = dsu.f(id); groups[root].push_back(id);
        }
        std::vector<std::vector<int>> image_nodes(n); // per image, list of node ids
        for (const auto &gkv : groups) for (int id : gkv.second) {
            const auto &o = obs_by_id[id]; if (o.cam>=0 && o.cam<n) image_nodes[o.cam].push_back(id);
        }
        // Track root -> 3D point index and per-image kp -> point index map
        std::unordered_map<int,int> track2pt; track2pt.reserve(groups.size());
        std::vector<std::unordered_map<int,int>> kp_to_pt(n);

        // Select a good seed pair (best by inliers then parallax), initialize poses, and register others by PnP
        int seed_a=-1, seed_b=-1; cv::Mat Rseed, tseed;
        int best_inl = -1; double best_par = -1.0;
        for (const auto &ps : pair_scores) {
            if (ps.ninl > best_inl || (ps.ninl == best_inl && ps.med_parallax > best_par)) {
                best_inl = ps.ninl; best_par = ps.med_parallax; seed_a = ps.i; seed_b = ps.j; Rseed = ps.R.clone(); tseed = ps.t.clone();
            }
        }
        if (seed_a < 0 || seed_b < 0) {
            seed_a = 0; seed_b = std::min(1, n-1);
            Rseed = cv::Mat::eye(3,3,CV_64F); tseed = (cv::Mat_<double>(3,1) << 1,0,0);
            logE("No validated seed pair found; fallback to (0,1)");
        }
        {
            std::ostringstream ss; ss << "Seed ("<<seed_a<<","<<seed_b<<") ninl="<<best_inl<<", parallax_med_deg="<<std::setprecision(2)<<best_par; logI(ss.str());
        }
        // Initialize poses for seed pair
        Rs[seed_a] = cv::Matx33d::eye(); ts[seed_a] = cv::Vec3d(0,0,0);
        if (!Rseed.empty() && !tseed.empty()) { Rs[seed_b] = cv::Matx33d(Rseed); ts[seed_b] = cv::Vec3d(tseed.at<double>(0),tseed.at<double>(1),tseed.at<double>(2)); }
        pose_ok.assign(n, 0); pose_ok[seed_a] = 1; pose_ok[seed_b] = 1;

        // Minimal two-view triangulation from seed pair inliers to bootstrap PnP
        auto match_and_triangulate_seed = [&](int a, int b){
            if (desc[a].empty() || desc[b].empty()) return;
            std::vector<std::vector<cv::DMatch>> knn; matcher.knnMatch(desc[a], desc[b], knn, 2);
            std::vector<cv::Point2f> p1, p2; std::vector<std::pair<int,int>> ij; p1.reserve(knn.size()); p2.reserve(knn.size()); ij.reserve(knn.size());
            for (auto &vv : knn) if (vv.size()>=2 && vv[0].distance < 0.75f * vv[1].distance) { p1.push_back(kps[a][vv[0].queryIdx].pt); p2.push_back(kps[b][vv[0].trainIdx].pt); ij.emplace_back(vv[0].queryIdx, vv[0].trainIdx); }
            if ((int)p1.size() < 8) return;
            std::vector<cv::Point2f> u1=p1,u2=p2; if (!distCoeffs.empty()) { cv::undistortPoints(p1,u1,cv::Mat(K),distCoeffs,cv::noArray(),cv::Mat(K)); cv::undistortPoints(p2,u2,cv::Mat(K),distCoeffs,cv::noArray(),cv::Mat(K)); }
            cv::Mat inl; cv::Mat E = cv::findEssentialMat(u1,u2,cv::Mat(K),cv::RANSAC,0.999,thr_e,inl); if (E.empty()) return;
            std::vector<int> inlier_idx; for (int k=0;k<(int)inl.total();++k) if (inl.at<uchar>(k)) inlier_idx.push_back(k);
            // Triangulate inliers
            for (int idc : inlier_idx) {
                int ia = ij[idc].first, ib = ij[idc].second; cv::Point2f ma = kps[a][ia].pt, mb = kps[b][ib].pt;
                cv::Matx34d P1, P2; cv::sfm::projectionFromKRt(K, Rs[a], ts[a], P1); cv::sfm::projectionFromKRt(K, Rs[b], ts[b], P2);
                std::vector<cv::Point2f> ina{ma}, inb{mb}; if (!distCoeffs.empty()) { cv::undistortPoints(ina, ina, cv::Mat(K), distCoeffs, cv::noArray(), cv::Mat(K)); cv::undistortPoints(inb, inb, cv::Mat(K), distCoeffs, cv::noArray(), cv::Mat(K)); }
                cv::Mat A(4,4,CV_64F);
                auto fill=[&](int r,const cv::Matx34d&P,const cv::Point2f&u){ A.at<double>(r+0,0)=u.x*P(2,0)-P(0,0); A.at<double>(r+0,1)=u.x*P(2,1)-P(0,1); A.at<double>(r+0,2)=u.x*P(2,2)-P(0,2); A.at<double>(r+0,3)=u.x*P(2,3)-P(0,3); A.at<double>(r+1,0)=u.y*P(2,0)-P(1,0); A.at<double>(r+1,1)=u.y*P(2,1)-P(1,1); A.at<double>(r+1,2)=u.y*P(2,2)-P(1,2); A.at<double>(r+1,3)=u.y*P(2,3)-P(1,3); };
                fill(0,P1,ina[0]); fill(2,P2,inb[0]); cv::SVD sv(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV); cv::Mat Vt=sv.vt; cv::Vec4d Xh(Vt.at<double>(3,0),Vt.at<double>(3,1),Vt.at<double>(3,2),Vt.at<double>(3,3)); if (std::abs(Xh[3])<1e-12) continue; cv::Vec3d X(Xh[0]/Xh[3],Xh[1]/Xh[3],Xh[2]/Xh[3]);
                cv::Vec3d Xa = Rs[a]*X + ts[a]; cv::Vec3d Xb = Rs[b]*X + ts[b]; if (Xa[2]<=0 || Xb[2]<=0) continue; cv::Point2f pa(K(0,0)*(Xa[0]/Xa[2])+K(0,2), K(1,1)*(Xa[1]/Xa[2])+K(1,2)); cv::Point2f pb(K(0,0)*(Xb[0]/Xb[2])+K(0,2), K(1,1)*(Xb[1]/Xb[2])+K(1,2)); if (cv::norm(pa-ma)>2.0 || cv::norm(pb-mb)>2.0) continue;
                int pid = (int)pts.size(); pts.emplace_back(X[0],X[1],X[2]); point_obs.push_back({}); point_obs.back().push_back(BAObs{a, ia, ma}); point_obs.back().push_back(BAObs{b, ib, mb});
                // map DSU root and per-image kp to this new 3D point if present in DSU
                auto it1 = node_id.find(((long long)a<<32) | (unsigned long long)ia);
                auto it2 = node_id.find(((long long)b<<32) | (unsigned long long)ib);
                if (it1 != node_id.end() && it2 != node_id.end()) { int root = dsu.f(it1->second); track2pt[root] = pid; }
                kp_to_pt[a][ia] = pid; kp_to_pt[b][ib] = pid;
            }
            std::ostringstream ss; ss << "Seed bootstrap points: " << point_obs.size(); logI(ss.str());
        };
        match_and_triangulate_seed(seed_a, seed_b);

        // Initialize per-image (kp -> 3D) mapping from seed bootstrap
        for (int pidx=0; pidx<(int)point_obs.size(); ++pidx) {
            for (const auto &o : point_obs[pidx]) { if (o.cam>=0 && o.cam<n) kp_to_pt[o.cam][o.kp] = pidx; }
        }

        // Triangulate DSU tracks that involve a given camera and at least one already-posed neighbor
        auto triangulate_tracks_with_cam = [&](int newCam)->int{
            int added = 0;
            if (newCam<0 || newCam>=n) return 0;
            for (int id : image_nodes[newCam]) {
                int root = dsu.f(id); if (track2pt.find(root) != track2pt.end()) continue; // already has 3D
                // gather candidate neighbor obs in registered cams
                int bestCam = -1; double bestAng = 0.0; int idBest = -1;
                for (int id2 : groups[root]) {
                    const auto &o2 = obs_by_id[id2]; if (o2.cam == newCam) continue; if (!pose_ok[o2.cam]) continue;
                    // approximate parallax angle between world rays
                    cv::Point2f m1 = obs_by_id[id].m; cv::Point2f m2 = o2.m;
                    // undistort to pixel domain consistent with K
                    std::vector<cv::Point2f> in1{m1}, in2{m2}; if (!distCoeffs.empty()) { cv::undistortPoints(in1, in1, cv::Mat(K), distCoeffs, cv::noArray(), cv::Mat(K)); cv::undistortPoints(in2, in2, cv::Mat(K), distCoeffs, cv::noArray(), cv::Mat(K)); }
                    cv::Vec3d d1((in1[0].x-K(0,2))/K(0,0),(in1[0].y-K(1,2))/K(1,1),1.0); d1/=std::sqrt(d1.dot(d1));
                    cv::Vec3d d2((in2[0].x-K(0,2))/K(0,0),(in2[0].y-K(1,2))/K(1,1),1.0); d2/=std::sqrt(d2.dot(d2));
                    cv::Vec3d w1 = Rs[newCam].t()*d1; w1/=std::sqrt(w1.dot(w1)); cv::Vec3d w2 = Rs[o2.cam].t()*d2; w2/=std::sqrt(w2.dot(w2));
                    double ang = std::acos(std::clamp(w1.dot(w2), -1.0, 1.0)) * 180.0/M_PI;
                    if (ang > bestAng) { bestAng=ang; bestCam=o2.cam; idBest=id2; }
                }
                if (bestCam < 0 || bestAng < 2.0) continue; // require minimal parallax
                // two-view triangulate newCam vs bestCam
                const auto &o1 = obs_by_id[id]; const auto &o2 = obs_by_id[idBest];
                cv::Matx34d P1, P2; cv::sfm::projectionFromKRt(K, Rs[newCam], ts[newCam], P1); cv::sfm::projectionFromKRt(K, Rs[bestCam], ts[bestCam], P2);
                std::vector<cv::Point2f> u1{ o1.m }, u2{ o2.m };
                if (!distCoeffs.empty()) { cv::undistortPoints(u1, u1, cv::Mat(K), distCoeffs, cv::noArray(), cv::Mat(K)); cv::undistortPoints(u2, u2, cv::Mat(K), distCoeffs, cv::noArray(), cv::Mat(K)); }
                cv::Mat A(4,4,CV_64F);
                auto fill=[&](int r,const cv::Matx34d&P,const cv::Point2f&u){ A.at<double>(r+0,0)=u.x*P(2,0)-P(0,0); A.at<double>(r+0,1)=u.x*P(2,1)-P(0,1); A.at<double>(r+0,2)=u.x*P(2,2)-P(0,2); A.at<double>(r+0,3)=u.x*P(2,3)-P(0,3); A.at<double>(r+1,0)=u.y*P(2,0)-P(1,0); A.at<double>(r+1,1)=u.y*P(2,1)-P(1,1); A.at<double>(r+1,2)=u.y*P(2,2)-P(1,2); A.at<double>(r+1,3)=u.y*P(2,3)-P(1,3); };
                fill(0,P1,u1[0]); fill(2,P2,u2[0]); cv::SVD sv(A, cv::SVD::MODIFY_A|cv::SVD::FULL_UV); cv::Mat Vt=sv.vt; cv::Vec4d Xh(Vt.at<double>(3,0),Vt.at<double>(3,1),Vt.at<double>(3,2),Vt.at<double>(3,3)); if (std::abs(Xh[3])<1e-12) continue; cv::Vec3d X(Xh[0]/Xh[3],Xh[1]/Xh[3],Xh[2]/Xh[3]);
                cv::Vec3d X1 = Rs[newCam]*X + ts[newCam]; cv::Vec3d X2 = Rs[bestCam]*X + ts[bestCam]; if (X1[2]<=0 || X2[2]<=0) continue;
                auto proj = [&](int ci)->cv::Point2f{ cv::Vec3d Xc = Rs[ci]*X + ts[ci]; return cv::Point2f((float)(K(0,0)*(Xc[0]/Xc[2])+K(0,2)), (float)(K(1,1)*(Xc[1]/Xc[2])+K(1,2))); };
                if (cv::norm(proj(newCam) - o1.m) > 1.5 || cv::norm(proj(bestCam) - o2.m) > 1.5) continue;
                int pidx = (int)pts.size(); pts.emplace_back(X[0],X[1],X[2]); point_obs.push_back({});
                // attach observations for this root from all registered cameras
                for (int id2 : groups[root]) { const auto &o = obs_by_id[id2]; if (!pose_ok[o.cam]) continue; point_obs.back().push_back(o); kp_to_pt[o.cam][o.kp] = pidx; }
                track2pt[root] = pidx; ++added;
            }
            return added;
        };

        // Local BA on a small window around a newly registered camera
        auto run_local_ba = [&](int centerIdx){
            int win = 6; int s = std::max(0, centerIdx - win/2); int e = std::min(n-1, s + win - 1); s = std::max(0, e - win + 1);
            std::vector<int> cams_idx; for (int i=s;i<=e;++i) if (pose_ok[i]) cams_idx.push_back(i);
            if ((int)cams_idx.size() < 2) return; // nothing to optimize
            // build parameter arrays
            std::vector<double> cams(6*n, 0.0);
            for (int ci=0; ci<n; ++ci) { cv::Mat rvec; cv::Rodrigues(cv::Mat(Rs[ci]), rvec); cams[6*ci+0]=rvec.at<double>(0); cams[6*ci+1]=rvec.at<double>(1); cams[6*ci+2]=rvec.at<double>(2); cams[6*ci+3]=ts[ci][0]; cams[6*ci+4]=ts[ci][1]; cams[6*ci+5]=ts[ci][2]; }
            std::vector<double> Xs(3*pts.size(), 0.0); for (size_t k=0;k<pts.size();++k){ Xs[3*k+0]=pts[k].x; Xs[3*k+1]=pts[k].y; Xs[3*k+2]=pts[k].z; }
            std::array<double,4> Kpar{{K(0,0),K(1,1),K(0,2),K(1,2)}}; std::array<double,5> Dpar{{0,0,0,0,0}}; if (meta.hasDist) Dpar = std::array<double,5>{{meta.k1,meta.k2,meta.p1,meta.p2,meta.k3}};
            ceres::Problem problem; ceres::LossFunction* loss = new ceres::CauchyLoss(2.0);
            std::vector<char> cam_included(n, 0); for (int ci : cams_idx) cam_included[ci]=1;
            for (size_t k=0;k<pts.size() && k<point_obs.size(); ++k) {
                for (const auto &o : point_obs[k]) { if (!cam_included[o.cam]) continue; ceres::CostFunction* cost = ReprojErrorKD::Create(o.m.x, o.m.y); problem.AddResidualBlock(cost, loss, &cams[6*o.cam], &Xs[3*k], Kpar.data(), Dpar.data()); }
            }
            if (problem.NumResidualBlocks() == 0) return;
            problem.SetParameterBlockConstant(Kpar.data()); problem.SetParameterBlockConstant(Dpar.data());
            problem.SetParameterBlockConstant(&cams[6*cams_idx.front()]); // anchor
            ceres::Solver::Options opts; opts.linear_solver_type = ceres::SPARSE_SCHUR; opts.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE; opts.minimizer_progress_to_stdout=false; opts.max_num_iterations=50; opts.num_threads=std::max(1, cv::getNumThreads());
            ceres::Solver::Summary summary; ceres::Solve(opts, &problem, &summary);
            // unpack
            for (int ci : cams_idx) { cv::Mat rvec = (cv::Mat_<double>(3,1) << cams[6*ci+0], cams[6*ci+1], cams[6*ci+2]); cv::Mat Rm; cv::Rodrigues(rvec, Rm); Rs[ci] = cv::Matx33d(Rm); ts[ci] = cv::Vec3d(cams[6*ci+3], cams[6*ci+4], cams[6*ci+5]); }
            for (size_t k=0;k<pts.size();++k) { pts[k] = cv::Point3d(Xs[3*k+0], Xs[3*k+1], Xs[3*k+2]); }
        };
        auto try_register_pnp = [&](int idx)->bool{
            if (pose_ok[idx]) return false;
            std::vector<cv::Point3f> X; std::vector<cv::Point2f> uv;
            // match with all posed cameras to collect 2D-3D
            for (int j=0;j<n;++j) if (pose_ok[j] && !desc[idx].empty() && !desc[j].empty()) {
                std::vector<std::vector<cv::DMatch>> kn; matcher.knnMatch(desc[idx], desc[j], kn, 2);
                for (auto &vv : kn) if (vv.size()>=2 && vv[0].distance < 0.75f*vv[1].distance) {
                    int qi = vv[0].queryIdx; int qj = vv[0].trainIdx; auto it = kp_to_pt[j].find(qj); if (it == kp_to_pt[j].end()) continue; int pidx = it->second; X.emplace_back((float)pts[pidx].x,(float)pts[pidx].y,(float)pts[pidx].z); uv.push_back(kps[idx][qi].pt);
                }
            }
            if ((int)X.size() < 30) return false;
            cv::Mat rvec, tvec; std::vector<int> inliers;
            bool ok = cv::solvePnPRansac(X, uv, cv::Mat(K), distCoeffs, rvec, tvec, false, 1000, 3.0, 0.999, inliers, cv::SOLVEPNP_AP3P);
            if (!ok || (int)inliers.size() < 20) return false;
            cv::Mat Rm; cv::Rodrigues(rvec, Rm); Rs[idx] = cv::Matx33d(Rm); ts[idx] = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)); pose_ok[idx] = 1; std::ostringstream ss; ss << "PnP-registered cam #"<<idx<<" with inliers="<<inliers.size(); logI(ss.str());
            int added = triangulate_tracks_with_cam(idx);
            if (added > 0) run_local_ba(idx);
            // attempt a second triangulation after BA
            int added2 = triangulate_tracks_with_cam(idx);
            if (added + added2 > 0) run_local_ba(idx);
            return true;
        };
        // Greedy passes along filename order until no progress
        bool pnp_progress=true; int pnp_pass=0; while (pnp_progress && pnp_pass<3) { pnp_progress=false; ++pnp_pass; for (int i=0;i<n;++i) if (!pose_ok[i]) { if (try_register_pnp(i)) pnp_progress=true; } }

        std::vector<uchar> use_cam(n, 0);
        int disabled_low_support = 0;
        for (int ci=0; ci<n; ++ci) {
            if (pose_ok[ci] && cam_obs_unique[ci] >= 20) use_cam[ci] = 1; else if (pose_ok[ci]) ++disabled_low_support;
        }
        {
            std::ostringstream ss; ss << "Camera unique obs counts:";
            for (int ci=0; ci<n; ++ci) if (pose_ok[ci]) {
                ss << " " << ci << ":" << cam_obs_unique[ci] << (use_cam[ci]?"":"*");
            }
            logI(ss.str());
            if (disabled_low_support > 0) {
                std::ostringstream ss2; ss2 << "Disable " << disabled_low_support << " low-support cameras (<20 tracks) for triang/BA";
                logI(ss2.str());
            }
        }

        // groups already built above
        size_t n_tracks = 0, n_triangulated = 0;
        auto undistort_pixel = [&](const cv::Point2f& m)->cv::Point2f {
            if (distCoeffs.empty()) return m;
            std::vector<cv::Point2f> in{m}, out;
            cv::undistortPoints(in, out, cv::Mat(K), distCoeffs, cv::noArray(), cv::Mat(K));
            return out[0];
        };
        auto proj_px_nodist = [&](int camIdx, const cv::Vec3d &X)->cv::Point2f {
            const auto &R = Rs[camIdx]; const auto &t = ts[camIdx];
            cv::Vec3d Xc = R * X + t;
            double u = K(0,0) * (Xc[0]/Xc[2]) + K(0,2);
            double v = K(1,1) * (Xc[1]/Xc[2]) + K(1,2);
            return cv::Point2f((float)u,(float)v);
        };
        // Parallax-based filtering: require minimum viewing ray angle to avoid far-away "fog" points
        const double min_parallax_deg = 3.0; // reject tracks with too little baseline
        const double cos_min_parallax = std::cos(min_parallax_deg * M_PI / 180.0);
        for (auto &gkv : groups) {
            const std::vector<int> &nodes = gkv.second; if (nodes.size() < 2) continue; ++n_tracks;
            // skip if already triangulated earlier in incremental loop
            if (track2pt.find(gkv.first) != track2pt.end()) continue;
            // Collect observations in cameras with known pose
            std::vector<BAObs> obs; obs.reserve(nodes.size());
            for (int id : nodes) {
                BAObs o = obs_by_id[id]; if (o.cam>=0 && o.cam<n && pose_ok[o.cam] && use_cam[o.cam]) obs.push_back(o);
            }
            // Require at least 3-view support for stable triangulation
            if (obs.size() < 3) continue;
            // Quick parallax check based on viewing ray angles in world frame
            bool low_parallax = false;
            {
                // Precompute world-frame unit ray directions for each obs
                std::vector<cv::Vec3d> rays; rays.reserve(obs.size());
                for (const auto &o : obs) {
                    // undistorted pixel -> normalized cam coords
                    cv::Point2f mu = undistort_pixel(o.m);
                    double x = (mu.x - K(0,2)) / K(0,0);
                    double y = (mu.y - K(1,2)) / K(1,1);
                    cv::Vec3d dc(x, y, 1.0);
                    dc *= 1.0 / std::sqrt(dc.dot(dc));
                    // world direction = R^T * dc
                    cv::Vec3d dw = Rs[o.cam].t() * dc;
                    dw *= 1.0 / std::sqrt(dw.dot(dw));
                    rays.push_back(dw);
                }
                double best_cos = 1.0;
                for (size_t a=0;a<rays.size();++a) for (size_t b=a+1;b<rays.size();++b) {
                    double c = std::abs(rays[a].dot(rays[b])); // angle between rays
                    if (c < best_cos) best_cos = c;
                }
                if (best_cos > cos_min_parallax) low_parallax = true; // all angles smaller than threshold
            }
            if (low_parallax) continue;
            // Triangulate by DLT with all obs
            int m = (int)obs.size();
            cv::Mat A(2*m, 4, CV_64F);
            int row = 0;
            for (const auto &o : obs) {
                cv::Matx34d P; cv::sfm::projectionFromKRt(K, Rs[o.cam], ts[o.cam], P);
                for (int c=0;c<4;++c) {
                    cv::Point2f mu = undistort_pixel(o.m);
                    A.at<double>(row+0, c) = mu.x * P(2,c) - P(0,c);
                    A.at<double>(row+1, c) = mu.y * P(2,c) - P(1,c);
                }
                row += 2;
            }
            cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
            cv::Mat Vt = svd.vt;
            cv::Vec4d Xh(Vt.at<double>(3,0), Vt.at<double>(3,1), Vt.at<double>(3,2), Vt.at<double>(3,3));
            if (std::abs(Xh[3]) <= 1e-12) continue;
            cv::Vec3d X(Xh[0]/Xh[3], Xh[1]/Xh[3], Xh[2]/Xh[3]);
            // Cheirality: positive depth in at least two views
            int pos_depth = 0; for (const auto &o : obs) { cv::Vec3d Xc = Rs[o.cam]*X + ts[o.cam]; if (Xc[2] > 0) ++pos_depth; }
            if (pos_depth < 2) continue;
            std::vector<double> errs; errs.reserve(obs.size());
            for (const auto &o : obs) {
                cv::Point2f mu = undistort_pixel(o.m);
                cv::Point2f pred = proj_px_nodist(o.cam, X);
                double du = pred.x - mu.x, dv = pred.y - mu.y;
                errs.push_back(std::sqrt(du*du + dv*dv));
            }
            if (errs.size() < 2) continue; std::sort(errs.begin(), errs.end());
            double med = errs[errs.size()/2]; if (med > reproj_thresh_px) continue;
            pts.emplace_back(X[0], X[1], X[2]);
            point_obs.push_back({}); point_obs.back().reserve(obs.size());
            for (const auto &o : obs) point_obs.back().push_back(o);
            ++n_triangulated;
        }
        {
            std::ostringstream ss; ss << "Tracks built=" << n_tracks << ", triangulated=" << n_triangulated;
            logI(ss.str());
        }

        // Current K,D used for stats projection (updated after BA)
        std::array<double,4> currK{{K(0,0), K(1,1), K(0,2), K(1,2)}};
        std::array<double,5> currD{{0,0,0,0,0}};
        if (meta.hasDist) { currD = std::array<double,5>{{meta.k1, meta.k2, meta.p1, meta.p2, meta.k3}}; }
        auto proj_pixel = [&](int camIdx, const cv::Vec3d &X)->cv::Point2f{
            const cv::Matx33d &R = Rs[camIdx];
            const cv::Vec3d &t = ts[camIdx];
            cv::Vec3d Xc = R * X + t;
            double x = Xc[0]/Xc[2], y = Xc[1]/Xc[2];
            double r2 = x*x + y*y; double r4 = r2*r2; double r6 = r4*r2;
            double radial = 1.0 + currD[0]*r2 + currD[1]*r4 + currD[4]*r6;
            double x_tan = 2.0*currD[2]*x*y + currD[3]*(r2 + 2.0*x*x);
            double y_tan = currD[2]*(r2 + 2.0*y*y) + 2.0*currD[3]*x*y;
            double xd = x*radial + x_tan;
            double yd = y*radial + y_tan;
            double u = currK[0]*xd + currK[2];
            double v = currK[1]*yd + currK[3];
            return cv::Point2f((float)u, (float)v);
        };

        auto compute_reproj_stats = [&](const std::vector<cv::Point3d> &pts3d,
                                        const std::vector<std::vector<BAObs>> &obs)->std::tuple<double,double,double,double,size_t>{
            std::vector<double> errs; errs.reserve(1024);
            size_t skipped_nonfinite = 0;
            for (size_t k=0;k<pts3d.size() && k<obs.size(); ++k) {
                const auto &P = pts3d[k];
                if (!std::isfinite(P.x) || !std::isfinite(P.y) || !std::isfinite(P.z)) { ++skipped_nonfinite; continue; }
                cv::Vec3d X(P.x, P.y, P.z);
                for (const auto &o : obs[k]) {
                    cv::Point2f pred = proj_pixel(o.cam, X);
                    double du = pred.x - o.m.x; double dv = pred.y - o.m.y;
                    double e = std::sqrt(du*du + dv*dv);
                    if (std::isfinite(e)) errs.push_back(e); else ++skipped_nonfinite;
                }
            }
            if (errs.empty()) return {0,0,0,0,0};
            double sum=0, sq=0; for (double e: errs){ sum+=e; sq+=e*e; }
            std::sort(errs.begin(), errs.end());
            double mean = sum / errs.size();
            double rmse = std::sqrt(sq / errs.size());
            double median = errs[errs.size()/2];
            double p90 = (errs.size() < 10) ? errs.back() : errs[(size_t)std::floor(0.9*(errs.size()-1))];
            if (p90 < median) p90 = median;
            return {mean, median, rmse, p90, errs.size()};
        };

        // Global BA with Ceres: optimize cameras, points, and intrinsics; optional GPS prior via similarity
        if (!point_obs.empty()) {
            auto [mean0, med0, rmse0, p900, nobs0] = compute_reproj_stats(pts, point_obs);
            {
                std::ostringstream ss; ss << std::fixed << std::setprecision(3)
                    << "Pre-BA reproj (px): mean=" << mean0 << ", median=" << med0
                    << ", rmse=" << rmse0 << ", p90=" << p900 << " (n=" << nobs0 << ")";
                logI(ss.str());
            }
            Stat preStat{mean0, med0, rmse0, p900, nobs0};

            // Pack parameters
            std::vector<double> cams(6*n);
            for (int ci=0; ci<n; ++ci) {
                cv::Mat rvec; cv::Rodrigues(cv::Mat(Rs[ci]), rvec);
                cams[6*ci + 0] = rvec.at<double>(0);
                cams[6*ci + 1] = rvec.at<double>(1);
                cams[6*ci + 2] = rvec.at<double>(2);
                cams[6*ci + 3] = ts[ci][0];
                cams[6*ci + 4] = ts[ci][1];
                cams[6*ci + 5] = ts[ci][2];
            }
            const int npts = (int)pts.size();
            std::vector<double> Xs(3*npts);
            for (int k=0;k<npts;++k) { Xs[3*k+0]=pts[k].x; Xs[3*k+1]=pts[k].y; Xs[3*k+2]=pts[k].z; }

            // Intrinsics parameter block [fx, fy, cx, cy] and distortion [k1,k2,p1,p2,k3]
            std::array<double,4> Kpre{{K(0,0), K(1,1), K(0,2), K(1,2)}};
            std::array<double,4> Kpar = Kpre;
            std::array<double,5> Dpre{{0,0,0,0,0}};
            if (meta.hasDist) Dpre = std::array<double,5>{{meta.k1, meta.k2, meta.p1, meta.p2, meta.k3}};
            std::array<double,5> Dpar = Dpre;

            ceres::Problem problem;
            ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);
            for (int k=0; k<npts && k<(int)point_obs.size(); ++k) {
                for (const auto &o : point_obs[k]) {
                    ceres::CostFunction* cost = ReprojErrorKD::Create((double)o.m.x, (double)o.m.y);
                    problem.AddResidualBlock(cost, loss, &cams[6*o.cam], &Xs[3*k], Kpar.data(), Dpar.data());
                }
            }
            // Camera observation counts per index (for anchor selection)
            std::vector<int> cam_obs_count(n, 0);
            for (const auto &olist : point_obs) {
                for (const auto &o : olist) if (o.cam >= 0 && o.cam < n) ++cam_obs_count[o.cam];
            }

            // Phase-1 stability: fix intrinsics and distortion constants to avoid gauge/degeneracy blowups.
            if (problem.HasParameterBlock(Kpar.data())) {
                problem.SetParameterBlockConstant(Kpar.data());
            }
            if (problem.HasParameterBlock(Dpar.data())) {
                problem.SetParameterBlockConstant(Dpar.data());
            }

            // Set bounds (effective if later we allow them to vary)
            if (problem.HasParameterBlock(Kpar.data())) {
                // fx, fy positive with sane range around initial
                double fx0 = Kpre[0], fy0 = Kpre[1], cx0 = Kpre[2], cy0 = Kpre[3];
                problem.SetParameterLowerBound(Kpar.data(), 0, 0.5 * std::max(1e-3, fx0));
                problem.SetParameterUpperBound(Kpar.data(), 0, 2.0 * std::max(1e-3, fx0));
                problem.SetParameterLowerBound(Kpar.data(), 1, 0.5 * std::max(1e-3, fy0));
                problem.SetParameterUpperBound(Kpar.data(), 1, 2.0 * std::max(1e-3, fy0));
                // Keep principal point near original (±10 px)
                problem.SetParameterLowerBound(Kpar.data(), 2, cx0 - 10.0);
                problem.SetParameterUpperBound(Kpar.data(), 2, cx0 + 10.0);
                problem.SetParameterLowerBound(Kpar.data(), 3, cy0 - 10.0);
                problem.SetParameterUpperBound(Kpar.data(), 3, cy0 + 10.0);
            }
            if (problem.HasParameterBlock(Dpar.data())) {
                // Tighter, per-parameter distortion bounds for stability
                problem.SetParameterLowerBound(Dpar.data(), 0, -0.5);  // k1
                problem.SetParameterUpperBound(Dpar.data(), 0,  0.5);
                problem.SetParameterLowerBound(Dpar.data(), 1, -0.5);  // k2
                problem.SetParameterUpperBound(Dpar.data(), 1,  0.5);
                problem.SetParameterLowerBound(Dpar.data(), 4, -0.5);  // k3
                problem.SetParameterUpperBound(Dpar.data(), 4,  0.5);
                problem.SetParameterLowerBound(Dpar.data(), 2, -0.02); // p1
                problem.SetParameterUpperBound(Dpar.data(), 2,  0.02);
                problem.SetParameterLowerBound(Dpar.data(), 3, -0.02); // p2
                problem.SetParameterUpperBound(Dpar.data(), 3,  0.02);
            }

            // Optional GPS prior via similarity
            bool used_gps_transform = false;
            std::array<double,7> sim{{0,0,0, 0,0,0, 0}};
            std::vector<cv::Point3d> enu_gps(n);
            std::vector<char> has_gps(n, 0);
            if (jAlignGps) {
                int refIdx = -1; for (int i=0;i<n;++i) { DjiMeta mx; if (extract_dji_meta(images_meta[i], mx) && mx.hasGeo) { refIdx=i; break; } }
                if (refIdx >= 0) {
                    DjiMeta mref; extract_dji_meta(images_meta[refIdx], mref);
                    cv::Point3d ecef0 = llh_to_ecef(mref.lat, mref.lon, mref.alt);
                    cv::Matx33d Renu = ecef_to_enu_R(mref.lat, mref.lon);
                    for (int i=0;i<n;++i) { DjiMeta mi; if (extract_dji_meta(images_meta[i], mi) && mi.hasGeo) { cv::Point3d ecef = llh_to_ecef(mi.lat, mi.lon, mi.alt); enu_gps[i]=ecef_to_enu(ecef,ecef0,Renu); has_gps[i]=1; } }
                    std::vector<cv::Point3d> centers_init; centers_init.reserve(n);
                    for (int i=0;i<n;++i){ cv::Matx33d Rm=Rs[i]; cv::Vec3d t=ts[i]; cv::Vec3d C=-Rm.t()*t; centers_init.emplace_back(C[0],C[1],C[2]); }
                    std::vector<cv::Point3d> src, dst; for (int i=0;i<n;++i) if (has_gps[i]) { src.push_back(centers_init[i]); dst.push_back(enu_gps[i]); }
                    if (src.size() >= 3) { Similarity sime = umeyama(src,dst); cv::Mat rvecSim; cv::Rodrigues(cv::Mat(sime.R), rvecSim); sim[0]=rvecSim.at<double>(0); sim[1]=rvecSim.at<double>(1); sim[2]=rvecSim.at<double>(2); sim[3]=sime.t[0]; sim[4]=sime.t[1]; sim[5]=sime.t[2]; sim[6]=std::log(std::max(1e-8, sime.s)); }
                    ceres::LossFunction* gps_loss = new ceres::CauchyLoss(10.0);
                    double gps_sigma_m = 5.0; double wx = 1.0 / std::max(1e-6, gps_sigma_m);
                    double wy = wx; double wz = 0.1 * wx; // stronger downweight altitude (U) axis
                    for (int i=0;i<n;++i) if (has_gps[i]) {
                        ceres::CostFunction* cost = GpsCenterError::Create(enu_gps[i].x, enu_gps[i].y, enu_gps[i].z, wx, wy, wz);
                        problem.AddResidualBlock(cost, gps_loss, &cams[6*i], sim.data()); used_gps_transform = true; }
                    // Bound similarity scale to avoid extreme values
                    problem.SetParameterLowerBound(sim.data(), 6, std::log(0.1));
                    problem.SetParameterUpperBound(sim.data(), 6, std::log(10.0));
                }
            }

            // Always anchor one camera for gauge stability (even with GPS).
            // Policy: pick the camera with the most feature observations that is present in the problem;
            // fallback to the first present camera if none has feature observations.
            int anchor_ci = -1;
            int best_obs = -1;
            for (int ci=0; ci<n; ++ci) {
                if (!problem.HasParameterBlock(&cams[6*ci])) continue;
                if (cam_obs_count[ci] > best_obs) { best_obs = cam_obs_count[ci]; anchor_ci = ci; }
            }
            if (anchor_ci < 0) {
                for (int ci=0; ci<n; ++ci) if (problem.HasParameterBlock(&cams[6*ci])) { anchor_ci = ci; best_obs = cam_obs_count[ci]; break; }
            }
            if (anchor_ci >= 0) {
                problem.SetParameterBlockConstant(&cams[6*anchor_ci]);
                std::ostringstream ssa; ssa << "Anchoring camera #" << anchor_ci << " with obs=" << best_obs; logI(ssa.str());
            } else {
                logI("No eligible camera found to anchor");
            }

            // Solve with SPARSE_SCHUR
            ceres::Solver::Options opts;
            opts.linear_solver_type = ceres::SPARSE_SCHUR;
            opts.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
            opts.minimizer_progress_to_stdout = false;
            opts.max_num_iterations = 100;
            opts.num_threads = std::max(1, cv::getNumThreads());
            ceres::Solver::Summary summary;
            if (problem.NumResidualBlocks() == 0) {
                logI("No BA residuals; skipping Ceres solve");
            } else {
                // Backup parameters before solve for rollback if needed
                std::vector<double> cams_backup = cams;
                std::vector<double> Xs_backup = Xs;
                auto Kpar_backup = Kpar;
                auto Dpar_backup = Dpar;

                ceres::Solve(opts, &problem, &summary);
                {
                    std::ostringstream ss; ss << "Ceres BA (SPARSE_SCHUR): " << (summary.IsSolutionUsable()?"OK":"NOT OK")
                                              << ", iters=" << summary.iterations.size()
                                              << ", final_cost=" << summary.final_cost;
                    logI(ss.str());
                }
                auto isfinite_vec = [](const std::vector<double>& v)->bool{
                    for (double x : v) if (!std::isfinite(x)) return false; return true;
                };
                auto isfinite_arr4 = [](const std::array<double,4>& a)->bool{
                    for (double x: a) if (!std::isfinite(x)) return false; return true;
                };
                auto isfinite_arr5 = [](const std::array<double,5>& a)->bool{
                    for (double x: a) if (!std::isfinite(x)) return false; return true;
                };
                bool bad = !isfinite_vec(cams) || !isfinite_vec(Xs) || !isfinite_arr4(Kpar) || !isfinite_arr5(Dpar);
                if (bad) {
                    logE("BA produced NaN/Inf; reverting to pre-BA parameters");
                    cams.swap(cams_backup);
                    Xs.swap(Xs_backup);
                    Kpar = Kpar_backup;
                    Dpar = Dpar_backup;
                }
            }

            // Unpack back
            for (int ci=0; ci<n; ++ci) {
                cv::Mat rvec = (cv::Mat_<double>(3,1) << cams[6*ci+0], cams[6*ci+1], cams[6*ci+2]);
                cv::Mat Rm; cv::Rodrigues(rvec, Rm);
                Rs[ci] = cv::Matx33d(Rm);
                ts[ci] = cv::Vec3d(cams[6*ci+3], cams[6*ci+4], cams[6*ci+5]);
            }
            for (int k=0;k<npts;++k) { pts[k] = cv::Point3d(Xs[3*k+0], Xs[3*k+1], Xs[3*k+2]); }
            K(0,0)=Kpar[0]; K(1,1)=Kpar[1]; K(0,2)=Kpar[2]; K(1,2)=Kpar[3];
            currK = Kpar; currD = Dpar;

            // Apply similarity transform to output if GPS used
            if (used_gps_transform) {
                cv::Matx33d Rsim; { cv::Mat rvec = (cv::Mat_<double>(3,1) << sim[0], sim[1], sim[2]); cv::Mat Rm; cv::Rodrigues(rvec, Rm); Rsim = cv::Matx33d(Rm); }
                double s_exp = std::exp(sim[6]); cv::Vec3d t_sim(sim[3], sim[4], sim[5]);
                for (auto &p : pts) { cv::Vec3d X(p.x,p.y,p.z); cv::Vec3d Xe = s_exp * (Rsim * X) + t_sim; p = cv::Point3d(Xe[0],Xe[1],Xe[2]); }
                cv::Matx33d RsimT = Rsim.t();
                for (int ci=0; ci<n; ++ci) {
                    // Center method: transform camera center and update (R,t)
                    cv::Matx33d Rold = Rs[ci];
                    cv::Vec3d told = ts[ci];
                    cv::Vec3d Cold = -Rold.t() * told;
                    cv::Vec3d Cn = s_exp * (Rsim * Cold) + t_sim;
                    cv::Matx33d Rnew = Rold * RsimT;
                    cv::Vec3d tnew = -Rnew * Cn;
                    Rs[ci] = Rnew;
                    ts[ci] = tnew;
                }
                skipGpsAlign = true;
            }

            auto [mean1, med1, rmse1, p901, nobs1] = compute_reproj_stats(pts, point_obs);
            {
                std::ostringstream ss; ss << std::fixed << std::setprecision(3)
                    << "Post-BA reproj (px): mean=" << mean1 << ", median=" << med1
                    << ", rmse=" << rmse1 << ", p90=" << p901 << " (n=" << nobs1 << ")";
                logI(ss.str());
            }

            // Per-camera stats & GPS stats to JSON
            std::vector<Stat> perCam(n);
            for (int ci=0; ci<n; ++ci) {
                std::vector<double> errs; errs.reserve(512);
                for (size_t k=0;k<pts.size() && k<point_obs.size(); ++k) {
                    const auto &P = pts[k];
                    if (!std::isfinite(P.x) || !std::isfinite(P.y) || !std::isfinite(P.z)) continue;
                    cv::Vec3d X(P.x, P.y, P.z);
                    for (const auto &o : point_obs[k]) if (o.cam == ci) {
                        cv::Point2f pred = proj_pixel(ci, X);
                        double du = pred.x - o.m.x; double dv = pred.y - o.m.y;
                        double e = std::sqrt(du*du + dv*dv);
                        if (std::isfinite(e)) errs.push_back(e);
                    }
                }
                if (!errs.empty()) {
                    double sum=0, sq=0; for (double e: errs){ sum+=e; sq+=e*e; }
                    std::sort(errs.begin(), errs.end());
                    perCam[ci].mean = sum/errs.size();
                    perCam[ci].rmse = std::sqrt(sq/errs.size());
                    perCam[ci].median = errs[errs.size()/2];
                    perCam[ci].p90 = (errs.size()<10) ? errs.back() : errs[(size_t)std::floor(0.9*(errs.size()-1))];
                    if (perCam[ci].p90 < perCam[ci].median) perCam[ci].p90 = perCam[ci].median;
                    perCam[ci].n = errs.size();
                }
            }
            // Log worst 5 cameras by RMSE to help diagnostics
            {
                struct CamErr { int idx; double rmse; double median; size_t n; };
                std::vector<CamErr> ce; ce.reserve(n);
                for (int i=0;i<n;++i) if (perCam[i].n>0 && std::isfinite(perCam[i].rmse)) ce.push_back({i, perCam[i].rmse, perCam[i].median, perCam[i].n});
                std::sort(ce.begin(), ce.end(), [](const CamErr&a,const CamErr&b){ return a.rmse > b.rmse; });
                int top = std::min<int>(5, (int)ce.size());
                if (top>0) {
                    std::ostringstream ss; ss << "Worst cameras by RMSE:";
                    for (int i=0;i<top;++i) ss << " [#" << ce[i].idx << ": rmse=" << std::setprecision(3) << ce[i].rmse << ", med=" << ce[i].median << ", n=" << ce[i].n << "]";
                    logI(ss.str());
                }
            }
            double gps_sq=0; size_t gps_n=0; if (jAlignGps) {
                for (int i=0;i<n;++i) if (has_gps[i]) { cv::Vec3d C = -Rs[i].t()*ts[i]; cv::Vec3d eg(enu_gps[i].x,enu_gps[i].y,enu_gps[i].z); cv::Vec3d d=C-eg; gps_sq+=d.dot(d); ++gps_n; }
            }
            std::array<double,4> Kpost{{K(0,0), K(1,1), K(0,2), K(1,2)}}; string statsPath = outDir + "/ba_stats.json";
            auto wstats = write_ba_stats_json(statsPath, preStat, Stat{mean1,med1,rmse1,p901,nobs1}, perCam, Kpre, Kpost, Dpre, Dpar, jAlignGps, gps_n, (gps_n? std::sqrt(gps_sq/gps_n):0));
            if (wstats.empty()) logE(string("Failed writing BA stats to ") + statsPath); else logI(string("Wrote BA stats: ") + statsPath);
        }
    } else {
        // Default and mode 1
        // OpenCV SFM returns cameras as vector<Mat> and 3D points as vector<Mat>
        // (each point is a 3x1 (or 1x3) CV_64F matrix). Using Mat (single) for points triggers
        // an assertion in getMatRef expecting a vector-of-Mat. Align with samples.
        vector<cv::Mat> Rs_cv, Ts_cv;
        vector<cv::Mat> points3d_cv;
        try {
            logI("Calling cv::sfm::reconstruct (projective=false, metric)...");
            cv::sfm::reconstruct(images_use, Rs_cv, Ts_cv, K, points3d_cv, /*is_projective=*/false);
            std::ostringstream ss; ss << "cv::sfm::reconstruct done: views=" << Rs_cv.size() << ", points=" << points3d_cv.size(); logI(ss.str());
        } catch (const cv::Exception &e1) {
            // Fallback to projective if metric fails
            std::ostringstream ws; ws << "Metric reconstruction failed: " << e1.what() << "; retrying with projective=true";
            logE(ws.str());
            try {
                cv::sfm::reconstruct(images_use, Rs_cv, Ts_cv, K, points3d_cv, /*is_projective=*/true);
                std::ostringstream ss; ss << "cv::sfm::reconstruct (projective) done: views=" << Rs_cv.size() << ", points=" << points3d_cv.size(); logI(ss.str());
            } catch (const cv::Exception &e2) {
                string msg = string("SFM failed (projective fallback): ") + e2.what();
                logE(msg);
                write_log_to_file(outDir);
                string out = log.str() + msg + "\nLog: " + logPath;
                return env->NewStringUTF(out.c_str());
            } catch (const std::exception &e2) {
                string msg = string("SFM failed (projective fallback): ") + e2.what();
                logE(msg);
                write_log_to_file(outDir);
                string out = log.str() + msg + "\nLog: " + logPath;
                return env->NewStringUTF(out.c_str());
            } catch (...) {
                string msg = "SFM failed (projective fallback): unknown error";
                logE(msg);
                write_log_to_file(outDir);
                string out = log.str() + msg + "\nLog: " + logPath;
                return env->NewStringUTF(out.c_str());
            }
        } catch (const std::exception &e) {
            string msg = string("SFM failed: ") + e.what();
            logE(msg);
            write_log_to_file(outDir);
            string out = log.str() + msg + "\nLog: " + logPath;
            return env->NewStringUTF(out.c_str());
        } catch (...) {
            string msg = "SFM failed: unknown error";
            logE(msg);
            write_log_to_file(outDir);
            string out = log.str() + msg + "\nLog: " + logPath;
            return env->NewStringUTF(out.c_str());
        }
        Rs.reserve(Rs_cv.size()); ts.reserve(Ts_cv.size());
        for (size_t i=0;i<Rs_cv.size();++i) {
            Rs.emplace_back(Rs_cv[i]);
            cv::Mat t = Ts_cv[i];
            // Ts are 3x1 doubles
            ts.emplace_back(t.at<double>(0), t.at<double>(1), t.at<double>(2));
        }
        // Flatten points3d (vector<Mat>) into vector<Point3d>
        for (const auto &m : points3d_cv) {
            if (m.empty()) continue;
            if (m.rows == 3 && m.cols == 1) {
                pts.emplace_back(m.at<double>(0), m.at<double>(1), m.at<double>(2));
            } else if (m.rows == 1 && m.cols == 3) {
                pts.emplace_back(m.at<double>(0,0), m.at<double>(0,1), m.at<double>(0,2));
            } else if (m.total() >= 3) {
                const double* d = m.ptr<double>(0);
                pts.emplace_back(d[0], d[1], d[2]);
            }
        }
    }

    // Compute camera centers
    vector<cv::Point3d> centers; centers.reserve(Rs.size());
    for (size_t i=0;i<Rs.size();++i) {
        cv::Matx33d R = Rs[i];
        cv::Vec3d t = ts[i];
        cv::Vec3d C = -R.t() * t; // camera center in SFM coords
        centers.emplace_back(C[0], C[1], C[2]);
    }

    // Align to ENU using GPS if requested (skipped if BA already aligned via GPS prior)
    if (jAlignGps && !skipGpsAlign && centers.size() == images_use.size()) {
        logI("Aligning to ENU using GPS from XMP (reading metadata from originals)...");
        // Collect ENU positions from XMP GPS
        // NOTE: pre-size to align indices with frames; fill by index.
        vector<cv::Point3d> enu(images_use.size());
        vector<bool> ok(images_use.size(), false);
        DjiMeta m0;
        // Reference origin as first that has geo
        int refIdx = -1;
        for (size_t i=0;i<images_use.size();++i) {
            DjiMeta mx; if (extract_dji_meta(images_meta[i], mx) && mx.hasGeo) { refIdx=(int)i; break; }
        }
        if (refIdx >= 0) {
            DjiMeta mref; extract_dji_meta(images_meta[refIdx], mref);
            cv::Point3d ecef0 = llh_to_ecef(mref.lat, mref.lon, mref.alt);
            cv::Matx33d Renu = ecef_to_enu_R(mref.lat, mref.lon);
            for (size_t i=0;i<images_use.size();++i) {
                DjiMeta mi; if (extract_dji_meta(images_meta[i], mi) && mi.hasGeo) {
                    cv::Point3d ecef = llh_to_ecef(mi.lat, mi.lon, mi.alt);
                    cv::Point3d p = ecef_to_enu(ecef, ecef0, Renu);
                    enu[i] = p;   // align by frame index
                    ok[i] = true;
                }
            }
            // Build matched lists for Umeyama (only where ok)
            vector<cv::Point3d> src, dst;
            src.reserve(images_use.size()); dst.reserve(images_use.size());
            for (size_t i=0;i<images_use.size();++i) if (ok[i]) { src.push_back(centers[i]); dst.push_back(enu[i]); }
            {
                std::ostringstream ss;
                ss << "GPS matched: " << src.size() << "/" << images_use.size() << " with ref=" << refIdx;
                logI(ss.str());
            }
            if (src.size() >= 3) {
                Similarity sim = umeyama(src, dst);
                {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(6)
                       << "Similarity: s=" << sim.s << ", R=[";
                    ss << std::setprecision(3)
                       << sim.R(0,0) << " " << sim.R(0,1) << " " << sim.R(0,2) << "; "
                       << sim.R(1,0) << " " << sim.R(1,1) << " " << sim.R(1,2) << "; "
                       << sim.R(2,0) << " " << sim.R(2,1) << " " << sim.R(2,2) << "], t=["
                       << sim.t[0] << " " << sim.t[1] << " " << sim.t[2] << "]";
                    logI(ss.str());
                }
                // Apply to points
                for (auto &p: pts) {
                    cv::Vec3d v(p.x, p.y, p.z);
                    cv::Vec3d ve = sim.s * (sim.R * v) + sim.t;
                    p = {ve[0], ve[1], ve[2]};
                }
                // Apply to cameras using center method and rotate R by Rsim^T
                cv::Matx33d RsimT(sim.R.t());
                for (size_t i=0;i<centers.size();++i) {
                    cv::Matx33d Rold = Rs[i];
                    cv::Vec3d told = ts[i];
                    cv::Vec3d Cold = -Rold.t() * told;
                    cv::Vec3d Cn = sim.s * (sim.R * Cold) + sim.t;
                    cv::Matx33d Rnew = Rold * RsimT;
                    cv::Vec3d tnew = -Rnew * Cn;
                    centers[i] = {Cn[0], Cn[1], Cn[2]};
                    Rs[i] = Rnew;
                    ts[i] = tnew;
                }
            } else {
                std::ostringstream ss;
                ss << "Not enough GPS tags for alignment: " << src.size() << " found";
                logE(ss.str());
            }
        } else {
            logE("No images with GPS metadata found; skipping alignment");
        }
    }

    // Write outputs
    string plyPath = outDir + "/reconstruction_points.ply";
    string jsonPath = outDir + "/camera_poses.json";
    auto plyW = write_ply(plyPath, pts);
    auto jsonW = write_extrinsics_json(jsonPath, images_use, Rs, ts, centers, K);
    if (plyW.empty()) logE(string("Failed writing PLY to ") + plyPath); else logI(string("Wrote PLY: ") + plyPath);
    if (jsonW.empty()) logE(string("Failed writing JSON to ") + jsonPath); else logI(string("Wrote JSON: ") + jsonPath);

    std::ostringstream result;
    result << log.str();
    result << "OK\nPLY: " << plyPath << "\nJSON: " << jsonPath << "\nLog: " << logPath;
    write_log_to_file(outDir);
string resStr = result.str();
    return env->NewStringUTF(resStr.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_rebuild_1edge_SfmNative_runColmapSfm(
        JNIEnv* env, jclass /*clazz*/, jstring jDatasetPath, jstring jRunDir, jstring jImageListPath,
        jstring jLogPath, jstring jCameraModel, jboolean jSingleCamera, jint jSeqOverlap,
        jint jGpsDist, jint jMaxImageSize, jint jThreads, jboolean jAlignGps,
        jstring jAlignmentType, jdouble jAlignmentMaxError) {
    std::ostringstream log;
    size_t lastFlushedLen = 0;
    std::string datasetPath = JStringToString(env, jDatasetPath);
    std::string runDir = JStringToString(env, jRunDir);
    std::string imageListPath = JStringToString(env, jImageListPath);
    std::string logPath = JStringToString(env, jLogPath);
    std::string cameraModel = JStringToString(env, jCameraModel);
    std::string alignmentType = JStringToString(env, jAlignmentType);
    if (cameraModel.empty()) cameraModel = "OPENCV";
    if (alignmentType.empty()) alignmentType = "ecef";
    bool singleCamera = (jSingleCamera == JNI_TRUE);
    int sequentialOverlap = std::max(1, static_cast<int>(jSeqOverlap));
    int gpsDist = std::max(1, static_cast<int>(jGpsDist));
    int maxImageSize = std::max(256, static_cast<int>(jMaxImageSize));
    int threads = std::max(1, static_cast<int>(jThreads));
    double alignMaxError = jAlignmentMaxError > 0.0 ? jAlignmentMaxError : 20.0;
    bool alignGps = (jAlignGps == JNI_TRUE);

    auto write_log_to_file = [&](const std::string& path) {
        if (path.empty()) return;
        const std::string current = log.str();
        if (current.size() <= lastFlushedLen) return;
        std::ofstream ofs(path, std::ios::app);
        if (ofs) {
            ofs.write(current.data() + lastFlushedLen,
                      static_cast<std::streamsize>(current.size() - lastFlushedLen));
            lastFlushedLen = current.size();
        }
    };
    auto flush = [&]() { write_log_to_file(logPath); };
    auto logI = [&](const std::string& msg) {
        ALOGI("%s", msg.c_str());
        log << "[I] " << msg << "\n";
        flush();
    };
    auto logE = [&](const std::string& msg) {
        ALOGE("%s", msg.c_str());
        log << "[E] " << msg << "\n";
        flush();
    };

    if (datasetPath.empty() || !dir_exists(datasetPath)) {
        string out = "Error: dataset path invalid -> " + datasetPath;
        logE(out);
        return env->NewStringUTF(out.c_str());
    }
    if (runDir.empty() || !mkdir_p(runDir)) {
        string out = "Error: cannot create run dir -> " + runDir;
        logE(out);
        return env->NewStringUTF(out.c_str());
    }
    if (imageListPath.empty() || !file_exists(imageListPath)) {
        string out = "Error: image_list.txt missing -> " + imageListPath;
        logE(out);
        return env->NewStringUTF(out.c_str());
    }
    if (!logPath.empty()) {
        std::ofstream ofs(logPath, std::ios::trunc);
        if (!ofs) {
            ALOGE("Failed to open log file: %s", logPath.c_str());
        }
    }

    std::string sparseDir = runDir + "/sparse";
    std::string alignedDir = runDir + "/sparse_aligned";
    mkdir_p(sparseDir);
    mkdir_p(alignedDir);
    mkdir_p(runDir + "/logs");

    size_t imageCount = 0;
    {
        std::ifstream ifs(imageListPath);
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty()) ++imageCount;
        }
    }
    if (imageCount < 2) {
        string out = "Error: image list has fewer than 2 entries";
        logE(out);
        return env->NewStringUTF(out.c_str());
    }

    ensure_glog_initialized();
    FileLogSink* sink = GetOrCreateFileLogSink();
    if (sink) {
        sink->set_path(logPath);
    }

    std::string dbPath = runDir + "/database.db";
    logI("COLMAP SfM start. dataset=" + datasetPath + ", runDir=" + runDir);

    bool pipelineOk = true;
    std::string failStage;

    std::vector<std::string> featArgs = {
            "colmap",
            "--database_path", dbPath,
            "--image_path", datasetPath,
            "--image_list_path", imageListPath,
            "--ImageReader.camera_model", cameraModel,
            "--ImageReader.single_camera", singleCamera ? "1" : "0",
            "--SiftExtraction.use_gpu", "0",
            "--SiftExtraction.max_image_size", std::to_string(maxImageSize),
            "--SiftExtraction.max_num_features", "8192",
            "--SiftExtraction.num_threads", std::to_string(threads)
    };
    if (!run_colmap_stage("特征提取 (feature_extractor)", &colmap::RunFeatureExtractor,
                          featArgs, log, flush, logI, logE)) {
        pipelineOk = false;
        failStage = "feature_extractor";
    }

    std::vector<std::string> seqArgs = {
            "colmap",
            "--database_path", dbPath,
            "--SiftMatching.use_gpu", "0",
            "--SequentialMatching.overlap", std::to_string(sequentialOverlap),
            "--SequentialMatching.quadratic_overlap", "1",
            "--SequentialMatching.loop_detection", "0",
            "--SiftMatching.num_threads", std::to_string(threads)
    };
    if (pipelineOk && !run_colmap_stage("序列匹配 (sequential_matcher)", &colmap::RunSequentialMatcher,
                                        seqArgs, log, flush, logI, logE)) {
        pipelineOk = false;
        failStage = "sequential_matcher";
    }

    std::vector<std::string> spatialArgs = {
            "colmap",
            "--database_path", dbPath,
            "--SiftMatching.use_gpu", "0",
            "--SpatialMatching.max_num_neighbors", "50",
            "--SpatialMatching.max_distance", std::to_string(gpsDist),
            "--SiftMatching.num_threads", std::to_string(threads)
    };
    if (pipelineOk && !run_colmap_stage("空间匹配 (spatial_matcher)", &colmap::RunSpatialMatcher,
                                        spatialArgs, log, flush, logI, logE)) {
        pipelineOk = false;
        failStage = "spatial_matcher";
    }

    std::vector<std::string> mapperArgs = {
            "colmap",
            "--database_path", dbPath,
            "--image_path", datasetPath,
            "--output_path", sparseDir,
            "--Mapper.multiple_models", "1",
            "--Mapper.min_model_size", "10",
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "1",
            "--Mapper.num_threads", std::to_string(threads),
            "--Mapper.ba_use_gpu", "0",
            "--Mapper.ba_gpu_index", "-1",
            "--Mapper.ba_local_max_num_iterations", "15",
            "--Mapper.ba_global_max_num_iterations", "25",
            "--Mapper.ba_global_images_ratio", "1.6",
            "--Mapper.ba_global_points_ratio", "1.6",
            "--Mapper.ba_global_max_refinements", "3",
            "--Mapper.ba_local_max_refinements", "1"
    };
    if (pipelineOk && !run_colmap_stage("稀疏重建 (mapper)", &colmap::RunMapper,
                                        mapperArgs, log, flush, logI, logE)) {
        pipelineOk = false;
        failStage = "mapper";
    }

    std::string sparseModelDir;
    if (pipelineOk) {
        sparseModelDir = pick_sparse_model_dir(sparseDir);
        if (sparseModelDir.empty()) {
            logE("未找到稀疏模型输出，目录=" + sparseDir);
            pipelineOk = false;
            failStage = "mapper_output";
        } else {
            logI("选择稀疏模型: " + sparseModelDir);
        }
    }

    if (pipelineOk && alignGps) {
        std::ostringstream alignErr;
        alignErr << std::fixed << std::setprecision(3) << alignMaxError;
        std::vector<std::string> alignArgs = {
                "colmap",
                "--input_path", sparseModelDir,
                "--output_path", alignedDir,
                "--database_path", dbPath,
                "--ref_is_gps", "1",
                "--alignment_type", alignmentType,
                "--alignment_max_error", alignErr.str()
        };
        if (run_colmap_stage("GPS对齐 (model_aligner)", &colmap::RunModelAligner,
                              alignArgs, log, flush, logI, logE)) {
            bool alignedOk = file_exists(alignedDir + "/images.bin") && file_exists(alignedDir + "/points3D.bin");
            if (alignedOk) {
                sparseModelDir = alignedDir;
                logI("GPS对齐成功，使用 " + sparseModelDir);
            } else {
                logE("GPS对齐完成但输出缺失，继续使用未对齐模型");
            }
        } else {
            logE("GPS对齐失败，继续使用未对齐模型");
        }
    }

    std::string result;
    if (pipelineOk && file_exists(sparseModelDir + "/points3D.bin")) {
        result = "OK: Sparse model ready at " + sparseModelDir + " (runDir=" + runDir + ")";
        logI(result);

        // Export PLY point cloud to runDir for easier sharing/inspection.
        const std::string plyPath = runDir + "/reconstruction_points.ply";
        std::vector<std::string> convArgs = {
                "colmap",
                "--input_path", sparseModelDir,
                "--output_path", plyPath,
                "--output_type", "PLY"
        };
        if (run_colmap_stage("点云导出 (model_converter)", &colmap::RunModelConverter,
                             convArgs, log, flush, logI, logE)) {
            if (file_exists(plyPath)) {
                logI("点云 PLY 已导出: " + plyPath);
                result += "\nPLY: " + plyPath;
            } else {
                logE("model_converter 运行成功但未找到 PLY 文件: " + plyPath);
            }
        } else {
            logE("点云导出 (model_converter) 失败，跳过 PLY 导出");
        }

        // Export camera poses JSON using COLMAP reconstruction.
        const std::string jsonPath = runDir + "/camera_poses.json";
        std::function<void(const std::string&)> fnLogI = logI;
        std::function<void(const std::string&)> fnLogE = logE;
        if (!write_colmap_extrinsics_json(sparseModelDir, jsonPath, fnLogI, fnLogE).empty()) {
            result += "\nJSON: " + jsonPath;
        }
    } else if (!pipelineOk) {
        result = "ERR: COLMAP stage failed -> " + failStage + ". Log: " + logPath;
        logE(result);
    } else {
        result = "ERR: Missing points3D.bin under " + sparseModelDir + ". Log: " + logPath;
        logE(result);
    }

    write_log_to_file(logPath);
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_rebuild_1edge_SfmNative_runColmapPatchMatch(
        JNIEnv* env, jclass /*clazz*/, jstring jDatasetPath, jstring jRunDir,
        jstring jWorkspaceDir, jstring jLogPath, jint jMaxImageSize,
        jboolean jGeomConsistency, jdouble jDepthMin, jdouble jDepthMax,
        jint jNumIterations, jint jWindowRadius, jint jNumSamples,
        jint jCacheSize, jint jThreads, jdouble jFusionMaxReprojError,
        jdouble jFusionMaxDepthError, jdouble jFusionMaxNormalError,
        jint jFusionMinNumConsistent) {
    std::ostringstream log;
    size_t lastFlushedLen = 0;
    std::string datasetPath = JStringToString(env, jDatasetPath);
    std::string runDir = JStringToString(env, jRunDir);
    std::string workspaceDir = JStringToString(env, jWorkspaceDir);
    std::string logPath = JStringToString(env, jLogPath);

    auto write_log_to_file = [&](const std::string& path) {
        if (path.empty()) return;
        const std::string current = log.str();
        if (current.size() <= lastFlushedLen) return;
        std::ofstream ofs(path, std::ios::app);
        if (ofs) {
            ofs.write(current.data() + lastFlushedLen,
                      static_cast<std::streamsize>(current.size() - lastFlushedLen));
            lastFlushedLen = current.size();
        }
    };
    auto flush = [&]() { write_log_to_file(logPath); };
    auto logI = [&](const std::string& msg) {
        ALOGI("%s", msg.c_str());
        log << "[I] " << msg << "\n";
        flush();
    };
    auto logE = [&](const std::string& msg) {
        ALOGE("%s", msg.c_str());
        log << "[E] " << msg << "\n";
        flush();
    };

    const int maxImageSize = std::max(256, static_cast<int>(jMaxImageSize));
    const bool geomConsistency = (jGeomConsistency == JNI_TRUE);
    const double depthMin = jDepthMin;
    const double depthMax = jDepthMax;
    const int numIterations = std::max(1, static_cast<int>(jNumIterations));
    const int windowRadius = std::max(1, static_cast<int>(jWindowRadius));
    const int numSamples = std::max(1, static_cast<int>(jNumSamples));
    const double cacheSize = std::max(1.0, static_cast<double>(jCacheSize));
    const int threads = std::max(1, static_cast<int>(jThreads));
    const double fusionMaxReprojError = jFusionMaxReprojError > 0.0 ? jFusionMaxReprojError : 4.0;
    const double fusionMaxDepthError = jFusionMaxDepthError > 0.0 ? jFusionMaxDepthError : 0.02;
    const double fusionMaxNormalError = jFusionMaxNormalError > 0.0 ? jFusionMaxNormalError : 20.0;
    const int fusionMinConsistent = std::max(1, static_cast<int>(jFusionMinNumConsistent));

    if (datasetPath.empty() || !dir_exists(datasetPath)) {
        std::string out = "Error: dataset path invalid -> " + datasetPath;
        logE(out);
        return env->NewStringUTF(out.c_str());
    }
    if (runDir.empty() || !dir_exists(runDir)) {
        std::string out = "Error: run dir missing -> " + runDir;
        logE(out);
        return env->NewStringUTF(out.c_str());
    }
    if (workspaceDir.empty()) {
        std::string out = "Error: workspace dir empty";
        logE(out);
        return env->NewStringUTF(out.c_str());
    }
    if (!mkdir_p(workspaceDir)) {
        std::string out = "Error: cannot create workspace dir -> " + workspaceDir;
        logE(out);
        return env->NewStringUTF(out.c_str());
    }
    if (!logPath.empty()) {
        std::ofstream ofs(logPath, std::ios::trunc);
        if (!ofs) {
            ALOGE("Failed to open log file: %s", logPath.c_str());
        }
    }

    ensure_glog_initialized();
    FileLogSink* sink = GetOrCreateFileLogSink();
    if (sink) {
        sink->set_path(logPath);
    }

    std::string sparseModelDir = pick_sparse_model_for_dense(runDir);
    if (sparseModelDir.empty()) {
        std::string out = "Error: 未找到稀疏模型目录 (sparse 或 sparse_aligned)";
        logE(out);
        return env->NewStringUTF(out.c_str());
    }
    logI("PatchMatch 输入模型: " + sparseModelDir);
    logI("PatchMatch 工作目录: " + workspaceDir);

    bool pipelineOk = true;
    std::string failStage;

    std::vector<std::string> undistArgs = {
            "colmap",
            "--image_path", datasetPath,
            "--input_path", sparseModelDir,
            "--output_path", workspaceDir,
            "--output_type", "COLMAP",
            "--max_image_size", std::to_string(maxImageSize)
    };
    if (!run_colmap_stage("图像去畸变 (image_undistorter)",
                          &colmap::RunImageUndistorter,
                          undistArgs, log, flush, logI, logE)) {
        pipelineOk = false;
        failStage = "image_undistorter";
    }

    std::vector<std::string> pmArgs = {
            "colmap",
            "--workspace_path", workspaceDir,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.gpu_index", "-1",
            "--PatchMatchStereo.max_image_size", std::to_string(maxImageSize),
            "--PatchMatchStereo.geom_consistency", geomConsistency ? "true" : "false",
            "--PatchMatchStereo.num_iterations", std::to_string(numIterations),
            "--PatchMatchStereo.window_radius", std::to_string(windowRadius),
            "--PatchMatchStereo.num_samples", std::to_string(numSamples),
            "--PatchMatchStereo.cache_size", std::to_string(cacheSize),
            "--PatchMatchStereo.num_threads", std::to_string(threads),
            "--PatchMatchStereo.filter_min_num_consistent", std::to_string(fusionMinConsistent),
            "--PatchMatchStereo.filter", "1"
    };
    if (depthMin > 0.0) {
        pmArgs.push_back("--PatchMatchStereo.depth_min");
        pmArgs.push_back(std::to_string(depthMin));
    }
    if (depthMax > 0.0) {
        pmArgs.push_back("--PatchMatchStereo.depth_max");
        pmArgs.push_back(std::to_string(depthMax));
    }
    if (pipelineOk && !run_colmap_stage("PatchMatch 立体匹配 (patch_match_stereo)",
                                        &colmap::RunPatchMatchStereo,
                                        pmArgs, log, flush, logI, logE)) {
        pipelineOk = false;
        failStage = "patch_match_stereo";
    }

    const std::string fusedPath = workspaceDir + "/fused.ply";
    std::vector<std::string> fusionArgs = {
            "colmap",
            "--workspace_path", workspaceDir,
            "--workspace_format", "COLMAP",
            "--input_type", geomConsistency ? "geometric" : "photometric",
            "--output_path", fusedPath,
            "--output_type", "PLY",
            "--StereoFusion.max_reproj_error", std::to_string(fusionMaxReprojError),
            "--StereoFusion.max_depth_error", std::to_string(fusionMaxDepthError),
            "--StereoFusion.max_normal_error", std::to_string(fusionMaxNormalError),
            "--StereoFusion.min_num_pixels", std::to_string(fusionMinConsistent),
            "--StereoFusion.max_image_size", std::to_string(maxImageSize),
            "--StereoFusion.num_threads", std::to_string(threads)
    };
    if (pipelineOk && !run_colmap_stage("稠密融合 (stereo_fusion)",
                                        &colmap::RunStereoFuser,
                                        fusionArgs, log, flush, logI, logE)) {
        pipelineOk = false;
        failStage = "stereo_fusion";
    }

    std::string result;
    if (pipelineOk && file_exists(fusedPath)) {
        result = "OK: PatchMatch MVS 输出 " + fusedPath;
        logI(result);
    } else {
        result = "ERR: PatchMatch pipeline 失败 (" + failStage + ")";
        logE(result);
    }

    write_log_to_file(logPath);
    return env->NewStringUTF(result.c_str());
}
