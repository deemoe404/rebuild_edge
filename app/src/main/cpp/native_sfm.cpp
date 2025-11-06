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

#include <android/log.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/sfm.hpp>

#include <sys/stat.h>
#include <sys/types.h>

#define LOG_TAG "sfm_native"
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using std::string;
using std::vector;

namespace {

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
    string buf;
    if (!read_file(jpegPath, buf)) return false;
    string xmp;
    if (!extract_xmp_xml(buf, xmp)) return false;
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

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_rebuild_1edge_SfmNative_runSfm(
        JNIEnv* env, jclass /*clazz*/, jobjectArray jImagePaths, jstring jOutDir, jboolean jAlignGps, jint jMaxLongEdge) {
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
    auto write_log_to_file = [&](const string &dir){
        if (dir.empty()) return;
        std::ofstream ofs(logPath);
        if (ofs) {
            ofs << log.str();
            ofs.close();
            ALOGI("Wrote log to %s", logPath.c_str());
        } else {
            ALOGE("Failed to write log to %s", logPath.c_str());
        }
    };

    // Enable incremental flushing for UI tailing
    flush = [&](){ write_log_to_file(outDir); };

    {
        std::ostringstream ss;
        ss << "RunSfm start: " << images.size() << " images, outDir=" << outDir
           << ", alignGps=" << (jAlignGps?"true":"false")
           << ", maxLongEdge=" << (int)jMaxLongEdge;
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

    // Read first image to know original size
    cv::Mat firstGray = cv::imread(images[0], cv::IMREAD_GRAYSCALE);
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
    if (!extract_dji_meta(images[0], meta) || !meta.hasIntrinsics) {
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
    images_use.reserve(images.size());
    if (s < 1.0) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4)
           << "Downscaling images by s=" << s << " to fit maxLongEdge=" << maxLongEdge << " px";
        logI(ss.str());

        std::string dsDir = outDir + "/downscaled_" + std::to_string(maxLongEdge);
        // Create directory (single level under outDir) – best-effort
        ::mkdir(dsDir.c_str(), 0755);
        for (size_t i = 0; i < images.size(); ++i) {
            const std::string &srcPath = images[i];
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
        if (images_use.size() != images.size()) {
            logE("Warning: some images failed to downscale; proceeding with successfully written ones.");
        }
        if (images_use.empty()) {
            // fallback to originals if downscale failed
            images_use = images;
            logE("Downscale produced no images, using originals.");
        }
    } else {
        images_use = images;
    }

    // Run reconstruction with libmv simple pipeline via cv::sfm
    vector<cv::Mat> Rs_cv, Ts_cv;
    cv::Mat points3d_cv;
    try {
        logI("Calling cv::sfm::reconstruct (projective=true)...");
        cv::sfm::reconstruct(images_use, Rs_cv, Ts_cv, cv::Mat(K), points3d_cv, /*is_projective=*/true);
        {
            std::ostringstream ss;
            ss << "cv::sfm::reconstruct done: views=" << Rs_cv.size()
               << " Ts=" << Ts_cv.size()
               << " points=" << points3d_cv.rows
               << " (images=" << images.size() << ")";
            logI(ss.str());
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

    // Convert outputs
    vector<cv::Matx33d> Rs; Rs.reserve(Rs_cv.size());
    vector<cv::Vec3d> ts; ts.reserve(Ts_cv.size());
    for (size_t i=0;i<Rs_cv.size();++i) {
        Rs.emplace_back(Rs_cv[i]);
        cv::Mat t = Ts_cv[i];
        ts.emplace_back(t.at<double>(0), t.at<double>(1), t.at<double>(2));
    }
    vector<cv::Point3d> pts;
    if (!points3d_cv.empty()) {
        for (int r=0; r<points3d_cv.rows; ++r) {
            cv::Vec3d p = points3d_cv.at<cv::Vec3d>(r);
            pts.emplace_back(p[0], p[1], p[2]);
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

    // Align to ENU using GPS if requested
    if (jAlignGps && centers.size() == images.size()) {
        logI("Aligning to ENU using GPS from XMP...");
        // Collect ENU positions from XMP GPS
        vector<cv::Point3d> enu;
        enu.reserve(images.size());
        vector<bool> ok(images.size(), false);
        DjiMeta m0;
        // Reference origin as first that has geo
        int refIdx = -1;
        for (size_t i=0;i<images.size();++i) {
            DjiMeta mx; if (extract_dji_meta(images[i], mx) && mx.hasGeo) { refIdx=(int)i; break; }
        }
        if (refIdx >= 0) {
            DjiMeta mref; extract_dji_meta(images[refIdx], mref);
            cv::Point3d ecef0 = llh_to_ecef(mref.lat, mref.lon, mref.alt);
            cv::Matx33d Renu = ecef_to_enu_R(mref.lat, mref.lon);
            for (size_t i=0;i<images.size();++i) {
                DjiMeta mi; if (extract_dji_meta(images[i], mi) && mi.hasGeo) {
                    cv::Point3d ecef = llh_to_ecef(mi.lat, mi.lon, mi.alt);
                    cv::Point3d p = ecef_to_enu(ecef, ecef0, Renu);
                    enu.emplace_back(p);
                    ok[i] = true;
                }
            }
            // Build matched lists for Umeyama (only where ok)
            vector<cv::Point3d> src, dst;
            src.reserve(images.size()); dst.reserve(images.size());
            for (size_t i=0;i<images.size();++i) if (ok[i]) { src.push_back(centers[i]); dst.push_back(enu[i]); }
            {
                std::ostringstream ss;
                ss << "GPS matched: " << src.size() << "/" << images.size() << " with ref=" << refIdx;
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
                // Apply to points and centers; update t accordingly
                for (auto &p: pts) {
                    cv::Vec3d v(p.x, p.y, p.z);
                    cv::Vec3d ve = sim.s * (sim.R * v) + sim.t;
                    p = {ve[0], ve[1], ve[2]};
                }
                for (size_t i=0;i<centers.size();++i) {
                    cv::Vec3d c(centers[i].x, centers[i].y, centers[i].z);
                    cv::Vec3d ce = sim.s * (sim.R * c) + sim.t;
                    centers[i] = {ce[0], ce[1], ce[2]};
                    ts[i] = -Rs[i] * cv::Vec3d(centers[i].x, centers[i].y, centers[i].z);
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
    auto jsonW = write_extrinsics_json(jsonPath, images, Rs, ts, centers, K);
    if (plyW.empty()) logE(string("Failed writing PLY to ") + plyPath); else logI(string("Wrote PLY: ") + plyPath);
    if (jsonW.empty()) logE(string("Failed writing JSON to ") + jsonPath); else logI(string("Wrote JSON: ") + jsonPath);

    std::ostringstream result;
    result << log.str();
    result << "OK\nPLY: " << plyPath << "\nJSON: " << jsonPath << "\nLog: " << logPath;
    write_log_to_file(outDir);
    string resStr = result.str();
    return env->NewStringUTF(resStr.c_str());
}
