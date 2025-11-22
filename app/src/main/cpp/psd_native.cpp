#include <jni.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_rebuild_1edge_ui_psd_PsdNative_depthToPoint(
        JNIEnv* env, jobject /*thiz*/,
        jfloatArray depthArr,
        jfloatArray kArr,
        jint batch,
        jint height,
        jint width) {
    const jsize depthSize = env->GetArrayLength(depthArr);
    const jsize kSize = env->GetArrayLength(kArr);
    if (kSize < 9 || depthSize != batch * height * width) {
        return nullptr;
    }
    std::vector<float> depth(depthSize);
    env->GetFloatArrayRegion(depthArr, 0, depthSize, depth.data());
    float K[9];
    env->GetFloatArrayRegion(kArr, 0, 9, K);
    const float fx = K[0];
    const float fy = K[4];
    const float cx = K[2];
    const float cy = K[5];

    const int outSize = batch * 3 * height * width;
    std::vector<float> out(outSize);
    int idx = 0;
    int outIdx = 0;
    for (int b = 0; b < batch; ++b) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x, ++idx) {
                const float z = depth[idx];
                const float xn = (static_cast<float>(x) - cx) / fx;
                const float yn = (static_cast<float>(y) - cy) / fy;
                out[outIdx++] = xn * z;
                out[outIdx++] = yn * z;
                out[outIdx++] = z;
            }
        }
    }
    jfloatArray result = env->NewFloatArray(outSize);
    env->SetFloatArrayRegion(result, 0, outSize, out.data());
    return result;
}

// Simple 3x3 CSPN with cosine affinity over feature map. Shapes:
// sparse/dense: [B,1,H,W], feature: [B,C,H,W]
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_rebuild_1edge_ui_psd_PsdNative_cspn(
        JNIEnv* env, jobject /*thiz*/,
        jfloatArray sparseArr,
        jfloatArray denseArr,
        jfloatArray featArr,
        jint batch,
        jint channels,
        jint height,
        jint width,
        jint iteration) {
    const int hw = height * width;
    const int sparseSize = env->GetArrayLength(sparseArr);
    const int denseSize = env->GetArrayLength(denseArr);
    const int featSize = env->GetArrayLength(featArr);
    if (sparseSize != batch * hw || denseSize != batch * hw ||
        featSize != batch * channels * hw) {
        return nullptr;
    }
    std::vector<float> sparse(sparseSize);
    std::vector<float> dense(denseSize);
    std::vector<float> feat(featSize);
    env->GetFloatArrayRegion(sparseArr, 0, sparseSize, sparse.data());
    env->GetFloatArrayRegion(denseArr, 0, denseSize, dense.data());
    env->GetFloatArrayRegion(featArr, 0, featSize, feat.data());

    std::vector<float> out = dense;
    std::vector<char> mask(sparseSize);
    for (int i = 0; i < sparseSize; ++i) mask[i] = sparse[i] > 0.f;

    for (int it = 0; it < iteration; ++it) {
        std::vector<float> next = out;
        for (int b = 0; b < batch; ++b) {
            const int baseDense = b * hw;
            const int baseFeat = b * channels * hw;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const int idxCenter = baseDense + y * width + x;
                    if (mask[idxCenter]) {
                        next[idxCenter] = sparse[idxCenter];
                        continue;
                    }
                    float acc = 0.f;
                    float wsum = 0.f;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dy == 0 && dx == 0) continue;
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
                            int nIdx = baseDense + ny * width + nx;
                            float dot = 0.f, normN = 0.f, normC = 0.f;
                            for (int c = 0; c < channels; ++c) {
                                const int off = baseFeat + c * hw;
                                float cv = feat[off + y * width + x];
                                float nv = feat[off + ny * width + nx];
                                dot += cv * nv;
                                normC += cv * cv;
                                normN += nv * nv;
                            }
                            float denom = std::sqrt(std::max(normC * normN, 1e-12f));
                            float sim = denom > 0.f ? dot / denom : 0.f;
                            float wgt = std::max(sim, 0.f);
                            acc += out[nIdx] * wgt;
                            wsum += wgt;
                        }
                    }
                    if (wsum > 1e-6f) {
                        next[idxCenter] = acc / wsum;
                    }
                }
            }
        }
        out.swap(next);
    }
    jfloatArray result = env->NewFloatArray(out.size());
    env->SetFloatArrayRegion(result, 0, out.size(), out.data());
    return result;
}

// Simple neighbor-average fill for sparse depth.
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_rebuild_1edge_ui_psd_PsdNative_fillSparseFast(
        JNIEnv* env, jobject /*thiz*/,
        jfloatArray sparseArr,
        jint batch,
        jint height,
        jint width,
        jint iterations) {
    const int hw = height * width;
    const int size = env->GetArrayLength(sparseArr);
    if (size != batch * hw) return nullptr;
    std::vector<float> out(size);
    env->GetFloatArrayRegion(sparseArr, 0, size, out.data());
    for (int it = 0; it < iterations; ++it) {
        std::vector<float> tmp = out;
        for (int b = 0; b < batch; ++b) {
            const int base = b * hw;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int idx = base + y * width + x;
                    if (out[idx] > 0.f) continue;
                    float sum = 0.f;
                    int cnt = 0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dy == 0 && dx == 0) continue;
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
                            int nIdx = base + ny * width + nx;
                            float v = out[nIdx];
                            if (v > 0.f) {
                                sum += v;
                                cnt += 1;
                            }
                        }
                    }
                    if (cnt > 0) tmp[idx] = sum / cnt;
                }
            }
        }
        out.swap(tmp);
    }
    jfloatArray result = env->NewFloatArray(size);
    env->SetFloatArrayRegion(result, 0, size, out.data());
    return result;
}

// Brute-force KNN propagation using geometry+feature re-ranking (B=1 supported).
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_rebuild_1edge_ui_psd_PsdNative_knnPropagate(
        JNIEnv* env, jobject /*thiz*/,
        jfloatArray pointArr,
        jfloatArray featArr,
        jfloatArray depthArr,
        jfloatArray sparseArr,
        jint batch,
        jint channels,
        jint height,
        jint width,
        jint knn,
        jint candidateLimit) {
    const int hw = height * width;
    const int pointSize = env->GetArrayLength(pointArr);
    const int featSize = env->GetArrayLength(featArr);
    const int depthSize = env->GetArrayLength(depthArr);
    const int sparseSize = env->GetArrayLength(sparseArr);
    if (batch != 1) {
        return nullptr; // only B=1 for now
    }
    if (pointSize != batch * 3 * hw || featSize != batch * channels * hw ||
        depthSize != batch * hw || sparseSize != batch * hw) {
        return nullptr;
    }
    std::vector<float> point(pointSize);
    std::vector<float> feat(featSize);
    std::vector<float> depth(depthSize);
    std::vector<float> sparse(sparseSize);
    env->GetFloatArrayRegion(pointArr, 0, pointSize, point.data());
    env->GetFloatArrayRegion(featArr, 0, featSize, feat.data());
    env->GetFloatArrayRegion(depthArr, 0, depthSize, depth.data());
    env->GetFloatArrayRegion(sparseArr, 0, sparseSize, sparse.data());

    struct SparsePix { int idx; float z; float px; float py; float pz; };
    std::vector<SparsePix> sparsePts;
    sparsePts.reserve(hw);
    for (int i = 0; i < hw; ++i) {
        if (sparse[i] > 0.f) {
            SparsePix sp;
            sp.idx = i;
            sp.z = sparse[i];
            const int base = i * 3;
            sp.px = point[base];
            sp.py = point[base + 1];
            sp.pz = point[base + 2];
            sparsePts.push_back(sp);
        }
    }
    if (candidateLimit > 0 && static_cast<int>(sparsePts.size()) > candidateLimit) {
        sparsePts.resize(candidateLimit);
    }
    std::vector<float> out = depth;
    // For each pixel, pick geometric KNN, then re-rank by feature cosine
    for (int idx = 0; idx < hw; ++idx) {
        const int basePt = idx * 3;
        const float px = point[basePt];
        const float py = point[basePt + 1];
        const float pz = point[basePt + 2];
        // geometric distances
        std::vector<std::pair<float, int>> geo;
        geo.reserve(sparsePts.size());
        for (size_t si = 0; si < sparsePts.size(); ++si) {
            const float dx = px - sparsePts[si].px;
            const float dy = py - sparsePts[si].py;
            const float dz = pz - sparsePts[si].pz;
            const float dist2 = dx*dx + dy*dy + dz*dz;
            geo.emplace_back(dist2, static_cast<int>(si));
        }
        const int k2 = std::min<int>(knn * 2, geo.size());
        std::nth_element(geo.begin(), geo.begin() + k2, geo.end(),
                         [](const auto& a, const auto& b){ return a.first < b.first; });
        geo.resize(k2);

        // feature of target
        const float* featTarget = &feat[idx]; // layout: [C,H*W] contiguous per channel?
        // Our feat layout is [C,H,W] flattened channel-major: off = c*hw + idx
        std::vector<std::pair<float,int>> sims;
        sims.reserve(k2);
        for (const auto& g : geo) {
            const int si = g.second;
            const int sidx = sparsePts[si].idx;
            // cosine similarity
            float dot=0.f, normT=0.f, normS=0.f;
            for (int c = 0; c < channels; ++c) {
                const int off = c * hw;
                const float tv = feat[off + idx];
                const float sv = feat[off + sidx];
                dot += tv * sv;
                normT += tv * tv;
                normS += sv * sv;
            }
            float denom = std::sqrt(std::max(normT * normS, 1e-12f));
            float cos = denom > 0.f ? dot / denom : 0.f;
            sims.emplace_back(cos, si);
        }
        const int k = std::min<int>(knn, sims.size());
        std::nth_element(sims.begin(), sims.begin() + k, sims.end(),
                         [](const auto& a, const auto& b){ return a.first > b.first; });
        sims.resize(k);
        float wsum = 0.f;
        float acc = 0.f;
        for (const auto& s : sims) {
            float wgt = std::max(s.first, 0.f);
            acc += sparsePts[s.second].z * wgt;
            wsum += wgt;
        }
        if (wsum > 1e-6f) {
            out[idx] = acc / wsum;
        }
    }
    jfloatArray result = env->NewFloatArray(out.size());
    env->SetFloatArrayRegion(result, 0, out.size(), out.data());
    return result;
}
