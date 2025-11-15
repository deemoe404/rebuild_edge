#include <cstddef>

struct Plugin;
struct TIFF;

// COLMAP only needs FreeImage entry points to satisfy linkage on Android
// builds where the optional plugins are stripped out. We provide empty
// definitions so the loader does not attempt to resolve the real codecs.
void InitJP2(Plugin* /*plugin*/, int /*format_id*/) {}
void InitWEBP(Plugin* /*plugin*/, int /*format_id*/) {}

extern "C" {

int TIFFInitOJPEG(TIFF* /*tif*/, int /*scheme*/) { return 0; }

}
