#include <net.h>

struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

extern std::vector<Object> detect_v5lite(ncnn::Extractor & ex, ncnn::Mat in1, bool use_gpu, int wpad, int hpad, int width, int height, float scale);
