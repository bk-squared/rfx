#include <time.h>
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
void busy_seconds(double seconds) {
    const double end = now() + seconds;
    volatile double value = 0.0;
    while (now() < end) {
        for (int i = 0; i < 1000; ++i) value += 0.000001;
    }
}
