#include <stdio.h>
#include <stdlib.h>
#include "helper_cuda.h"

#define DIM 1000 /* rozmiar rysunku w pikselach */
#define DIM_BLOCK 16

#define REFRESH_DELAY     10 //ms

#include "GL/freeglut.h"

const int K = 1000; /* max iteracji Julii */

__device__
int julia(float x, float y) {
    float cr = -0.123F;
    float ci = 0.745F;
    float mod_c = sqrtf(cr * cr + ci * ci);
    float R2 = fmaxf(mod_c, 2) * fmaxf(mod_c, 2);
    float x2 = x * x;
    float y2 = y * y;
    int i = 0;
    while (i < K) {
        if (x2 + y2 > R2) {
            return 0;
        }
        y = 2 * x * y + ci;
        x = x2 - y2 + cr;
        i++;
        x2 = x * x;
        y2 = y * y;
    }
    return 1;
}

__global__
void kernel(unsigned char *ptr,
            const float dx, const float dy,
            const float scale) {
    /* przeliczenie współrzędnych pikselowych (xw, yw)
       na matematyczne (x, y) z uwzględnieniem skali
       (scale) i matematycznego środka (dx, dy) */
    int xw = threadIdx.x + blockIdx.x * blockDim.x;
    if (xw < DIM) {
        int yw = threadIdx.y + blockIdx.y * blockDim.y;
        if (yw < DIM) {
            float x = scale * (float) (xw - DIM / 2) / (DIM / 2) + dx,
                    y = scale * (float) (yw - DIM / 2) / (DIM / 2) + dy;
            int offset /* w buforze pikseli */ = xw + yw * DIM;
            /* kolor: czarny dla uciekinierów (julia == 0)
                      czerwony dla wiêniów (julia == 1) */
            ptr[offset * 4 + 0 /* R */] = (unsigned char) (255 * julia(x, y));
            ptr[offset * 4 + 1 /* G */] = 0;
            ptr[offset * 4 + 2 /* B */] = 0;
            ptr[offset * 4 + 3 /* A */] = 255;
        }
    }
}

/**************************************************/

// Uncoment if you are using Windows
//#define WIN32


static unsigned char pixbuf[DIM * DIM * 4];
unsigned char *d_pixbuf;

static float dx = 0.0f, dy = 0.0f;
static float scale = 1.5f;

static void disp(void) {
    glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, pixbuf);
    glutSwapBuffers();
}

void set_time_as_window_title(float time) {
    char buffer[64];
    snprintf(buffer, sizeof buffer, "Compute time: %f", time);
    glutSetWindowTitle(buffer);
}

void setup_GPU_time_measuring(cudaEvent_t

& start, cudaEvent_t& stop){
checkCudaErrors (cudaEventCreate(

&start));

checkCudaErrors (cudaEventCreate(

&stop));
}

float get_GPU_elapsed_time(cudaEvent_t

& start, cudaEvent_t& stop){
float elapsedTime;

checkCudaErrors (cudaEventElapsedTime(

&elapsedTime,
start, stop));

checkCudaErrors (cudaEventDestroy(start));

checkCudaErrors (cudaEventDestroy(stop));

return
elapsedTime;
}

static void recompute() {
    dim3
    dimGrid((DIM + DIM_BLOCK - 1) / DIM_BLOCK,
            (DIM + DIM_BLOCK - 1) / DIM_BLOCK);
    dim3
    dimBlock(DIM_BLOCK, DIM_BLOCK);

    // copy pixels array to GPU
    checkCudaErrors(cudaMemcpy(d_pixbuf, pixbuf, sizeof(pixbuf),
                               cudaMemcpyHostToDevice));

    // GPU kernel time measure init
    cudaEvent_t start, stop;
    setup_GPU_time_measuring(start, stop);
    checkCudaErrors(cudaEventRecord(start, 0));

    kernel << < dimGrid, dimBlock >> > (d_pixbuf, dx, dy, scale);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    // GPU kernel time count
    float elapsedTime = get_GPU_elapsed_time(start, stop);

    // retrieve pixels array from GPU to local array
    checkCudaErrors(cudaMemcpy(pixbuf, d_pixbuf, sizeof(pixbuf),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaGetLastError());

    set_time_as_window_title(elapsedTime);
    glutPostRedisplay();
}

static void kbd(unsigned char key, int x, int y) {
    switch (key) {
        case 'p':
            dx += scale * (float) (x - DIM / 2) / (DIM / 2);
            dy -= scale * (float) (y - DIM / 2) / (DIM / 2);
            break;
        case 'z':
            scale *= 0.80f;
            break;
        case 'Z':
            scale *= 1.25f;
            break;
        case '=':
            scale = 1.50f;
            dx = dy = 0.0f;
            break;
        case 27: /* Esc */ exit(0);
    }
    recompute();
}

void timerEvent(int value) {
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}


int main(int argc, char *argv[]) {

    checkCudaErrors(cudaSetDevice(0));

    // allocate memory for pixels array on GPU
    checkCudaErrors(cudaMalloc(&d_pixbuf, sizeof(pixbuf)));
    checkCudaErrors(cudaGetLastError());

    glutInit(&argc, argv); /* inicjacja biblioteki GLUT */
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); /* opcje */
    glutInitWindowSize(DIM, DIM); /* rozmiar okna graficznego */
    glutCreateWindow("RIM - fraktal Julii"); /* tytuł okna */
    glutDisplayFunc(disp); /* funkcja zwrotna zobrazowania */
    glutKeyboardFunc(kbd); /* funkcja zwrotna klawiatury */
    recompute();           /* obliczenie pierwszego rysunku */
    glutMainLoop();        /* g³ówna pêtla obsługi zdarzeń */
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

}
