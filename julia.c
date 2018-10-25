#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM 1000 /* rozmiar rysunku w pikselach */

const int K = 1000; /* max iteracji Julii */

float max(float a, float b){
    if (a > b) return a; else return b;
}

const float cr = -0.123F;
const float ci = 0.745F;


int julia(float x, float y)
{
    float mod_c = sqrtf(cr*cr + ci*ci);
    float R2 = max(mod_c,2)*max(mod_c,2);
    float x2 = x*x;
    float y2 = y*y;
    int i=0;
    while(i<K){
        if (x2 + y2 > R2) {
//            if (x > 1 || y > 1 )  return 1; else return 0;
            return 1;
        }
        y = 2*x*y + ci;
        x = x2 - y2 +cr;
        i++;
        x2=x*x;
        y2=y*y;
    }
//    if (x > 1 || y > 1 )  return 1; else return 0;
    return 0;
}

void kernel(unsigned char *ptr,
            const int xw, const int yw,
            const float dx, const float dy,
            const float scale)
{
    /* przeliczenie współrzędnych pikselowych (xw, yw)
       na matematyczne (x, y) z uwzględnieniem skali
       (scale) i matematycznego środka (dx, dy) */
    float x = scale*(float)(xw-DIM/2)/(DIM/2) + dx,
            y = scale*(float)(yw-DIM/2)/(DIM/2) + dy;
    int offset /* w buforze pikseli */ = xw + yw*DIM;
    /* kolor: czarny dla uciekinierów (julia == 0)
              czerwony dla wiêniów (julia == 1) */
    ptr[offset*4 + 0 /* R */] = (unsigned char) (255*julia(x,y));
    ptr[offset*4 + 1 /* G */] = 0;
    ptr[offset*4 + 2 /* B */] = 0;
    ptr[offset*4 + 3 /* A */] = 255;
}

/**************************************************/

#include "GL/freeglut.h"

static unsigned char pixbuf[DIM * DIM * 4];
static float dx = 0.0f, dy = 0.0f;
static float scale = 1.5f;

static void disp(void)
{
    glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, pixbuf);
    glutSwapBuffers();
}

static void recompute(void)
{
    int xw, yw;
    for (yw = 0; yw < DIM; yw++)
        for (xw = 0; xw < DIM; xw++)
            kernel(pixbuf, xw, yw, dx, dy, scale);
    glutPostRedisplay();
}

static void kbd(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'p': dx += scale*(float)(x-DIM/2)/(DIM/2);
            dy -= scale*(float)(y-DIM/2)/(DIM/2);
            break;
        case 'z': scale *= 0.80f;                 break;
        case 'Z': scale *= 1.25f;                 break;
        case '=': scale  = 1.50f; dx = dy = 0.0f; break;
        case 27: /* Esc */ exit(0);
    }
    recompute();
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv); /* inicjacja biblioteki GLUT */
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); /* opcje */
    glutInitWindowSize(DIM, DIM); /* rozmiar okna graficznego */
    glutCreateWindow("RIM - fraktal Julii"); /* tytuł okna */
    glutDisplayFunc(disp); /* funkcja zwrotna zobrazowania */
    glutKeyboardFunc(kbd); /* funkcja zwrotna klawiatury */
    recompute();           /* obliczenie pierwszego rysunku */
    glutMainLoop();        /* g³ówna pêtla obsługi zdarzeń */
}
