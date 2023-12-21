/***************************************************************************\
 * Copyright (c) 2023, Sofie Martins                                         *   
 * All rights reserved.                                                      * 
 \***************************************************************************/

 /**
 * @file stitch_configs.c
 * @brief Stitch configurations from smaller lattices together to larger configs.
 */

#include "libhr.h"
#include <string.h>

#ifdef WITH_MPI
#error Please commpile without MPI!
#endif

int GLB_T_IN, GLB_X_IN, GLB_Y_IN, GLB_Z_IN;
int GLB_T_OUT, GLB_X_OUT, GLB_Y_OUT, GLB_Z_OUT;

#define init_input_config(varname) \
    { \
        .read = { \
            { "GLB_T_IN", "GLB_T_IN = %d", INT_T, &GLB_T_IN }, \
            { "GLB_X_IN", "GLB_X_IN = %d", INT_T, &GLB_X_IN }, \
            { "GLB_Y_IN", "GLB_Y_IN = %d", INT_T, &GLB_Y_IN }, \
            { "GLB_Z_IN", "GLB_Z_IN = %d", INT_T, &GLB_Z_IN }, \
            { "file_in", "file_in = %s", STRING_T, &((varname).file[0]) }, \
        } \
    }

#define init_output_config(varname) \
    { \
        .read = { \
            { "GLB_T_OUT", "GLB_T_OUT = %d", INT_T, &GLB_T_OUT }, \
            { "GLB_X_OUT", "GLB_X_OUT = %d", INT_T, &GLB_X_OUT }, \
            { "GLB_Y_OUT", "GLB_Y_OUT = %d", INT_T, &GLB_Y_OUT }, \
            { "GLB_Z_OUT", "GLB_Z_OUT = %d", INT_T, &GLB_Z_OUT }, \
            { "file_in", "file_out = %s", STRING_T, &((varname).file[0]) }, \
        } \
    }

typedef struct {
    char file[256];
    input_record_t read[5];
} input_file_var;

typedef struct {
    char file[256];
    input_record_t read[5];
} output_file_var;

static int *ipt_old;

#define _lexi_convert(_T, _X, _Y, _Z, _t, _x, _y, _z) (((((_t) % (_T)) * (_X) + ((_x) % (_X))) * (_Y) + ((_y) % (_Y))) * (_Z) + ((_z) % (_Z)))
#define ipt_box(_T, _X, _Y, _Z, _t, _x, _y, _z) ipt_old[_lexi_convert(_T, _X, _Y, _Z, _t, _x, _y, _z)];

int main(int argc, char *argv[]) {
    setup_process(&argc, &argv);
    input_file_var input_var = init_input_config(input_var);
    output_file_var output_var = init_output_config(output_var);

    char *inputfile = get_input_filename();

    read_input(input_var.read, inputfile);
    read_input(output_var.read, inputfile);

    GLB_T = GLB_T_IN;
    GLB_X = GLB_X_IN;
    GLB_Y = GLB_Y_IN;
    GLB_Z = GLB_Z_IN;

    setup_gauge_fields();

    lprintf("INFO", 0, "Input Config dimensions given: (T, X, Y, Z) = (%d, %d, %d, %d)\n", GLB_T_IN, GLB_X_IN, GLB_Y_IN, GLB_Z_IN);
    lprintf("INFO", 0, "Input filename: %s\n", input_var.file);
    lprintf("INFO", 0, "Output Config dimensions given: (T, X, Y, Z) = (%d, %d, %d, %d)\n", GLB_T_OUT, GLB_X_OUT, GLB_Y_OUT, GLB_Z_OUT);
    lprintf("INFO", 0, "Output filename: %s\n", output_var.file);

    read_gauge_field(input_var.file);
    apply_BCs_on_represented_gauge_field(); // remove antiperiodic bcs

    int T_old = GLB_T;
    int X_old = GLB_X;
    int Y_old = GLB_Y;
    int Z_old = GLB_Z;
    size_t old_field_size = 4 * glattice.gsize_gauge;

    ipt_old = (int *)malloc(GLB_VOLUME * sizeof(int));
    memcpy(ipt_old, ipt,  GLB_VOLUME * sizeof(int));

    suNg *tmp = malloc(old_field_size * sizeof(suNg));

    memcpy(tmp, u_gauge->ptr, 4 * glattice.gsize_gauge * sizeof(suNg));

    GLB_T = GLB_T_OUT;
    GLB_X = GLB_X_OUT;
    GLB_Y = GLB_Y_OUT;
    GLB_Z = GLB_Z_OUT;

#ifdef ALLOCATE_REPR_GAUGE_FIELD
    free_gfield_f(u_gauge_f);
#endif
    GLB_VOL3 = ((long int)GLB_X) * ((long int)GLB_Y) * ((long int)GLB_Z);
    GLB_VOLUME = GLB_VOL3 * ((long int)GLB_T);

    T = GLB_T;
    X = GLB_X;
    Y = GLB_Y;
    Z = GLB_Z;

    VOL3 = GLB_VOL3;
    VOLUME = GLB_VOLUME;

    X_EXT = X;
    Y_EXT = Y;
    Z_EXT = Z;
    T_EXT = T;

    define_geometry();
    setup_gauge_fields();
    random_u(u_gauge);

    for (int t = 0; t < T; t++) {
        for (int x = 0; x < X; x++) {
            for (int y = 0; y < Y; y++) {
                for (int z = 0; z < Z; z++) {
                    int idx;
                    idx = ipt_box(T_old, X_old, Y_old, Z_old, t, x, y, z);
                    int idx_new = ipt(t, x, y, z);
                    for (int mu = 0; mu < 4; mu++) {
                        suNg in = *_4FIELD_AT_PTR(tmp, idx, mu, 0);
                        *_4FIELD_AT(u_gauge, idx_new, mu) = in;
                    }
                }
            }
        }
    }

    represent_gauge_field();
    double plaq = avr_plaquette();
    lprintf("PLAQ", 0, "Check output plaquette: %0.15e\n", plaq);
    write_gauge_field(output_var.file);
    
    finalize_process();
    return 0;
}
