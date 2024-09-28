/***************************************************************************\
* Copyright (c) 2024, Sofie Martins                                         *
* All rights reserved.                                                      *
\***************************************************************************/

// Ported from boundary_conditions_core.c by Agostino Patella and
// Claudio Pica

#ifdef WITH_GPU

#include "libhr_core.h"
#include "geometry.h"

#define ipt_ext_gpu_loc(t, x, y, z) ipt_gpu_d[_lexi(T_EXT_GPU, X_EXT_GPU, Y_EXT_GPU, Z_EXT_GPU, t, x, y, z)]

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE CLOVER TERM                    */
/***************************************************************************/

__global__ void apply_cl_SF_BCs(suNfc *C, int border, int *ipt_gpu_d, int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = 0;
                suNfc c;
                _suNfc_zero(c);
                if (border > 0) {
                    index = ipt_ext_gpu_loc(border - 1, ix, iy, iz);
                    if (index != -1) {
                        write_gpu<double>(0, &c, C, index, 0, 4);
                        write_gpu<double>(0, &c, C, index, 1, 4);
                        write_gpu<double>(0, &c, C, index, 2, 4);
                        write_gpu<double>(0, &c, C, index, 3, 4);
                    }
                }

                index = ipt_ext_gpu_loc(border, ix, iy, iz);
                if (index != -1) {
                    write_gpu<double>(0, &c, C, index, 0, 4);
                    write_gpu<double>(0, &c, C, index, 1, 4);
                    write_gpu<double>(0, &c, C, index, 2, 4);
                    write_gpu<double>(0, &c, C, index, 3, 4);
                }

                index = ipt_ext_gpu_loc(border + 1, ix, iy, iz);
                if (index != -1) {
                    write_gpu<double>(0, &c, C, index, 0, 4);
                    write_gpu<double>(0, &c, C, index, 1, 4);
                    write_gpu<double>(0, &c, C, index, 2, 4);
                    write_gpu<double>(0, &c, C, index, 3, 4);
                }
            }
        }
    }
}

__global__ void apply_cl_open_BCs1(suNfc *C, int border, int *ipt_gpu_d, int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = 0;
                suNfc c;
                _suNfc_zero(c);
                if (border > 0) {
                    index = ipt_ext_gpu_loc(border - 1, ix, iy, iz);
                    if (index != -1) {
                        write_gpu<double>(0, &c, C, index, 0, 4);
                        write_gpu<double>(0, &c, C, index, 1, 4);
                        write_gpu<double>(0, &c, C, index, 2, 4);
                        write_gpu<double>(0, &c, C, index, 3, 4);
                    }
                }

                index = ipt_ext_gpu_loc(border, ix, iy, iz);
                if (index != -1) {
                    write_gpu<double>(0, &c, C, index, 0, 4);
                    write_gpu<double>(0, &c, C, index, 1, 4);
                    write_gpu<double>(0, &c, C, index, 2, 4);
                    write_gpu<double>(0, &c, C, index, 3, 4);
                }
            }
        }
    }
}

__global__ void apply_cl_open_BCs2(suNfc *C, int border, int *ipt_gpu_d, int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = 0;
                suNfc c;
                _suNfc_zero(c);
                if (border > 0) {
                    index = ipt_ext_gpu_loc(T_GPU + border, ix, iy, iz);
                    if (index != -1) {
                        write_gpu<double>(0, &c, C, index, 0, 4);
                        write_gpu<double>(0, &c, C, index, 1, 4);
                        write_gpu<double>(0, &c, C, index, 2, 4);
                        write_gpu<double>(0, &c, C, index, 3, 4);
                    }
                }

                index = ipt_ext_gpu_loc(T_GPU + border - 1, ix, iy, iz);
                if (index != -1) {
                    write_gpu<double>(0, &c, C, index, 0, 4);
                    write_gpu<double>(0, &c, C, index, 1, 4);
                    write_gpu<double>(0, &c, C, index, 2, 4);
                    write_gpu<double>(0, &c, C, index, 3, 4);
                }
            }
        }
    }
}

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE REPRESENTED GAUGE FIELD        */
/***************************************************************************/

__global__ void apply_boundary_conditions_T(suNf *g, int border, int *ipt_gpu_d, int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(2 * border, ix, iy, iz);
                if (index != -1) {
                    suNf u;
                    read_gpu<double>(0, &u, g, index, 0, 4);
                    _suNf_minus(u, u);
                    write_gpu<double>(0, &u, g, index, 0, 4);
                }
            }
        }
    }
}

__global__ void apply_boundary_conditions_X(suNf *g, int border, int *ipt_gpu_d, int tmax, int ymax, int zmax) {
    for (int it = blockDim.x * blockIdx.x + threadIdx.x; it < tmax; it += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(it, 2 * border, iy, iz);
                if (index != -1) {
                    suNf u;
                    read_gpu<double>(0, &u, g, index, 1, 4);
                    _suNf_minus(u, u);
                    write_gpu<double>(0, &u, g, index, 1, 4);
                }
            }
        }
    }
}

__global__ void apply_boundary_conditions_Y(suNf *g, int border, int *ipt_gpu_d, int tmax, int xmax, int zmax) {
    for (int it = blockDim.x * blockIdx.x + threadIdx.x; it < tmax; it += gridDim.x * blockDim.x) {
        for (int ix = blockDim.y * blockIdx.y + threadIdx.y; ix < xmax; ix += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(it, ix, 2 * border, iz);
                if (index != -1) {
                    suNf u;
                    read_gpu<double>(0, &u, g, index, 2, 4);
                    _suNf_minus(u, u);
                    write_gpu<double>(0, &u, g, index, 2, 4);
                }
            }
        }
    }
}

__global__ void apply_boundary_conditions_Z(suNf *g, int border, int *ipt_gpu_d, int tmax, int xmax, int ymax) {
    for (int it = blockDim.x * blockIdx.x + threadIdx.x; it < tmax; it += gridDim.x * blockDim.x) {
        for (int ix = blockDim.y * blockIdx.y + threadIdx.y; ix < xmax; ix += gridDim.y * blockDim.y) {
            for (int iy = blockDim.z * blockIdx.z + threadIdx.z; iy < ymax; iy += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(it, ix, iy, 2 * border);
                if (index != -1) {
                    suNf u;
                    read_gpu<double>(0, &u, g, index, 3, 4);
                    _suNf_minus(u, u);
                    write_gpu<double>(0, &u, g, index, 3, 4);
                }
            }
        }
    }
}

__global__ void apply_chiSF_ds_BT(double ds, suNf *g, int border, int *ipt_gpu_d, int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(border, ix, iy, iz);
                if (index != -1) {
                    suNf u;
                    read_gpu<double>(0, &u, g, index, 1, 4);
                    _suNf_mul(u, ds, u);
                    write_gpu<double>(0, &u, g, index, 1, 4);

                    read_gpu<double>(0, &u, g, index, 2, 4);
                    _suNf_mul(u, ds, u);
                    write_gpu<double>(0, &u, g, index, 2, 4);

                    read_gpu<double>(0, &u, g, index, 3, 4);
                    _suNf_mul(u, ds, u);
                    write_gpu<double>(0, &u, g, index, 3, 4);
                }
            }
        }
    }
}

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE FUNDAMENTAL GAUGE FIELD        */
/***************************************************************************/

__global__ void apply_gf_SF_BCs_1(suNg *g, suNg *up, suNg *dn, int border, int *ipt_gpu_d, int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(border - 1, ix, iy, iz);
                if (border > 0) {
                    if (index != -1) {
                        suNg u;
                        _suNg_unit(u);
                        write_gpu<double>(0, &u, g, index, 0, 4);
                        _suNg_unit(u);
                        write_gpu<double>(0, &u, g, index, 1, 4);
                        _suNg_unit(u);
                        write_gpu<double>(0, &u, g, index, 2, 4);
                        _suNg_unit(u);
                        write_gpu<double>(0, &u, g, index, 3, 4);
                    }
                }

                index = ipt_ext_gpu_loc(border, ix, iy, iz);
                if (index != -1) {
                    suNg u;
                    _suNg_unit(u);
                    write_gpu<double>(0, &u, g, index, 0, 4);
                    _suNg_unit(u);
                    write_gpu<double>(0, &u, g, index, 1, 4);
                    _suNg_unit(u);
                    write_gpu<double>(0, &u, g, index, 2, 4);
                    _suNg_unit(u);
                    write_gpu<double>(0, &u, g, index, 3, 4);
                }

                index = ipt_ext_gpu_loc(border + 1, ix, iy, iz);
                if (index != -1) {
                    write_gpu<double>(0, dn, g, index, 1, 4);
                    write_gpu<double>(0, dn, g, index, 2, 4);
                    write_gpu<double>(0, dn, g, index, 3, 4);
                }
            }
        }
    }
}

__global__ void apply_gf_SF_BCs_2(suNg *g, suNg *up, suNg *dn, int border, int *ipt_gpu_d, int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(T_GPU + border - 1, ix, iy, iz);
                if (index != -1) {
                    suNg u;
                    _suNg_unit(u);
                    write_gpu<double>(0, &u, g, index, 0, 4);
                    write_gpu<double>(0, up, g, index, 1, 4);
                    write_gpu<double>(0, up, g, index, 2, 4);
                    write_gpu<double>(0, up, g, index, 3, 4);
                }

                if (border > 0) {
                    int index = ipt_ext_gpu_loc(T_GPU + border, ix, iy, iz);
                    if (index != -1) {
                        suNg u;
                        _suNg_unit(u);
                        write_gpu<double>(0, &u, g, index, 0, 4);
                        _suNg_unit(u);
                        write_gpu<double>(0, &u, g, index, 1, 4);
                        _suNg_unit(u);
                        write_gpu<double>(0, &u, g, index, 2, 4);
                        _suNg_unit(u);
                        write_gpu<double>(0, &u, g, index, 3, 4);
                    }
                }
            }
        }
    }
}

__global__ void apply_SF_classical_solution(suNg *g, suNg *U, int it, int border, int *ipt_gpu_d, int xmax, int ymax,
                                            int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(it, ix, iy, iz);
                if (index != -1) {
                    suNg u;
                    _suNg_unit(u);
                    write_gpu<double>(0, &u, g, index, 0, 4);

                    write_gpu<double>(0, U, g, index, 1, 4);
                    write_gpu<double>(0, U, g, index, 2, 4);
                    write_gpu<double>(0, U, g, index, 3, 4);
                }
            }
        }
    }
}

__global__ void apply_gf_open_BCs(suNg *g, int border, int *ipt_gpu_d, int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(border, ix, iy, iz);
                if (index != -1) {
                    suNg u;
                    _suNg_zero(u);
                    write_gpu<double>(0, &u, g, index, 0, 4);
                    write_gpu<double>(0, &u, g, index, 1, 4);
                    write_gpu<double>(0, &u, g, index, 2, 4);
                    write_gpu<double>(0, &u, g, index, 3, 4);
                }
            }
        }
    }
}

__global__ void apply_gf_open_BCs2(suNg *g, int border, int *ipt_gpu_d, int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(border, ix, iy, iz);
                if (index != -1) {
                    suNg u;
                    _suNg_zero(u);
                    write_gpu<double>(0, &u, g, index, 0, 4);
                }
            }
        }
    }
}

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE MOMENTUM FIELDS                */
/***************************************************************************/

__global__ void apply_mf_Dirichlet_BCs(suNg_algebra_vector *V, int border, int *ipt_gpu_d, int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(border, ix, iy, iz);
                if (index != -1) {
                    suNg_algebra_vector v;
                    _algebra_vector_zero_g(v);
                    write_gpu<double>(0, &v, V, index, 0, 4);
                    write_gpu<double>(0, &v, V, index, 1, 4);
                    write_gpu<double>(0, &v, V, index, 2, 4);
                    write_gpu<double>(0, &v, V, index, 3, 4);
                }
            }
        }
    }
}

__global__ void apply_mf_Dirichlet_BCs_spatial(suNg_algebra_vector *V, int border, int *ipt_gpu_d, int xmax, int ymax,
                                               int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(border, ix, iy, iz);
                if (index != -1) {
                    suNg_algebra_vector v;
                    _algebra_vector_zero_g(v);
                    write_gpu<double>(0, &v, V, index, 1, 4);
                    write_gpu<double>(0, &v, V, index, 2, 4);
                    write_gpu<double>(0, &v, V, index, 3, 4);
                }
            }
        }
    }
}

__global__ void apply_mf_Dirichlet_BCs_temporal(suNg_algebra_vector *V, int border, int *ipt_gpu_d, int xmax, int ymax,
                                                int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                int index = ipt_ext_gpu_loc(border, ix, iy, iz);
                if (index != -1) {
                    suNg_algebra_vector v;
                    _algebra_vector_zero_g(v);
                    write_gpu<double>(0, &v, V, index, 0, 4);
                }
            }
        }
    }
}

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE SPINOR FIELDS                  */
/***************************************************************************/

__global__ void apply_sf_Dirichlet_BCs1(suNf_spinor *sp, int master_shift, int gsize, int border, int *ipt_gpu_d, int xmax,
                                        int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                suNf_spinor s;
                _spinor_zero_f(s);

                int index = ipt_ext_gpu_loc(border, ix, iy, iz) - master_shift;
                if (index != -1 && 0 <= index && gsize > index) { write_gpu<double>(0, &s, sp, index, 0, 1); }

                index = ipt_ext_gpu_loc(border + 1, ix, iy, iz) - master_shift;
                if (index != -1 && 0 <= index && gsize > index) { write_gpu<double>(0, &s, sp, index, 0, 1); }
            }
        }
    }
}

__global__ void apply_sf_Dirichlet_BCs2(suNf_spinor *sp, int master_shift, int gsize, int border, int *ipt_gpu_d, int xmax,
                                        int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                suNf_spinor s;
                _spinor_zero_f(s);

                int index = ipt_ext_gpu_loc(border, ix, iy, iz) - master_shift;
                if (index != -1 && 0 <= index && gsize > index) { write_gpu<double>(0, &s, sp, index, 0, 1); }
            }
        }
    }
}

__global__ void apply_sf_Dirichlet_BCs3(suNf_spinor *sp, int master_shift, int gsize, int border, int *ipt_gpu_d, int xmax,
                                        int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                suNf_spinor s;
                _spinor_zero_f(s);
                int index = ipt_ext_gpu_loc(border - 1, ix, iy, iz) - master_shift;
                if (index != -1 && 0 <= index && gsize > index) { write_gpu<double>(0, &s, sp, index, 0, 1); }
            }
        }
    }
}

__global__ void apply_sf_Dirichlet_BCs1_flt(suNf_spinor_flt *sp, int master_shift, int gsize, int border, int *ipt_gpu_d,
                                            int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                suNf_spinor_flt s;
                _spinor_zero_f(s);

                int index = ipt_ext_gpu_loc(border, ix, iy, iz) - master_shift;
                if (index != -1 && 0 <= index && gsize > index) { write_gpu<float>(0, &s, sp, index, 0, 1); }

                index = ipt_ext_gpu_loc(border + 1, ix, iy, iz) - master_shift;
                if (index != -1 && 0 <= index && gsize > index) { write_gpu<float>(0, &s, sp, index, 0, 1); }
            }
        }
    }
}

__global__ void apply_sf_Dirichlet_BCs2_flt(suNf_spinor_flt *sp, int master_shift, int gsize, int border, int *ipt_gpu_d,
                                            int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                suNf_spinor_flt s;
                _spinor_zero_f(s);

                int index = ipt_ext_gpu_loc(border, ix, iy, iz) - master_shift;
                if (index != -1 && 0 <= index && gsize > index) { write_gpu<float>(0, &s, sp, index, 0, 1); }
            }
        }
    }
}

__global__ void apply_sf_Dirichlet_BCs3_flt(suNf_spinor_flt *sp, int master_shift, int gsize, int border, int *ipt_gpu_d,
                                            int xmax, int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                suNf_spinor_flt s;
                _spinor_zero_f(s);
                int index = ipt_ext_gpu_loc(border - 1, ix, iy, iz) - master_shift;
                if (index != -1 && 0 <= index && gsize > index) { write_gpu<float>(0, &s, sp, index, 0, 1); }
            }
        }
    }
}

__global__ void apply_sf_open_BCs(suNf_spinor *sp, int master_shift, int gsize, int border, int *ipt_gpu_d, int xmax, int ymax,
                                  int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                suNf_spinor s;
                _spinor_zero_f(s);

                int index = ipt_ext_gpu_loc(border, ix, iy, iz) - master_shift;
                if (index != -1 && 0 <= index && gsize > index) { write_gpu<double>(0, &s, sp, index, 0, 1); }
            }
        }
    }
}

__global__ void apply_sf_open_BCs_flt(suNf_spinor_flt *sp, int master_shift, int gsize, int border, int *ipt_gpu_d, int xmax,
                                      int ymax, int zmax) {
    for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < xmax; ix += gridDim.x * blockDim.x) {
        for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < ymax; iy += gridDim.y * blockDim.y) {
            for (int iz = blockDim.z * blockIdx.z + threadIdx.z; iz < zmax; iz += gridDim.z * blockDim.z) {
                suNf_spinor_flt s;
                _spinor_zero_f(s);

                int index = ipt_ext_gpu_loc(border, ix, iy, iz) - master_shift;
                if (index != -1 && 0 <= index && gsize > index) { write_gpu<double>(0, &s, sp, index, 0, 1); }
            }
        }
    }
}

#endif