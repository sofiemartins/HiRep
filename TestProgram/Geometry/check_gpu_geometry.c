/*******************************************************************************
*
* Check that the GPU reading and writing functions defined in suN.h 
* are bijective.
*
* NOCOMPILE = !WITH_GPU
* 
*******************************************************************************/

#include "libhr.h"

// TODO: get gaussian spinor fields to work with MPI-CUDA
// TODO: &glat_even, &glat_odd
// TODO: Test for n > 1
// TODO: Get scalar fields to pass

// Double precision
int test_write_read_gauge_field();
int test_write_read_gauge_field_f();
int test_write_read_suNg_scalar_field();
int test_write_read_avfield();
int test_write_read_gtransf();
int test_write_read_clover_term();
int test_write_read_clover_force();
int test_write_read_spinor_field_f();
int test_write_read_spinor_field_f_vector_wise();
int test_write_read_sfield();
int test_write_read_clover_ldl();

// Single precision
int test_write_read_gauge_field_flt();
int test_write_read_gauge_field_f_flt();
int test_write_read_spinor_field_f_flt();
int test_write_read_spinor_field_f_flt_vector_wise();

int main(int argc, char *argv[]) 
{
    // Init
    int return_val = 0;

    // Setup process and communication
    logger_map("DEBUG", "debug");
    setup_process(&argc, &argv);
    test_setup();

    // Double Precision Tests
    return_val += test_write_read_gauge_field();
    return_val += test_write_read_gauge_field_f();
    return_val += test_write_read_suNg_scalar_field();
    return_val += test_write_read_avfield();
    return_val += test_write_read_gtransf();
    return_val += test_write_read_clover_ldl();
    return_val += test_write_read_clover_term();
    return_val += test_write_read_clover_force();
    
    return_val += test_write_read_spinor_field_f();
    return_val += test_write_read_spinor_field_f_vector_wise();
    return_val += test_write_read_sfield();
    
    //Single Precision Tests
    return_val += test_write_read_gauge_field_flt();
    return_val += test_write_read_gauge_field_f_flt();
    return_val += test_write_read_spinor_field_f_flt();
    return_val += test_write_read_spinor_field_f_flt_vector_wise();

    // Finalize and return
    finalize_process();
    return return_val;
}

int test_write_read_gauge_field()
{
    lprintf("INFO", 0, " ======= TEST GAUGE FIELD ======= ");
    int return_val = 0;
    suNg_field *in, *gpu_format, *out;

    in = alloc_gfield(&glattice);
    out = alloc_gfield(&glattice);
    gpu_format = alloc_gfield(&glattice);

    random_u(in); 
    lprintf("SANITY CHECK", 0, "[In field norm unequal zero: %0.2e]\n", sqnorm_gfield_cpu(in));

    suNg *in_mat, *block_start, *out_mat;   
    int stride = 0; 
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _4FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_end[ixp] +1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            for (int comp = 0; comp < 4; comp++) 
            {
                in_mat = _4FIELD_AT(in, ix, comp);
                out_mat = _4FIELD_AT(out, ix, comp);
                write_gpu_suNg(stride, (*in_mat), block_start, ix_loc, comp);
                read_gpu_suNg(stride, (*out_mat), block_start, ix_loc, comp);
            }
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_gfield_cpu(in));
    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", sqnorm_gfield_cpu(out));
    sub_assign_gfield_cpu(out, in);
    double diff_norm = sqnorm_gfield_cpu(out);

    check_diff_norm_zero(diff_norm);

    free_gfield(in);
    free_gfield(out);
    free_gfield(gpu_format);
    return return_val;
}

int test_write_read_gauge_field_flt()
{
    lprintf("INFO", 0, " ======= TEST GAUGE FIELD SINGLE PRECISION ======= ");
    int return_val = 0;
    suNg_field_flt *in, *gpu_format, *out;

    in = alloc_gfield_flt(&glattice);
    out = alloc_gfield_flt(&glattice);
    gpu_format = alloc_gfield_flt(&glattice);

    random_gfield_flt_cpu(in);
    lprintf("SANITY CHECK", 0, "[In field norm unequal zero: %0.2e]\n", sqnorm_gfield_flt_cpu(in));

    suNg_flt *in_mat, *block_start, *out_mat;  
    int stride = 0;  
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _4FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_start[ixp] +1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            for (int comp = 0; comp < 4; comp++) 
            {
                in_mat = _4FIELD_AT(in, ix, comp);
                out_mat = _4FIELD_AT(out, ix, comp);
                write_gpu_suNg_flt(stride, (*in_mat), block_start, ix_loc, comp);
                read_gpu_suNg_flt(stride, (*out_mat), block_start, ix_loc, comp);
            }
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_gfield_flt_cpu(in));
    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", sqnorm_gfield_flt_cpu(out));
    sub_assign_gfield_flt_cpu(out, in);
    double diff_norm = sqnorm_gfield_flt_cpu(out);
    check_diff_norm_zero(diff_norm);

    free_gfield_flt(in);
    free_gfield_flt(out);
    free_gfield_flt(gpu_format);
    return return_val;
}

int test_write_read_gauge_field_f()
{
    lprintf("INFO", 0, " ======= TEST GAUGE FIELD FUNDAMENTAL REP ======= ");
    int return_val = 0;
    suNf_field *in, *gpu_format, *out;

    in = alloc_gfield_f(&glattice);
    out = alloc_gfield_f(&glattice);
    gpu_format = alloc_gfield_f(&glattice);

    random_u_f(in);
    lprintf("SANITY CHECK", 0, "[In field norm unequal zero: %0.2e]\n", sqnorm_gfield_f_cpu(in));

    suNf *in_mat, *block_start, *out_mat;
    int stride = 0;
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _4FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_start[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            for (int comp = 0; comp < 4; comp++) 
            {
                in_mat = _4FIELD_AT(in, ix, comp);
                out_mat = _4FIELD_AT(out, ix, comp);
                write_gpu_suNf(stride, (*in_mat), block_start, ix_loc, comp);
                read_gpu_suNf(stride, (*out_mat), block_start, ix_loc, comp);
            }
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_gfield_f_cpu(in));
    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", sqnorm_gfield_f_cpu(out));
    sub_assign_gfield_f_cpu(out, in);
    double diff_norm = sqnorm_gfield_f_cpu(out);

    check_diff_norm_zero(diff_norm);

    free_gfield_f(in);
    free_gfield_f(out);
    free_gfield_f(gpu_format);
    return return_val;
}

int test_write_read_gauge_field_f_flt()
{
    lprintf("INFO", 0, " ======= TEST GAUGE FIELD FUNDAMENTAL REP ======= ");
    int return_val = 0;
    suNf_field_flt *in, *gpu_format, *out;

    in = alloc_gfield_f_flt(&glattice);
    out = alloc_gfield_f_flt(&glattice);
    gpu_format = alloc_gfield_f_flt(&glattice);

    random_gfield_f_flt_cpu(in);
    lprintf("SANITY CHECK", 0, "[In field norm unequal zero: %0.2e]\n", sqnorm_gfield_f_flt_cpu(in));

    suNf_flt *in_mat, *block_start, *out_mat;
    int stride = 0;
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _4FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_start[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            for (int comp = 0; comp < 4; comp++) 
            {
                in_mat = _4FIELD_AT(in, ix, comp);
                out_mat = _4FIELD_AT(out, ix, comp);
                write_gpu_suNf_flt(stride, (*in_mat), block_start, ix_loc, comp);
                read_gpu_suNf_flt(stride, (*out_mat), block_start, ix_loc, comp);
            }
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_gfield_f_flt_cpu(in));
    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", sqnorm_gfield_f_flt_cpu(out));
    sub_assign_gfield_f_flt_cpu(out, in);
    double diff_norm = sqnorm_gfield_f_flt_cpu(out);

    check_diff_norm_zero(diff_norm);

    free_gfield_f_flt(in);
    free_gfield_f_flt(out);
    free_gfield_f_flt(gpu_format);
    return return_val;
}

int test_write_read_spinor_field_f() 
{
    lprintf("INFO", 0, " ======= TEST SPINOR FIELD ======= ");
    int return_val = 0;
    spinor_field *in, *gpu_format, *out;

    in = alloc_spinor_field_f(1, &glattice);
    gpu_format = alloc_spinor_field_f(1, &glattice);
    out = alloc_spinor_field_f(1, &glattice);

    gaussian_spinor_field(in);

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", spinor_field_sqnorm_f_cpu(in));

    suNf_spinor *in_spinor, *block_start, *out_spinor;
    int stride = 0;
    _PIECE_FOR(in->type, ixp)
    {
        block_start = _FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_start[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            in_spinor = _FIELD_AT(in, ix);
            out_spinor = _FIELD_AT(out, ix);
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            write_gpu_suNf_spinor(stride, (*in_spinor), block_start, ix_loc, 0);
            read_gpu_suNf_spinor(stride, (*out_spinor), block_start, ix_loc, 0);
        } 
    }

    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", spinor_field_sqnorm_f_cpu(out));
    spinor_field_sub_assign_f_cpu(out, in);
    double diff_norm = spinor_field_sqnorm_f_cpu(out);
    check_diff_norm_zero(diff_norm);

    free_spinor_field_f(in);
    free_spinor_field_f(gpu_format);
    free_spinor_field_f(out);
    return return_val;
}

int test_write_read_spinor_field_f_vector_wise() 
{
    lprintf("INFO", 0, " ======= TEST SPINOR FIELD II ======= ");
    int return_val = 0;
    spinor_field *in, *gpu_format, *out;

    in = alloc_spinor_field_f(1, &glattice);
    gpu_format = alloc_spinor_field_f(1, &glattice);
    out = alloc_spinor_field_f(1, &glattice);

    gaussian_spinor_field(in);

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", spinor_field_sqnorm_f_cpu(in));

    suNf_spinor *in_spinor, *block_start, *out_spinor;
    int stride = 0;
    _PIECE_FOR(in->type, ixp)
    {
        block_start = _FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_start[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            in_spinor = _FIELD_AT(in, ix);
            out_spinor = _FIELD_AT(out, ix);
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            for (int comp = 0; comp < 4; ++comp) 
            {
                write_gpu_suNf_vector(stride, (*in_spinor).c[comp], block_start, ix_loc, comp);
                read_gpu_suNf_vector(stride, (*out_spinor).c[comp], block_start, ix_loc, comp);
            }
        } 
    }

    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", spinor_field_sqnorm_f_cpu(out));
    spinor_field_sub_assign_f_cpu(out, in);
    double diff_norm = spinor_field_sqnorm_f_cpu(out);
    check_diff_norm_zero(diff_norm);

    free_spinor_field_f(in);
    free_spinor_field_f(gpu_format);
    free_spinor_field_f(out);
    return return_val;
}

int test_write_read_spinor_field_f_flt() 
{
    lprintf("INFO", 0, " ======= TEST SPINOR FIELD SINGLE PRECISION ======= \n");
    int return_val = 0;
    spinor_field_flt *in, *gpu_format, *out;

    in = alloc_spinor_field_f_flt(1, &glattice);
    gpu_format = alloc_spinor_field_f_flt(1, &glattice);
    out = alloc_spinor_field_f_flt(1, &glattice);

    gaussian_spinor_field_flt(in);

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", spinor_field_sqnorm_f_flt_cpu(in));

    suNf_spinor_flt *in_spinor, *block_start, *out_spinor;
    int stride = 0;
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_start[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            in_spinor = _FIELD_AT(in, ix);
            out_spinor = _FIELD_AT(out, ix);
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            write_gpu_suNf_spinor_flt(stride, (*in_spinor), block_start, ix_loc, 0);
            read_gpu_suNf_spinor_flt(stride, (*out_spinor), block_start, ix_loc, 0);
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", spinor_field_sqnorm_f_flt_cpu(out));
    spinor_field_sub_assign_f_flt_cpu(out, in);
    double diff_norm = spinor_field_sqnorm_f_flt_cpu(out);
    check_diff_norm_zero(diff_norm);

    free_spinor_field_f_flt(in);
    free_spinor_field_f_flt(gpu_format);
    free_spinor_field_f_flt(out);
    return return_val;
}

int test_write_read_spinor_field_f_flt_vector_wise() 
{
    lprintf("INFO", 0, " ======= TEST SPINOR FIELD SINGLE PRECISION II ======= ");
    int return_val = 0;
    spinor_field_flt *in, *gpu_format, *out;

    in = alloc_spinor_field_f_flt(1, &glattice);
    gpu_format = alloc_spinor_field_f_flt(1, &glattice);
    out = alloc_spinor_field_f_flt(1, &glattice);

    gaussian_spinor_field_flt(in);

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", spinor_field_sqnorm_f_flt_cpu(in));

    suNf_spinor_flt *in_spinor, *block_start, *out_spinor;
    int stride = 0;
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_start[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            in_spinor = _FIELD_AT(in, ix);
            out_spinor = _FIELD_AT(out, ix);
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            for (int comp = 0; comp < 4; ++comp)
            {
                write_gpu_suNf_vector_flt(stride, (*in_spinor).c[comp], block_start, ix_loc, comp);
                read_gpu_suNf_vector_flt(stride, (*out_spinor).c[comp], block_start, ix_loc, comp);
            }
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", spinor_field_sqnorm_f_flt_cpu(out));
    spinor_field_sub_assign_f_flt_cpu(out, in);
    double diff_norm = spinor_field_sqnorm_f_flt_cpu(out);
    check_diff_norm_zero(diff_norm);

    free_spinor_field_f_flt(in);
    free_spinor_field_f_flt(gpu_format);
    free_spinor_field_f_flt(out);
    return return_val;
}

int test_write_read_sfield() 
{
    lprintf("INFO", 0, " ======= TEST SCALAR FIELD ======= ");
    int return_val = 0;
    scalar_field *in, *gpu_format, *out;

    in = alloc_sfield(1, &glattice);
    gpu_format = alloc_sfield(1, &glattice);
    out = alloc_sfield(1, &glattice);

    random_sfield_cpu(in);
    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_sfield_cpu(in));

    double *in_site, *block_start, *out_site;
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _FIELD_BLK(gpu_format, ixp);
        // int stride = in->type->master_end[ixp] - in->type->master_start[ixp] + 1; //double doesn't use stride
        _SITE_FOR(in->type, ixp, ix) 
        {
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            in_site = _FIELD_AT(in, ix);
            out_site = _FIELD_AT(out, ix);
            write_gpu_double(stride, (*in_site), block_start, ix_loc, 0);
            read_gpu_double(stride, (*out_site), block_start, ix_loc, 0);
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_sfield_cpu(in));
    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", sqnorm_sfield_cpu(out));
    sub_assign_sfield_cpu(out, in);
    double diff_norm = sqnorm_sfield_cpu(out);

    check_diff_norm_zero(diff_norm);

    free_sfield(in);
    free_sfield(gpu_format);
    free_sfield(out);
    return return_val;
}

int test_write_read_suNg_scalar_field() 
{
    lprintf("INFO", 0, " ======= TEST SU(NG) SCALAR FIELD ======= ");
    int return_val = 0;
    suNg_scalar_field *in, *gpu_format, *out;

    in = alloc_suNg_scalar_field(&glattice);
    out = alloc_suNg_scalar_field(&glattice);
    gpu_format = alloc_suNg_scalar_field(&glattice);

    random_suNg_scalar_field_cpu(in);
    lprintf("SANITY CHECK", 0, "[In field norm unequal zero: %0.2e]\n", sqnorm_suNg_scalar_field_cpu(in));

    suNg_vector *in_vec, *block_start, *out_vec;
    int stride = 0;
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_end[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            in_vec = _FIELD_AT(in, ix);
            out_vec = _FIELD_AT(out, ix);
            write_gpu_suNg_vector(stride, (*in_vec), block_start, ix_loc, 0);
            read_gpu_suNg_vector(stride, (*out_vec), block_start, ix_loc, 0);
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_suNg_scalar_field_cpu(in));
    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", sqnorm_suNg_scalar_field_cpu(out));
    sub_assign_suNg_scalar_field_cpu(out, in);
    double diff_norm = sqnorm_suNg_scalar_field_cpu(out);

    check_diff_norm_zero(diff_norm);

    free_suNg_scalar_field(in);
    free_suNg_scalar_field(gpu_format);
    free_suNg_scalar_field(out);
    return return_val;
}

int test_write_read_avfield() 
{
    lprintf("INFO", 0, " ======= TEST AVFIELD ======= ");
    int return_val = 0;
    suNg_av_field *in, *gpu_format, *out;

    in = alloc_avfield(&glattice);
    out = alloc_avfield(&glattice);
    gpu_format = alloc_avfield(&glattice);

    random_avfield_cpu(in);
    lprintf("SANITY CHECK", 0, "[In field norm unequal zero: %0.2e]\n", sqnorm_avfield_cpu(in));

    suNg_algebra_vector *in_mat, *block_start, *out_mat;
    int stride = 0;
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _4FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_end[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            for (int comp = 0; comp < 4; comp++) 
            {
                in_mat = _4FIELD_AT(in, ix, comp);
                out_mat = _4FIELD_AT(out, ix, comp);
                write_gpu_suNg_algebra_vector(stride, (*in_mat), block_start, ix_loc, comp);
                read_gpu_suNg_algebra_vector(stride, (*out_mat), block_start, ix_loc, comp);
            }
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_avfield_cpu(in));
    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", sqnorm_avfield_cpu(out));
    sub_assign_avfield_cpu(out, in);
    double diff_norm = sqnorm_avfield_cpu(out);

    check_diff_norm_zero(diff_norm);

    free_avfield(in);
    free_avfield(out);
    free_avfield(gpu_format);
    return return_val;
}

int test_write_read_gtransf() 
{
    lprintf("INFO", 0, " ======= TEST GTRANSF ======= ");
    int return_val = 0;
    suNg_field *in, *gpu_format, *out;

    in = alloc_gtransf(&glattice);
    out = alloc_gtransf(&glattice);
    gpu_format = alloc_gtransf(&glattice);

    random_gtransf_cpu(in);
    lprintf("SANITY CHECK", 0, "[In field norm unequal zero: %0.2e]\n", sqnorm_gtransf_cpu(in));

    suNg *in_mat, *block_start, *out_mat;
    int stride = 0;
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_end[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            in_mat = _FIELD_AT(in, ix);
            out_mat = _FIELD_AT(out, ix);
            write_gpu_suNg(stride, (*in_mat), block_start, ix_loc, 0);
            read_gpu_suNg(stride, (*out_mat), block_start, ix_loc, 0);
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_gtransf_cpu(in));
    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", sqnorm_gtransf_cpu(out));
    sub_assign_gtransf_cpu(out, in);
    double diff_norm = sqnorm_gtransf_cpu(out);

    check_diff_norm_zero(diff_norm);

    free_gtransf(in);
    free_gtransf(out);
    free_gtransf(gpu_format);
    return return_val;
}

int test_write_read_clover_term() 
{
    lprintf("INFO", 0, " ======= TEST CLOVER TERM ======= ");
    int return_val = 0;
    suNfc_field *in, *gpu_format, *out;

    in = alloc_clover_term(&glattice);
    out = alloc_clover_term(&glattice);
    gpu_format = alloc_clover_term(&glattice);

    random_clover_term_cpu(in);
    lprintf("SANITY CHECK", 0, "[In field norm unequal zero: %0.2e]\n", sqnorm_clover_term_cpu(in));

    suNfc *in_mat, *block_start, *out_mat;
    int stride = 0;
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _4FIELD_BLK(gpu_format, ixp);
        stride = in->type->master_end[ixp] - in->type->master_end[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            for (int comp = 0; comp < 4; comp++) 
            {
                in_mat = _4FIELD_AT(in, ix, comp);
                out_mat = _4FIELD_AT(out, ix, comp);
                write_gpu_suNfc(stride, (*in_mat), block_start, ix_loc, comp);
                read_gpu_suNfc(stride, (*out_mat), block_start, ix_loc, comp);
            }
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_clover_term_cpu(in));
    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", sqnorm_clover_term_cpu(out));
    sub_assign_clover_term_cpu(out, in);
    double diff_norm = sqnorm_clover_term_cpu(out);

    check_diff_norm_zero(diff_norm);

    free_clover_term(in);
    free_clover_term(out);
    free_clover_term(gpu_format);
    return return_val;
}

int test_write_read_clover_force() 
{
    lprintf("INFO", 0, " ======= TEST CLOVER FORCE ======= ");
    int return_val = 0;
    suNf_field *in, *gpu_format, *out;

    in = alloc_clover_force(&glattice);
    out = alloc_clover_force(&glattice);
    gpu_format = alloc_clover_force(&glattice);

    random_clover_force_cpu(in);
    lprintf("SANITY CHECK", 0, "[In field norm unequal zero: %0.2e]\n", sqnorm_clover_force_cpu(in));

    suNf *in_mat, *block_start, *out_mat;
    int stride = 0;
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _DFIELD_BLK(gpu_format, ixp, 6);
        stride = in->type->master_end[ixp] - in->type->master_end[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            for (int comp = 0; comp < 6; comp++) 
            {
                in_mat = _6FIELD_AT(in, ix, comp);
                out_mat = _6FIELD_AT(out, ix, comp);
                write_gpu_suNf(stride, (*in_mat), block_start, ix_loc, comp);
                read_gpu_suNf(stride, (*out_mat), block_start, ix_loc, comp);
            }
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_clover_force_cpu(in));
    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", sqnorm_clover_force_cpu(out));
    sub_assign_clover_force_cpu(out, in);
    double diff_norm = sqnorm_clover_force_cpu(out);

    check_diff_norm_zero(diff_norm);

    free_clover_force(in);
    free_clover_force(out);
    free_clover_force(gpu_format);
    return return_val;
}

int test_write_read_clover_ldl() 
{
    lprintf("INFO", 0, " ======= TEST LDL FIELD ======= ");
    int return_val = 0;
    ldl_field *in, *gpu_format, *out;

    in = alloc_clover_ldl(&glattice);
    out = alloc_clover_ldl(&glattice);
    gpu_format = alloc_clover_ldl(&glattice);

    random_clover_ldl_cpu(in);
    lprintf("SANITY CHECK", 0, "[In field norm unequal zero: %0.2e]\n", sqnorm_clover_ldl_cpu(in)); // sqnorm Not implemented

    ldl_t *site_in, *block_start, *site_out;
    int stride = 0;
    _PIECE_FOR(in->type, ixp) 
    {
        block_start = _DFIELD_BLK(gpu_format, ixp, 0);
        stride = in->type->master_end[ixp] - in->type->master_end[ixp] + 1;
        _SITE_FOR(in->type, ixp, ix) 
        {
            int ix_loc = _GPU_IDX_TO_LOCAL(in, ix, ixp);
            site_in = _FIELD_AT(in, ix);
            site_out = _FIELD_AT(out, ix);
            write_gpu_ldl_t(stride, (*site_in), block_start, ix_loc, 0);
            read_gpu_ldl_t(stride, (*site_out), block_start, ix_loc, 0);
        }
    }

    lprintf("SANITY CHECK", 0, "[Sanity check in field norm unequal zero: %0.2e]\n", sqnorm_clover_ldl_cpu(in));
    lprintf("SANITY CHECK", 0, "[Sanity check out field norm unequal zero: %0.2e]\n", sqnorm_clover_ldl_cpu(out));

    sub_assign_clover_ldl_cpu(out, in);
    double diff_norm = sqnorm_clover_ldl_cpu(out);
    check_diff_norm_zero(diff_norm);

    free_clover_ldl(in);
    free_clover_ldl(out);
    free_clover_ldl(gpu_format);
    return return_val;
}
