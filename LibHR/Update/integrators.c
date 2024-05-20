/***************************************************************************\
* Copyright (c) 2008, Agostino Patella, Claudio Pica, Ari Hietanen          *
* All rights reserved.                                                      *
\***************************************************************************/

#include "update.h"
#include "libhr_core.h"
#include "io.h"
#include "memory.h"
#include "utils.h"
#include "inverters.h"

suNg_av_field *fg_momenta = NULL;
suNg_field *fg_gauge = NULL;
suNg_av_field *tmp_momenta;
suNg_field *tmp_gauge;

static void monomial_force(double dt, integrator_par *par) {
    for (int n = 0; n < par->nmon; n++) {
        monomial const *m = par->mon_list[n];
        m->update_force(dt, m->force_par);
    }
}

static void monomial_field(double dt, integrator_par *par) {
    for (int n = 0; n < par->nmon; n++) {
        monomial const *m = par->mon_list[n];
        if (m->update_field) { m->update_field(dt, m->field_par); }
    }
    if (par->next) { par->next->integrator(dt, par->next); }
}

static void monomial_fg(double dt_fg, double dt_ep, integrator_par *par) {
    zero(fg_momenta);
    copy(fg_gauge, u_gauge);

    tmp_momenta = suN_momenta;
    suN_momenta = fg_momenta;
    tmp_gauge = u_gauge;
    u_gauge = fg_gauge;

    for (int n = 0; n < par->nmon; n++) {
        monomial const *m = par->mon_list[n];
        m->update_force(dt_fg, m->force_par);
    }

    mul(suN_momenta, 1.0 / dt_fg, suN_momenta);

    for (int n = 0; n < par->nmon; n++) {
        monomial const *m = par->mon_list[n];
        if (m->update_field) { m->update_field(dt_fg, m->field_par); }
    }

    suN_momenta = tmp_momenta;

    for (int n = 0; n < par->nmon; n++) {
        monomial const *m = par->mon_list[n];
        m->update_force(dt_ep, m->force_par);
    }

    u_gauge = tmp_gauge;
}

void leapfrog_multistep(double tlen, integrator_par *par) {
    double dt = tlen / par->nsteps;
    int level = 10 + par->level * 10;

    if (par->nsteps == 0) { return; }

    lprintf("MD_INT", level, "Starting new MD trajectory with Leapfrog\n");
    lprintf("MD_INT", level, "MD parameters: level=%d tlen=%1.6f nsteps=%d => dt=%1.6f\n", par->level, tlen, par->nsteps, dt);

    monomial_force(dt / 2, par);
    monomial_field(dt, par);

    for (int n = 1; n < par->nsteps; n++) {
        monomial_force(dt, par);
        monomial_field(dt, par);
    }
    monomial_force(dt / 2, par);
}

// Implemented analogous to grid implementation
// https://github.com/paboyle/Grid
void force_gradient_multistep(double tlen, integrator_par *par) {
    if (fg_momenta == NULL) {
        fg_momenta = alloc(fg_momenta, 1, &glattice);
        fg_gauge = alloc(fg_gauge, 1, &glattice);
    }

    const double lambda = 1.0 / 6.0;
    const double chi0 = 1.0 / 72.0;
    double dt = tlen / par->nsteps;
    int level = 10 + par->level * 10;

    if (par->nsteps == 0) { return; }

    lprintf("MD_INT", level, "Starting new MD trajectory with Force Gradient\n");
    lprintf("MD_INT", level, "MD parameters: level=%d tlen=%1.6f nsteps=%d => dt=%1.6f\n", par->level, tlen, par->nsteps, dt);

    double chi = chi0 * dt * dt * dt;
    double dt_fg = 2 * chi / ((1.0 - 2.0 * lambda) * dt);
    double dt_ep = (1.0 - 2.0 * lambda) * dt;

    monomial_force(lambda * dt, par);
    monomial_field(dt / 2.0, par);
    monomial_fg(dt_fg, dt_ep, par);
    monomial_field(dt / 2.0, par);

    for (int n = 1; n < par->nsteps; n++) {
        monomial_force(2 * lambda * dt, par);
        monomial_field(dt / 2.0, par);
        monomial_fg(dt_fg, dt_ep, par);
        monomial_field(dt / 2.0, par);
    }

    monomial_force(lambda * dt, par);
}

void O2MN_multistep(double tlen, integrator_par *par) {
    const double lambda = 0.1931833275037836;
    double dt = tlen / par->nsteps;
    int level = 10 + par->level * 10;

    if (par->nsteps == 0) { return; }

    lprintf("MD_INT", level, "Starting new MD trajectory with O2MN_multistep\n");
    lprintf("MD_INT", level, "MD parameters: level=%d tlen=%1.6f nsteps=%d => dt=%1.6f\n", par->level, tlen, par->nsteps, dt);

    monomial_force(lambda * dt, par);
    monomial_field(dt / 2, par);
    monomial_force((1 - 2 * lambda) * dt, par);
    monomial_field(dt / 2, par);

    for (int n = 1; n < par->nsteps; n++) {
        monomial_force(2 * lambda * dt, par);
        monomial_field(dt / 2, par);
        monomial_force((1 - 2 * lambda) * dt, par);
        monomial_field(dt / 2, par);
    }

    monomial_force(lambda * dt, par);
}

/* 4th order  I.P. Omelyan, I.M. Mryglod, R. Folk, computer Physics Communications 151 (2003) 272-314 */
/* implementation take from "Testing and tuning symplectic integrators for Hybrid Monte Carlo algorithm in lattice QCD
Tetsuya Takaishia and Philippe de Forcrand
	 */

void O4MN_multistep(double tlen, integrator_par *par) {
    const double rho = 0.1786178958448091;
    const double theta = -0.06626458266981843;
    const double lambda = 0.7123418310626056;

    double dt = tlen / par->nsteps;
    int level = 10 + par->level * 10;

    if (par->nsteps == 0) { return; }

    lprintf("MD_INT", level, "Starting new MD trajectory with O4MN_multistep\n");
    lprintf("MD_INT", level, "MD parameters: level=%d tlen=%1.6f nsteps=%d => dt=%1.6f\n", par->level, tlen, par->nsteps, dt);

    monomial_force(rho * dt, par);
    monomial_field(lambda * dt, par);
    monomial_force(theta * dt, par);
    monomial_field((0.5 - lambda) * dt, par);
    monomial_force((1 - 2 * (theta + rho)) * dt, par);
    monomial_field((0.5 - lambda) * dt, par);
    monomial_force(theta * dt, par);
    monomial_field(lambda * dt, par);

    for (int n = 1; n < par->nsteps; n++) {
        monomial_force(2 * rho * dt, par);
        monomial_field(lambda * dt, par);
        monomial_force(theta * dt, par);
        monomial_field((0.5 - lambda) * dt, par);
        monomial_force((1 - 2 * (theta + rho)) * dt, par);
        monomial_field((0.5 - lambda) * dt, par);
        monomial_force(theta * dt, par);
        monomial_field(lambda * dt, par);
    }

    monomial_force(rho * dt, par);
}
