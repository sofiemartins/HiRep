/***************************************************************************\
* Copyright (c) 2008-2024, Claudio Pica, Sofie Martins                      *   
* All rights reserved.                                                      * 
\***************************************************************************/

/**
 * @file error.c
 * @brief Error handling functions
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
//#include "Geometry/setup.h"
//#include "error.h"
#include "libhr_core.h"
#include "io.h"
#include "geometry.h"

/**
 * @brief Print message to error file defined on startup.
 *
 * @param test              Condition on whether an error should be raised.
 *                          0 for no error and continue
 *                          1 for error, stop and print error message
 * @param no                Exit Code
 *                          Value smaller than zero exits immediately with code 0.
 *                          Value larger or equal then zero exits with code given
 *                          after finalizing.
 * @param name              Function name, where the error was raised
 * @param text              Error message text
 */
void error(int test, int no, const char *name, const char *text, ...) {
    if (test != 0) {
        va_list args;
        va_start(args, text);
        lprintf("ERROR", 0, "%s:\n", name);
        vlprintf("ERROR", 0, text, args);
        lprintf("ERROR", 0, "\n");
        va_end(args);
        lprintf("ERROR", 0, "Exiting program...\n");
        print_trace();
        if (no < 0) {
            exit(0);
        } else {
#ifdef WITH_MPI
            MPI_Abort(MPI_COMM_WORLD, no);
#endif
            finalize_process();
            exit(no);
        }
    }
}
