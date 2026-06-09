/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
#include "src/align_tiltseries_runner_mpi.h"

void AlignTiltseriesRunnerMpi::read(int argc, char **argv)
{
    // Define a new MpiNode
    node = new MpiNode(argc, argv);

    // First read in non-parallelisation-dependent variables
    AlignTiltseriesRunner::read(argc, argv);

    // Don't put any output to screen for mpi followers
    verb = (node->isLeader()) ? 1 : 0;

    // Possibly also read parallelisation-dependent variables here

    // Print out MPI info
    printMpiNodesMachineNames(*node);
}

void AlignTiltseriesRunnerMpi::run()
{

    // Each node does part of the work
    long int my_first_tomogram, my_last_tomogram, my_nr_tomograms;
    divide_equally(idx_tomograms.size(), node->size, node->rank, my_first_tomogram, my_last_tomogram);
    my_nr_tomograms = my_last_tomogram - my_first_tomogram + 1;

    int barstep;
    if (verb > 0)
    {
        // Leader-only: show how the tomograms are split across MPI processes, so a
        // per-share progress bar reaching 100% isn't mistaken for the whole job finishing.
        if (idx_tomograms.size() > 0)
        {
            std::cout << " Distributing " << idx_tomograms.size() << " tomograms across "
                      << node->size << " MPI process(es):" << std::endl;
            for (int r = 0; r < node->size; r++)
            {
                long int rf, rl;
                divide_equally(idx_tomograms.size(), node->size, r, rf, rl);
                std::cout << "   rank " << r << ": " << (rl - rf + 1) << " tomograms ("
                          << (rf + 1) << "-" << (rl + 1) << ")" << std::endl;
            }
        }
        std::cout << " Aligning tilt series (progress for this leader's share of "
                  << my_nr_tomograms << ") ..." << std::endl;
        init_progress_bar(my_nr_tomograms);
        barstep = XMIPP_MAX(1, my_nr_tomograms / 60);
    }

    std::vector<std::string> allmicnames;
    for (long int itomo = my_first_tomogram; itomo <= my_last_tomogram; itomo++)
    {

        // Abort through the pipeline_control system
        if (pipeline_control_check_abort_job())
            MPI_Abort(MPI_COMM_WORLD, RELION_EXIT_ABORTED);

        if (do_aretomo || do_aretomo3)
        {
            executeAreTomo(idx_tomograms[itomo], node->rank);
        }
        else if (do_imod_fiducials || do_imod_patchtrack)
        {
            executeIMOD(idx_tomograms[itomo], node->rank);
        }

        // Progress is relative to this rank's share (itomo is an absolute index into idx_tomograms)
        if (verb > 0 && (itomo - my_first_tomogram) % barstep == 0)
            progress_bar(itomo - my_first_tomogram);

    }

    if (verb > 0) progress_bar(my_nr_tomograms);

    MPI_Barrier(MPI_COMM_WORLD);

}
