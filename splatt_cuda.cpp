#include <splatter.h>

using namespace splatter;

double calculate_volumes(double* vertices, int num_vertices,
                         int* tets, int num_tets);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    // the starcd loader will populate these
    std::vector<starcd::node_coords>	coords; // node coordinates
    std::vector<int>			tets;	// tet nodes
    std::vector<starcd::tet_cond_data>	vcond;	// tet volume conditions
    std::vector<int>			bnds;	// boundary (triangle) nodes
    std::vector<starcd::bnd_cond_data>	bcond;	// boundary conditions

    
    LLOG(0, "in the beginning....");

    // this guy is in charge of everything -- load it with the
    // standard mesh entities, default MPI configuration,
    // and initialize immediately
    part_mgr mgr(topo::num_std_entities, topo::std_entities);

    // Timers to analyze performance of routines
    timer total_timer;
    timer part_timer;


    // reroute std(err|out) to files
    printf("z%d\n", mgr.pctx().np());
    if (mgr.pctx().np() > 1)
    {
	mgr.pctx().rebind_stdio();
    }

    if (starcd::load_all("small", mgr.pctx(), coords, tets, vcond, bnds, bcond) == SPLATT_FAIL)
    {
        ERROR("load failed");
        MPI_Abort(mgr.pctx().comm(), -1);
    }

    // build node distribution out of local vector sizes
    mgr.config_node_index(coords.size());
    mgr.finalize_load();

    LLOG(0, "log/finalize complete");
    LOG_MEM(-1, "predecomp");
    LLOG(0, "number of coords: " << coords.size());
    LLOG(0, "number of tets: " << (tets.size()/4));
    LLOG(0, "number of bnds: " << (bnds.size()/3));



    double ans =calculate_volumes((double*)&coords[0], coords.size(), (int*)&tets[0], tets.size()/4);
    LLOG(0, "the answer: " << ans);
}
