#include <splatter.h>
#include <splatt_geom.h>

using namespace splatter;

double copyit(double* vertices, int num_vertices);

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
    if (mgr.pctx().np() > 1)
    {
	mgr.pctx().rebind_stdio();
    }

    char load_file[50] = "small";

    if (starcd::load_all(load_file, mgr.pctx(), coords, tets, vcond, bnds, bcond) == SPLATT_FAIL)
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


//calculate volume host side
    double host_vol = 0;
    unsigned int i=0; while(i<coords.size()){
    	host_vol += coords[i][1]; //the y value
	i++;
    }
    LLOG(0, "host answer:" << host_vol);

//calculate volume device side
    double device_vol =copyit((double*)&coords[0], coords.size());
    LLOG(0, "device answer: " << device_vol); 
}


