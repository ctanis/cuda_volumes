#include <vector>
#include <stdio.h>


double calculate_volumes(double* vertices, int num_vertices,
                         int* tets, int num_tets);

using namespace std;

int main(int argc, char** argv)
{
    vector<double> coords;
    vector<int> tets;

    int i;
    double c[] = {0,0,0,1,0,1};//,0,1,1,2,2,2
    for(i=0;i<6;i++)
      coords.push_back(c[i]);

    int t[] = {50,1,2,3};
    for(i=0;i<4;i++)
      tets.push_back(t[i]);

//calculate volume device side
    double device_vol =calculate_volumes((double*)&coords[0], coords.size(), (int*)&tets[0], tets.size()/4);
    printf("woo %f\n", device_vol);
}


