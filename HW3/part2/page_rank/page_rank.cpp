#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence){
    
    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs
    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    double *sol_old = (double *) malloc(sizeof(double) * numNodes);
    int *outsize = (int *) malloc(sizeof(int) * numNodes);
    bool converge_flag = false;

    // initialization
    // pre-compute outgoing size of each nodes
    for(int i = 0; i < numNodes; i++){
        solution[i] = equal_prob;
        outsize[i] = outgoing_size(g, i);
    }

    while(!converge_flag){
        double global_diff = 0, noout_score = 0;

        // assign score_new to score_old
        memcpy(sol_old, solution, sizeof(double) * numNodes);

        // compute outgoing score in this iteration
        #pragma omp parallel for reduction (+:noout_score)
        for(int i = 0; i < numNodes; i++){
            if(outsize[i] == 0){
                noout_score += (damping * sol_old[i] / numNodes);
            }
        }

        // compute score_new[vi] for all nodes vi:
        #pragma omp parallel for reduction (+:global_diff)
        for(int i = 0; i < numNodes; i++){
            const Vertex *start = incoming_begin(g, i), *end = incoming_end(g, i);
            double tmp_score = 0;

            for(const Vertex* v = start; v != end; v++){
                tmp_score += (sol_old[*v] / outsize[*v]);
            }
            
            solution[i] = ((damping * tmp_score) + (1.0 - damping) / numNodes) + noout_score;

            // compute how much per-node scores have changed, quit once algorithm has converged
            global_diff += abs(solution[i] - sol_old[i]);
        }

        converge_flag = (global_diff < convergence);
    }

    free(sol_old);
    free(outsize);

    /*
        For PP students: Implement the page rank algorithm here.  You
        are expected to parallelize the algorithm using openMP.  Your
        solution may need to allocate (and free) temporary arrays.

        Basic page rank pseudocode is provided below to get you started:

        // initialization: see example code above
        score_old[vi] = 1/numNodes;

        while (!converged) {

        // compute score_new[vi] for all nodes vi:
        score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
        score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

        score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / numNodes }

        // compute how much per-node scores have changed
        // quit once algorithm has converged

        global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
        converged = (global_diff < convergence)
        }

    */
}
