param N, integer, > 0;
/* number of distinct input points (i.e. colors) */

param M, integer, > 0;
/* number of clusters */

param K, integer, > 0;
/* number of clusters */

set I := 1..N;
/* set of distinct input points */

set J := 1..M;
/* set of clusters */

param c{i in I}, > 0;
/* count of input points with color i */

param d{i in I, j in J};
/* distance from point i to center J */

var z{i in I, j in J}, >= 0, binary;
/* whether point i gets assigned to cluster j */

var y{j in J}, >= 0, binary;
/* whether j is a center */

s.t. c0: sum{j in J} y[j] <= K;
/* at most K clusters are formed */

s.t. c1{i in I}: sum{j in J} z[i, j] = 1;
/* each point gets assigned to exactly one cluster */

s.t. c2{i in I, j in J}: z[i, j] <= y[j];
/* if i is assigned to j, then j must be a center */

minimize obj: sum{i in I, j in J} d[i, j] * c[i] * z[i, j];
/* the total dissimilarity is minimized */

solve;

/* Create a set to hold the indices where z[i, j] = 1 */
set AssignedPairs := {i in I, j in J: z[i,j] = 1};

/* Output only the (i, j) pairs where z[i,j] = 1 to CSV */
table tbl_assignments{(i,j) in AssignedPairs} OUT "CSV" "cluster_assignments.csv": i, j;

/* Output the selected centers to CSV */
set SelectedCenters := {j in J: y[j] = 1};
table tbl_centers{j in SelectedCenters} OUT "CSV" "selected_centers.csv": j;

/* Output the optimal objective value as a parameter */
param objective_value := sum{i in I, j in J} d[i, j] * c[i] * z[i, j];
printf "Optimal Objective Value: %f\n", objective_value > "objective_value.csv";

end;
