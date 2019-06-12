import KMeans from './kmeans.js';
import KMeansAutoSolver from './kmeans-autosolver.js';
import example_data from './data.js';

console.log("Solving example: 2d data with 3 clusters:");
console.log("===============================================\n");


console.log("Solution for 2d data with 3 clusters:");
console.log("-------------------------------------");
const ex_1_solver = new KMeans(3, example_data.example_2d3k);
const ex_1_centroids = ex_1_solver.solve();
console.log(ex_1_centroids);
ex_1_solver.iterationLogs.forEach(log => console.log(log));

// To determine - Local and global optimum
console.log("Test 2d data with 3 clusters five times:");
console.log("----------------------------------------");
for (let i = 0; i < 5; i++) {
    ex_1_solver.reset();
    const solution = ex_1_solver.solve();
    console.log(solution);
}
console.log("");


console.log("Solving example: 3d data with 3 clusters:");
console.log("===============================================\n");
console.log("Solution for 3d data with 3 clusters:");
console.log("-------------------------------------");
const ex_2_solver = new KMeans(3, example_data.example_3d3k);
const ex_2_centroids = ex_2_solver.solve();
console.log(ex_2_centroids);
console.log("");


// AutoSolver
console.log("Solving example: 2d data with unknown clusters:");
console.log("===============================================\n");
console.log("Solution for 2d data with unknown clusters:");
console.log("-------------------------------------");
const ex_3_solver = new KMeansAutoSolver(1, 5, 5, example_data.example_2dnk);
const ex_3_solution = ex_3_solver.solve();
console.log(ex_3_solution);