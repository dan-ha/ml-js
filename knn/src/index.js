import KNN from './knn.js';
import {weight_height} from './data.js';
import decolorize from './decolorize.js';

console.log("Testing height and weight with k=5");
console.log("==========================");

const solver1 = new KNN(5, weight_height.data, weight_height.labels);

console.log("Testing a 'definitely male' point:");
console.log(solver1.predict([200, 75]));
console.log("\nTesting a 'probably male' point:");
console.log(solver1.predict([170, 70]));
console.log("\nTesting a 'totally uncertain' point:");
console.log(solver1.predict([140, 64]));
console.log("\nTesting a 'probably female' point:");
console.log(solver1.predict([130, 63]));
console.log("\nTesting a 'definitely female' point:");
console.log(solver1.predict([120, 60]));



console.log("Decolorizing " + 'lily.jpeg...');
decolorize('./files/lily.jpeg')
    .then(() => console.log('lily.jpeg decolorized'));