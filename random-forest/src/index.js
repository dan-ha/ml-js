import { RandomForestClassifier } from 'ml-random-forest';
import crossValidation from 'ml-cross-validation';
import IrisDataset from 'ml-dataset-iris';

const loss = (expected, actual) => {
    let incorrect = 0;
    let len = expected.length;
    for (let i in expected) {
        if (expected[i] !== actual[i]) {
            incorrect++;
        }
    }
    return incorrect / len;
}

console.log("Random Forest");
console.log("======================");

// Prepare data
const data = IrisDataset.getNumbers();
const labels = IrisDataset.getClasses().map(
    (elem) => IrisDataset.getDistinctClasses().indexOf(elem)
);

const rfOptions = {
    maxFeatures: 3,
    replacement: true,
    nEstimators: 100,
    useSampleBagging: true
};

const rf = new RandomForestClassifier(rfOptions);
rf.train(data, labels);
const rfPredictions = rf.predict(data);

const confusionMatrix = crossValidation.kFold(RandomForestClassifier, data, labels, rfOptions, 10);
const accuracy = confusionMatrix.getAccuracy();

console.log('Prediction: ');
console.log(rfPredictions.join(','));
console.log('\nLoss for predictions: ' + Math.round(loss(labels, rfPredictions) * 100) + '%');
console.log('Loss for crossValidated predictions: ' + Math.round((1 - accuracy) * 100) + '%\n');
console.log(confusionMatrix);