import SVM from 'libsvm-js/asm';
import IrisDataset from 'ml-dataset-iris';

// Prepare data
const data = IrisDataset.getNumbers();
const labels = IrisDataset.getClasses().map(
    (elem) => IrisDataset.getDistinctClasses().indexOf(elem)
);

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

console.log('Support vector Machine');
console.log('=========================');

// SVM with Radial Basis function Kernel
const svm = new SVM({
    kernel: SVM.KERNEL_TYPES.RBF,
    type: SVM.SVM_TYPES.C_SVC,
    gamma: 0.25,
    cost: 1,
    quiet: true
});

svm.train(data, labels);

const svmPredictions = svm.predict(data);
const svmCvPredictions = svm.crossValidation(data, labels, 5);

console.log('Loss for predictions: ' + Math.round(loss(labels, svmPredictions) * 100) + '%');
console.log('Loss for crossvalidation predictions: ' + Math.round(loss(labels, svmCvPredictions) * 100) + '%');




