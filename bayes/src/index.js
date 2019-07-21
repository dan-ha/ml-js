import readline from 'readline';
import fs from 'fs';
import BayesClassifier, { simpleTokenizer } from './bayes.js';

// Utility methods
const trainer = (filename, label, classifier) => {
    return new Promise((resolve) => {
        console.log('Training ' + label + ' examples...');
        readline.createInterface({
            input: fs.createReadStream(filename)
        })
            .on('line', line => classifier.train(label, line))
            .on('close', () => {
                console.log('Finished training ' + label + ' examples...');
                resolve();
            });
    });
}

const tester = (filename, label, classifier) => {
    return new Promise((resolve) => {
        let total = 0;
        let correct = 0;
        console.log('Testing ' + label + ' examples...');
        readline.createInterface({
            input: fs.createReadStream(filename)
        })
            .on('line', line => {
                const prediction = classifier.predict(line);
                total++;
                if (prediction.label === label) {
                    correct++;
                }
            })
            .on('close', () => {
                console.log('Finished testing ' + label + ' examples.');
                const results = { total, correct };
                console.log(results);
                resolve(results);
            })
    })
}

// Train
const classifier2 = new BayesClassifier(simpleTokenizer);

Promise.all([
    trainer('./data/train_positive.txt', 'positive', classifier2),
    trainer('./data/train_negative.txt', 'negative', classifier2)
])
    .then(() => {
        console.log("Finished training. Now testing.");

        Promise.all([
            tester('./data/test_negative.txt', 'negative', classifier2),
            tester('./data/test_positive.txt', 'positive', classifier2)
        ])
            .then(results => results.reduce(
                (obj, item) => ({ total: obj.total + item.total, correct: obj.correct + item.correct }), { total: 0, correct: 0 }
            ))
            .then(results => {
                const pct = (100 * results.correct / results.total).toFixed(2) + '%';
                console.log(results);
                console.log("Test results: " + pct);
            });
    })
