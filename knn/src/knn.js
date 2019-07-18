/**
 * Calculate distance between two points.
 * Point must be given as Arrays or objects with equivalent keys
 * @param {Array<number>} a 
 * @param {Array<number>} b 
 * @returns {nnumber}
 */
const distance = (a, b) => Math.sqrt(
    a.map((aPoint, i) => b[i] - aPoint)
        .reduce((sumOfSquares, diff) => sumOfSquares + (diff * diff), 0)
);

class KNN {

    constructor(k = 1, data, labels) {
        this.k = k;
        this.data = data;
        this.labels = labels;
    }

    generateDistanceMap(point) {
        /**
        * Keep at most k items in the map. 
        * Much more efficient for large sets, because this 
        * avoids storing and then sorting a million-item map.
        * This adds many more sort operations, but hopefully k is small.
        */
        const map = [];
        let maxDistanceInMap;

        for (let index = 0; index < this.data.length; index++) {

            const otherPoint = this.data[index];
            const otherPointLabel = this.labels[index];
            const thisDistance = distance(point, otherPoint);

            // Only add an item if it's closer than the farthest of the candidates
            if (!maxDistanceInMap || thisDistance < maxDistanceInMap) {
                map.push({
                    index,
                    distance: thisDistance,
                    label: otherPointLabel
                });
                // Sort the map so the closest is first
                map.sort((a, b) => a.distance < b.distance ? -1 : 1);

                // If the map became too long, drop the farthest item
                // Update this value for the next comparison
                if (map.length > this.k) {
                    map.pop();
                    maxDistanceInMap = map[map.length - 1].distance;
                }
            }
        }

        return map;
    }

    predict(point) {
        const map = this.generateDistanceMap(point);
        const votes = map.slice(0, this.k);
        const voteCounts = votes.reduce(
            (obj, vote) => Object.assign({}, obj, { [vote.label]: (obj[vote.label] || 0) + 1 }), {});
        const sortedVotes = Object.keys(voteCounts)
            .map(label => ({ label, count: voteCounts[label] }))
            .sort((a, b) => a.count > b.count ? -1 : 1);
        return {
            label: sortedVotes[0].label,
            voteCounts,
            votes
        };
    }
}

export default KNN;