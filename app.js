// Data source: http://mlr.cs.umass.edu/ml/datasets/Wine

// 1) Alcohol (divide by 100)
// 2) Malic acid (ph 1-14, divide by 14)
// 3) Ash (divide by 3?)
// 4) Alcalinity of ash (divide by 100)
// 5) Magnesium (divide by 1000)
// 6) Total phenols
// 7) Flavanoids
// 8) Nonflavanoid phenols
// 9) Proanthocyanins
// 10)Color intensity
// 11)Hue
// 12)OD280/OD315 of diluted wines
// 13)Proline

let synaptic = require('synaptic');
let Architect = synaptic.Architect
let fs = require('fs');

console.log("ML practice ;)");

let trainingSet = []

let array = fs.readFileSync('trainingData.txt').toString().split("\n");
for(let i = 0; i<array.length-1;i++) {
    let split = array[i].split(',')
    let outputArray = []
    if(split[0] == 1) {
      outputArray = [1,0,0]
    } else if (split[0] == 2) {
      outputArray = [0,1,0]
    } else {
      outputArray = [0,0,1]
    }

    let obj = {
      input: [Math.round(split[1])/100,split[2]/14,split[3]/3,Math.round(split[4])/100,split[5]/1000, split[6]/10],
      output: outputArray
    }
    // console.log(obj)
    trainingSet.push(obj)
}

console.log(`training with ${trainingSet.length} data points`)

let wineNet = new Architect.Perceptron(6, 6, 3);
let trainingOptions = {
  rate: .0003,
  iterations: 100000,
  error: .005,
  schedule: {
    every: 10000,
    do: data => console.log("error", data.error,"iterations", data.iterations, "rate", data.rate)
  }
}

wineNet.trainer.train(trainingSet, trainingOptions);

let testData = []

let testArray = fs.readFileSync('testingData.txt').toString().split("\n");
for(let i = 0; i<testArray.length-1;i++) {
    let split = testArray[i].split(',')
    let outputArray = []
    if(split[0] == 1) {
      outputArray = [1,0,0]
    } else if (split[0] == 2) {
      outputArray = [0,1,0]
    } else {
      outputArray = [0,0,1]
    }
    let obj = {
      input: [Math.round(split[1])/100,split[2]/14,split[3]/3,Math.round(split[4])/100,split[5]/1000,split[6]/10],
      output: outputArray
    }
    // console.log(obj)
    testData.push(obj)
}
console.log(`testing with ${testData.length} data points`)

let probabilityError = 0
let classificationAccuracy = 0

testData.forEach(data => {
  let activation = data.input;
  let result = wineNet.activate(activation)
  let error1 = Math.pow((data.output[0]-result[0]),2)
  let error2 = Math.pow((data.output[1]-result[1]),2)
  let error3 = Math.pow((data.output[2]-result[2]),2)
  let totalError = error1+error2+error3
  probabilityError += totalError
  // console.log(`probability error: ${totalError}, probability accuracy:${(1-totalError)}`)
  checkIfCorrect(result, data.output)
})

function checkIfCorrect(predicted, actual) {
  // console.log(predicted, actual)
  let correctSpot = actual.indexOf(1)
  let highestPredicted = predicted.indexOf(Math.max(...predicted))

  // console.log(correctSpot, highestPredicted)
  if(correctSpot === highestPredicted) {
    classificationAccuracy++
    return 'correct'
  } else {
    return 'incorrect'
  }
}

console.log(`Sum of square means error: ${probabilityError}`)
console.log(`Classification correctness: ${classificationAccuracy/32}`)
