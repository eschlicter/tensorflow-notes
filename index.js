// shape tensor with 4 rows of 2 cols
const shape = [4, 2];

// feed data into the tensor
const data = tf.tensor([4, 6, 5, 9, 13, 25, 1, 57], shape);
//set variables with zeros method
const data2 = tf.variable(tf.zeros([8]));

// print data
data2.print();
//assign new 1 dimenional values
data2.assign(tf.tensor1d([4, 12, 5, 6, 56, 3, 45, 3]));

// print data
data2.print();

//two new tensors
const data3 = tf.tensor1d([4, 6, 5, 9]);
const data4 = tf.tensor1d([12, 5, 8, 6]);

data3.print();
data4.print();

// adds and multiplies
data3.add(data4).print();
data3.mul(data4).print();

function simpleAdd(input1, input2){
    // tidy is used to free up GPU memory once 
    // function returns

    return tf.tidy(() => {
        const x1 = input1;
        const x2 = input2;
        const y = x1.add(x2);
        return y;
    });
}

// new 1 dimentional tensor/arrays
const data5 = tf.tensor1d([4, 6, 5, 9]);
const data6 = tf.tensor1d([5, 4, 34, 21]);
//using the model to do input to output & print
const result = simpleAdd(data5, data6);
result.print();

// sequential model
const model = tf.sequential();

model.add(
    tf.layers.simpleRNN({
        // only needed on first layer
        inputShape: [20, 4], 
        // the number or units or neurons
        units: 20,
        // weight
        recurrentInitializer: 'GlorotNormal',
    })
);