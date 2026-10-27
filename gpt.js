// Programming from A to Z
// https://github.com/Programming-from-A-to-Z/A2Z-F26
//
// TinyGPT

class TinyGPT {
  constructor(vocabSize, contextLength, embedSize) {
    this.vocabSize = vocabSize;
    this.embedSize = embedSize;

    // 1. EMBEDDINGS
    // A vector for every token (character) in the vocabulary
    this.tokenEmbedding = tf.layers.embedding({ inputDim: vocabSize, outputDim: embedSize });
    // Positional encoding: a vector for every position in the context
    this.positionEmbedding = tf.layers.embedding({ inputDim: contextLength, outputDim: embedSize });

    // 2. ATTENTION
    // Three dense layers, one for a query, a key, and a value
    // Query: what is each position looking for?
    this.query = tf.layers.dense({ units: embedSize, useBias: false });
    // Key: what does each position have to offer?
    this.key = tf.layers.dense({ units: embedSize, useBias: false });
    // Value: what does each position pass along?
    this.value = tf.layers.dense({ units: embedSize, useBias: false });

    // The mask hides the future: position i can only look at positions 0 to i
    let mask = [];
    for (let i = 0; i < contextLength; i++) {
      // Start with every position hidden (a huge negative number)
      let row = new Array(contextLength).fill(-1000000000);
      // Then allow positions 0 to i (fill with 0)
      row.fill(0, 0, i + 1);
      mask.push(row);
    }
    this.mask = tf.tensor2d(mask);

    // 3. MULTILAYER PERCEPTRON
    // A vanilla neural network is applied to every position
    // The hidden layer is 4x bigger than the embedding size, and uses ReLU activation
    this.hidden = tf.layers.dense({ units: embedSize * 4, activation: 'relu' });
    this.mlpOutput = tf.layers.dense({ units: embedSize });

    // 4. OUTPUT
    // A score (aka "logit") for every character in the vocabulary
    this.output = tf.layers.dense({ units: vocabSize });

    this.optimizer = tf.train.adam(0.003);

    // The weights need to be initialized,
    // A trick is to run a prediction once which will cause all the weights to be created.
    tf.tidy(() => this.predict(tf.zeros([1, contextLength], 'int32')));
  }

  // Character ids [batch, length] becomes one vector per character [batch, length, embedSize]
  embed(inputs) {
    let length = inputs.shape[1];
    // Token embedding
    let tokens = this.tokenEmbedding.apply(inputs);
    // Positional embedding
    let positions = this.positionEmbedding.apply(tf.range(0, length, 1, 'int32'));
    // Add the two together, so each position has a unique vector
    return tokens.add(positions);
  }

  // Every position looks back at the earlier positions and mixes in what it finds
  attention(x) {
    let length = x.shape[1];
    // Apply the three weight matrices to the input vectors!
    let q = this.query.apply(x); // what is each position looking for?
    let k = this.key.apply(x); // what does each position have to offer?
    let v = this.value.apply(x); // what does each position pass along?

    // Score every pair of positions: the dot product of a query and a key
    let scores = tf.matMul(q, k, false, true); // [batch, length, length]
    // Scale the scores down so softmax doesn't get too extreme
    scores = scores.div(Math.sqrt(this.embedSize));
    // No peeking at the future!
    scores = scores.add(this.mask.slice([0, 0], [length, length]));
    // Softmax turns each row of scores into weights that add up to 1
    let weights = tf.softmax(scores);
    // Mix the values together according to the weights
    let output = tf.matMul(weights, v);
    return { output, weights };
  }

  // character ids -> scores for the next character at every position
  predict(inputs) {
    // Embed the inputs
    let x = this.embed(inputs);
    // Apply attention! (adding for a "residual connection")
    x = x.add(this.attention(x).output);
    // Apply the MLP (adding for a "residual connection")
    x = x.add(this.mlpOutput.apply(this.hidden.apply(x)));
    // Output scores for the next character
    return this.output.apply(x);
  }

  // One step of training. Returns the loss (lower is better).
  train(inputs, targets) {
    let loss = this.optimizer.minimize(() => {
      // Predict the next character at every position
      let logits = this.predict(inputs);
      // Get the answers for the next character (one-hot encoded)
      let answers = tf.oneHot(targets, this.vocabSize);
      // Compare and calculate loss!
      return tf.losses.softmaxCrossEntropy(answers, logits);
    }, true); // true: give back the loss so we can graph it
    // Get the loss as a number then dispose of the memory
    let value = loss.arraySync();
    loss.dispose();
    return value;
  }

  nextCharacter(context, temperature) {
    return tf.tidy(() => {
      // Turn the text into a tensor of character ids (a batch of 1)
      let inputs = tf.tensor2d([context], [1, context.length], 'int32');
      // Get the scores for the next character at every position
      let logits = this.predict(inputs);
      // Only the last position's score matters!
      // logits is [1, length, vocabSize]: take snippet 0, then the last position
      let last = logits.gather(0).gather(context.length - 1);
      // Turn the scores into probabilities.
      return tf.softmax(last.div(temperature)).dataSync();
    });
  }

  // A helper function to grab the attention weights
  attentionWeights(context) {
    return tf.tidy(() => {
      // With this input (a batch of 1)
      let inputs = tf.tensor2d([context], [1, context.length], 'int32');
      // Call the attention function and pull out the weights
      let { weights } = this.attention(this.embed(inputs));
      // We just want the weights as a plain JS array
      return weights.arraySync()[0];
    });
  }
}
